from typing import Callable, Iterable, Literal, Optional, List, Union, Tuple
from collections import defaultdict
from enum import Enum
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import logsumexp
from torch.distributions import Normal, Bernoulli, Categorical
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Decoder, Encoder, _base_components, one_hot, FCLayers

from torchmetrics import Accuracy, F1Score

torch.backends.cudnn.benchmark = True

from .utils import *

from scvi.module._classifier import Classifier

dim_indices = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# all tensors are samples_count x cov_count

# for hyperparameter tuning
# from ray import tune
from scvi._decorators import classproperty
from scvi.autotune._types import Tunable, TunableMixin


class fairVAE(BaseModuleClass):
    """Fair Variational auto-encoder module.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent_shared
        Dimensionality of the shared latent space (Z_{-s})
    n_latent_attribute
        Dimensionality of the latent space for each sensitive attributes (Z_{s_i})
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: Tunable[int] = 128,
            n_latent_shared: Tunable[int] = 10,
            n_latent_attribute: Tunable[int] = 10,
            n_layers: Tunable[int] = 1,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input)).to(device)

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders

        n_input_encoder = n_input
        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        self.zs_num = len(self.n_cat_list)

        self.z_encoders_list = nn.ModuleList(
            [
                Encoder(
                    n_input_encoder,
                    n_latent_shared,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
            ]
        )

        self.z_encoders_list.extend(
            [
                Encoder(
                    n_input_encoder,
                    n_latent_attribute,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.z_prior_encoders_list = nn.ModuleList(
            [
                Encoder(
                    0,
                    n_latent_attribute,
                    n_cat_list=[self.n_cat_list[k]],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        # Decoders

        self.x_decoders_list = nn.ModuleList(
            [
                DecoderSCVI(
                    n_latent_shared,
                    n_input,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
            ]
        )

        self.x_decoders_list.extend(
            [
                DecoderSCVI(
                    n_latent_attribute,
                    n_input,
                    n_cat_list=[self.n_cat_list[i] for i in range(len(self.n_cat_list)) if i != k],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.n_latent = n_latent_shared + n_latent_attribute * self.zs_num

        self.s_classifiers_list = nn.ModuleList([])
        for i in range(self.zs_num):
            self.s_classifiers_list.append(
                Classifier(
                    n_input=n_latent_attribute,
                    n_labels=self.n_cat_list[i],
                ).to(device)
            )

    def set_require_grad(self, mode):
        if TRAIN_MODE.CLASSIFICATION not in mode:
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = False
        else:
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = True

    def _get_inference_input(self, tensors):

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY

        cat_covs = tensors[cat_key][0]
        cat_covs_cf = tensors[cat_key][1:]

        x = tensors[REGISTRY_KEYS.X_KEY][0]
        x_cf = tensors[REGISTRY_KEYS.X_KEY][1:]

        input_dict = {
            "x": x,
            "x_cf": x_cf,
            "cat_covs": cat_covs,
            "cat_covs_cf": cat_covs_cf,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        input_dict = {
            "z_shared": inference_outputs["z_shared"],
            "zs": inference_outputs["zs"],  # a list of all zs
            "zs_cf": inference_outputs["zs_cf"],  # a list of all zs_cf
            "library": inference_outputs["library"],
            "library_cf": inference_outputs["library_cf"],
            "cat_covs": inference_outputs["cat_covs"],
        }
        return input_dict

    @auto_move_data
    def inference(self, x, x_cf,
                  cat_covs, cat_covs_cf,
                  nullify_cat_covs_indices: Optional[List[int]] = None,
                  nullify_shared: Optional[bool] = False,
                  ):

        nullify_cat_covs_indices = [] if nullify_cat_covs_indices is None else nullify_cat_covs_indices

        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        library_cf = [torch.log(x_c.sum(1)).unsqueeze(1) for x_c in x_cf]
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)

        # z_shared

        qz_shared, z_shared = self.z_encoders_list[0](x_, *cat_in)
        z_shared = z_shared.to(device)

        # zs

        encoders_outputs = []
        encoders_inputs = [(x_, *cat_in) for _ in cat_in]

        for i in range(len(self.z_encoders_list) - 1):
            encoders_outputs.append(self.z_encoders_list[i + 1](*encoders_inputs[i]))

        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]

        # zs_cf

        encoders_outputs_cf = []
        encoders_inputs_cf = [(x_, *torch.split(cat_covs_cf[i], 1, dim=1))
                              for i in range(len(cat_covs_cf))]

        for i in range(len(self.z_encoders_list) - 1):
            encoders_outputs_cf.append(self.z_encoders_list[i + 1](*encoders_inputs_cf[i]))

        qzs_cf = [enc_out[0] for enc_out in encoders_outputs_cf]
        zs_cf = [enc_out[1].to(device) for enc_out in encoders_outputs_cf]

        # zs_prior

        encoders_prior_outputs = []
        encoders_prior_inputs = [(torch.tensor([]).to(device), c) for c in cat_in]
        for i in range(len(self.z_prior_encoders_list)):
            encoders_prior_outputs.append(self.z_prior_encoders_list[i](*encoders_prior_inputs[i]))

        qzs_prior = [enc_out[0] for enc_out in encoders_prior_outputs]
        zs_prior = [enc_out[1].to(device) for enc_out in encoders_prior_outputs]

        # nullify if required

        if nullify_shared:
            z_shared = torch.zeros_like(z_shared).to(device)

        for i in range(self.zs_num):
            if i in nullify_cat_covs_indices:
                zs[i] = torch.zeros_like(zs[i]).to(device)
                zs_cf[i] = torch.zeros_like(zs_cf[i]).to(device)

        zs_concat = torch.cat(zs, dim=-1)
        z_concat = torch.cat([z_shared, zs_concat], dim=-1)

        output_dict = {
            "z_shared": z_shared,
            "zs": zs,
            "zs_cf": zs_cf,
            "zs_prior": zs_prior,
            "qz_shared": qz_shared,
            "qzs": qzs,
            "qzs_cf": qzs_cf,
            "qzs_prior": qzs_prior,
            "z_concat": z_concat,
            "library": library,
            "library_cf": library_cf,
            "cat_covs": cat_covs,
        }
        return output_dict

    @auto_move_data
    def generative(self, z_shared,
                   zs, zs_cf,
                   library, library_cf,
                   cat_covs,
                   ):

        output_dict = {"px": []}

        z = [z_shared] + zs
        z_cf = [z_shared] + zs_cf

        lib = [library] + library_cf

        cats_splits = torch.split(cat_covs, 1, dim=1)
        all_cats_but_one = []
        for i in range(self.zs_num):
            all_cats_but_one.append([cats_splits[j] for j in range(len(cats_splits)) if j != i])

        dec_cats_in = [cats_splits] + all_cats_but_one

        for dec_count in range(self.zs_num + 1):

            x_decoder = self.x_decoders_list[dec_count]

            for i in range(2):

                dec_covs = dec_cats_in[dec_count]

                x_decoder_input = [z[dec_count], z_cf[dec_count]][i]

                size_factor = [lib[0], lib[dec_count]][i]

                px_scale, px_r, px_rate, px_dropout = x_decoder(
                    self.dispersion,
                    x_decoder_input,
                    size_factor,
                    *dec_covs
                )
                px_r = torch.exp(self.px_r)

                if self.gene_likelihood == "zinb":
                    px = ZeroInflatedNegativeBinomial(
                        mu=px_rate,
                        theta=px_r,
                        zi_logits=px_dropout,
                        scale=px_scale,
                    )
                elif self.gene_likelihood == "nb":
                    px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
                elif self.gene_likelihood == "poisson":
                    px = Poisson(px_rate, scale=px_scale)

                if i == 0:
                    output_dict["px"] += [px]
                else:
                    output_dict[f"px_{dec_count}"] = px

        return output_dict

    def sub_forward(self, idx,
                    x, cat_covs,
                    detach_x=False,
                    detach_z=False):
        """

        performs forward (inference + generative) only on enc/dec idx

        Parameters
        ----------
        idx
            index of enc/dec in [1, ..., self.zs_num]
        x
        cat_covs
        detach_x
        detach_z

        """
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)

        qz, z = (self.z_encoders_list[idx](x_, *cat_in))
        if detach_z:
            z = z.detach()

        dec_cats = [cat_in[j] for j in range(len(cat_in)) if j != idx-1]

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            z,
            library,
            *dec_cats
        )
        px_r = torch.exp(self.px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        return px

    def classification_logits(self, inference_outputs):
        zs = inference_outputs["zs"]
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(zs[i])
            logits += [logits_i]

        return logits

    def compute_clf_metrics(self, logits, cat_covs):
        # CE, ACC, F1
        cats = torch.split(cat_covs, 1, dim=1)
        ce_losses = []
        accuracy_scores = []
        f1_scores = []
        for i in range(self.zs_num):
            s_i = one_hot_cat([self.n_cat_list[i]], cats[i]).to(device)
            ce_losses += [F.cross_entropy(logits[i], s_i)]
            kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
            predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
            acc = Accuracy(**kwargs).to(device)
            accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
            F1 = F1Score(**kwargs).to(device)
            f1_scores.append(F1(predicted_labels, cats[i]).to(device))

        ce_loss_sum = sum(torch.mean(ce) for ce in ce_losses)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)
        f1 = sum(f1_scores) / len(f1_scores)

        return ce_loss_sum, accuracy, f1

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            cf_weight: Tunable[Union[float, int]],  # RECONST_LOSS_X_CF weight
            alpha: Tunable[Union[float, int]],  # KL Zi weight
            clf_weight: Tunable[Union[float, int]],  # Si classifier weight
            mode: Tunable[Tuple[int]],
            cf_mode: Tunable[int],
            kl_weight: float = 1.0,
    ):
        # reconstruction loss X

        x = tensors[REGISTRY_KEYS.X_KEY]

        reconst_loss_x_list = [-torch.mean(px.log_prob(x[0]).sum(-1)) for px in generative_outputs["px"]]
        reconst_loss_x_dict = {'x_' + str(i): reconst_loss_x_list[i] for i in range(len(reconst_loss_x_list))}
        reconst_loss_x = sum(reconst_loss_x_list)

        #  reconstruction loss X'

        reconst_loss_x_cf_list = [-torch.mean(generative_outputs[f"px_{cf}"].log_prob(x[cf]).sum(-1)) for cf in
                                  range(1, len(x))]
        reconst_loss_x_cf_dict = {'xcf_' + str(i + 1): reconst_loss_x_cf_list[i] for i in
                                  range(len(reconst_loss_x_cf_list))}
        reconst_loss_x_cf = sum(reconst_loss_x_cf_list)

        # KL divergence Z

        kl_z_list = [torch.mean(kl(qzs, qzs_prior).sum(dim=1)) for qzs, qzs_prior in
                     zip(inference_outputs["qzs"], inference_outputs["qzs_prior"])]

        kl_z_dict = {'z_' + str(i+1): kl_z_list[i] for i in range(len(kl_z_list))}

        # classification metrics: CE, ACC, F1

        cat_covs = inference_outputs["cat_covs"]
        logits = self.classification_logits(inference_outputs)
        ce_loss_sum, accuracy, f1 = self.compute_clf_metrics(logits, cat_covs)
        ce_loss_mean = ce_loss_sum / len(range(self.zs_num))

        # total loss
        loss = reconst_loss_x
        if TRAIN_MODE.RECONST_CF in mode:
            loss += reconst_loss_x_cf * cf_weight
        if TRAIN_MODE.KL_Z in mode:
            loss += sum(kl_z_list) * kl_weight * alpha
        if TRAIN_MODE.CLASSIFICATION in mode:
            loss += ce_loss_sum * clf_weight

        loss_dict = {
            LOSS_KEYS.LOSS: loss,
            LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x_dict,
            LOSS_KEYS.RECONST_LOSS_X_CF: reconst_loss_x_cf_dict,
            LOSS_KEYS.KL_Z: kl_z_dict,
            LOSS_KEYS.CLASSIFICATION_LOSS: ce_loss_mean,
            LOSS_KEYS.ACCURACY: accuracy,
            LOSS_KEYS.F1: f1
        }

        return loss_dict
