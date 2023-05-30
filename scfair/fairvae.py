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
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Decoder, Encoder, _base_components, one_hot, FCLayers

from torchmetrics import Accuracy, F1Score

torch.backends.cudnn.benchmark = True

from ._utils import *

from scvi.module._classifier import Classifier

dim_indices = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'

# all tensors are samples_count x cov_count

# for hyperparameter tuning
from ray import tune
from scvi._decorators import classproperty
from scvi.autotune._types import Tunable, TunableMixin


for_train = True

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
            n_continuous_cov: int = 0,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
            alpha: Tunable[Union[List[float], Tuple[float], float]] = 1.0,  # coef for P(Si|Zi)
            beta: Tunable[float] = 1.0,  # coef for TC term
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.alpha = alpha
        self.beta = beta

        self.px_r = torch.nn.Parameter(torch.randn(n_input)).to(device)

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders

        n_input_encoder = n_input + n_continuous_cov
        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        self.zs_num = len(self.n_cat_list) + n_continuous_cov
        # TODO: should be changed (e.g. some cont_covs (pc_i) might be grouped to 1 zs)

        if isinstance(self.alpha, float):
            self.alpha = [self.alpha for _ in range(self.zs_num)]

        self.z_encoders_list = nn.ModuleList(
            [
                Encoder(
                    n_input_encoder,
                    n_latent_shared,
                    # n_cat_list=self.n_cat_list,
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
                    # n_cat_list=[self.n_cat_list[i] for i in range(len(self.n_cat_list)) if i != k],
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

        self.z_to_zcf_nn = nn.ModuleList(
            [
                FCLayers(
                    n_in=n_latent_shared,
                    n_out=n_latent_shared,
                    n_cat_list=[self.n_cat_list[i], self.n_cat_list[i]],    # cov, cov_cf
                    n_layers=0,
                    n_hidden=n_latent_shared,
                ).to(device)
                for i in range(self.zs_num)
            ]
        )

        self.s_prior = [nn.Parameter((1 / n_labels) * torch.ones(1, n_labels), requires_grad=False).to(device)
                        for n_labels in self.n_cat_list]

        self.ps_r = [nn.Parameter(torch.randn(self.n_cat_list[i])).to(device) for i in range(self.zs_num)]

    def set_require_grad(self, mode: int):
        if mode < TRAIN_MODE.CLASSIFICATION:   # train all Z_i (encoders and decoders)
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = False

        else:   # add classifiers P(Si | Zi) for all i
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = True

    def _get_inference_input(self, tensors, for_train=for_train):
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs_tot = tensors[cont_key].to(device) if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs_tot = tensors[cat_key].to(device) if cat_key in tensors.keys() else None

        x_tot = tensors[REGISTRY_KEYS.X_KEY].to(device)

        batch_size = cat_covs_tot.size(dim=dim_indices)
        indices = torch.tensor(list(range(batch_size))).to(device)
        indices_cf = torch.tensor(list(range(batch_size))).to(device)
        if for_train:
            # indices, indices_cf = get_paired_indices(cont_covs_tot, cat_covs_tot, dim_indices)
            indices = torch.tensor(list(range(batch_size // 2))).to(device)
            indices_cf = torch.tensor(list(range(batch_size // 2, 2 * (batch_size // 2)))).to(device)

        x = torch.index_select(x_tot, dim=dim_indices, index=indices)
        x_cf = torch.index_select(x_tot, dim=dim_indices, index=indices_cf)

        cont_covs = torch.index_select(cont_covs_tot, dim=dim_indices,
                                       index=indices).to(device) if cont_covs_tot is not None else None
        cont_covs_cf = torch.index_select(cont_covs_tot, dim=dim_indices,
                                          index=indices_cf).to(device) if cont_covs_tot is not None else None

        cat_covs = torch.index_select(cat_covs_tot, dim=dim_indices,
                                      index=indices).to(device) if cat_covs_tot is not None else None
        cat_covs_cf = torch.index_select(cat_covs_tot, dim=dim_indices,
                                         index=indices_cf).to(device) if cat_covs_tot is not None else None

        input_dict = {
            "x": x.to(device),
            "x_cf": x_cf.to(device),
            "cont_covs": cont_covs,
            "cont_covs_cf": cont_covs_cf,
            "cat_covs": cat_covs,
            "cat_covs_cf": cat_covs_cf,
            "indices": indices.to(device),
            "indices_cf": indices_cf.to(device)
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_shared = inference_outputs["z_shared"]
        z_shared_cf = inference_outputs["z_shared_cf"]
        zs = inference_outputs["zs"]  # a list of all zs
        zs_cf = inference_outputs["zs_cf"]  # a list of all zs_cf
        library = inference_outputs["library"]
        library_cf = inference_outputs["library_cf"]
        library_s = inference_outputs["library_s"]

        input_dict = {
            "z_shared": z_shared,
            "z_shared_cf": z_shared_cf,
            "zs": zs,  # a list of all zs
            "zs_cf": zs_cf,  # a list of all zs_cf
            "library": library,
            "library_cf": library_cf,
            "library_s": library_s,
            "cont_covs": inference_outputs["cont_covs"],
            "cont_covs_cf": inference_outputs["cont_covs_cf"],
            "cat_covs": inference_outputs["cat_covs"],
            "cat_covs_cf": inference_outputs["cat_covs_cf"],
            "indices": inference_outputs["indices"],
            "indices_cf": inference_outputs["indices_cf"]
        }
        return input_dict

    @auto_move_data
    def inference(self, x, x_cf,
                  cont_covs, cont_covs_cf,
                  cat_covs, cat_covs_cf,
                  indices, indices_cf,
                  nullify_cat_covs_indices: Optional[List[int]] = None,
                  nullify_cont_covs_indices: Optional[List[int]] = None,
                  nullify_shared: Optional[bool] = False,
                  for_train=for_train):

        nullify_cat_covs_indices = [] if nullify_cat_covs_indices is None else nullify_cat_covs_indices
        nullify_cont_covs_indices = [] if nullify_cont_covs_indices is None else nullify_cont_covs_indices

        x_ = x
        x_cf_ = x_cf
        library = torch.log(x.sum(1)).unsqueeze(1)
        library_cf = torch.log(x_cf.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            x_cf_ = torch.log(1 + x_cf_)

        library_s = []

        if cont_covs is not None:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_cf = torch.cat((x_cf_, cont_covs_cf), dim=-1)
            cont_input = torch.split(cont_covs, 1, dim=1)
            library_s.extend(torch.log(c.sum(1)).unsqueeze(1) for c in cont_input)
        else:
            encoder_input = x_
            encoder_input_cf = x_cf_

        qz_shared, z_shared = self.z_encoders_list[0](encoder_input)
        qz_shared_cf, z_shared_cf = self.z_encoders_list[0](encoder_input_cf)
        z_shared = z_shared.to(device)
        z_shared_cf = z_shared_cf.to(device)

        encoders_outputs = [self.z_encoders_list[i + 1](encoder_input)
                            for i in range(len(self.z_encoders_list) - 1)]
        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]

        encoders_outputs_cf = [self.z_encoders_list[i + 1](encoder_input_cf)
                               for i in range(len(self.z_encoders_list) - 1)]
        qzs_cf = [enc_out[0] for enc_out in encoders_outputs_cf]
        zs_cf = [enc_out[1].to(device) for enc_out in encoders_outputs_cf]

        # nullify if required

        if nullify_shared:
            z_shared = torch.zeros_like(z_shared).to(device)
            z_shared_cf = torch.zeros_like(z_shared_cf).to(device)

        for i in range(self.zs_num):
            if ((i - len(self.n_cat_list)) in nullify_cont_covs_indices) or (i in nullify_cat_covs_indices):
                zs[i] = torch.zeros_like(zs[i]).to(device)
                zs_cf[i] = torch.zeros_like(zs_cf[i]).to(device)

        zs_concat_f = torch.cat(zs, dim=-1)
        z_concat_f = torch.cat([z_shared, zs_concat_f], dim=-1)
        zs_concat_cf = torch.cat(zs_cf, dim=-1)
        z_concat_cf = torch.cat([z_shared_cf, zs_concat_cf], dim=-1)

        z_concat = z_concat_f
        if for_train:
            z_concat = torch.cat([z_concat_f, z_concat_cf], dim=0)

        output_dict = {
            "z_shared": z_shared,
            "z_shared_cf": z_shared_cf,
            "zs": zs,
            "zs_cf": zs_cf,
            "qz_shared": qz_shared,
            "qz_shared_cf": qz_shared_cf,
            "qzs": qzs,
            "qzs_cf": qzs_cf,
            "z_concat": z_concat,
            "library": library,
            "library_cf": library_cf,
            "library_s": library_s,
            "cont_covs": cont_covs,
            "cont_covs_cf": cont_covs_cf,
            "cat_covs": cat_covs,
            "cat_covs_cf": cat_covs_cf,
            "indices": indices,
            "indices_cf": indices_cf
        }
        return output_dict

    @auto_move_data
    def generative(self, z_shared, z_shared_cf,
                   zs, zs_cf,
                   library, library_cf, library_s,
                   cont_covs, cont_covs_cf,
                   cat_covs, cat_covs_cf,
                   indices, indices_cf
                   ):
        output_dict = {}
        output_dict["px"] = []
        output_dict["px_cf"] = []

        if cat_covs is not None:
            cat_in = torch.split(cat_covs, 1, dim=1)
            cat_in_cf = torch.split(cat_covs_cf, 1, dim=1)
            library_s.extend(torch.log(c.sum(1)).unsqueeze(1) for c in cat_in)
        else:
            cat_in = ()
            cat_in_cf = ()

        all_cat_in = []
        all_cat_in_cf = []
        for i in range(self.zs_num):
            all_cat_in.append([cat_in[j] for j in range(len(cat_in)) if j != i])
            all_cat_in_cf.append([cat_in_cf[j] for j in range(len(cat_in_cf)) if j != i])

        for k in [0, 1]:
            # p(x|z), p(x|z')

            cats = [cat_in, cat_in_cf][k]
            all_cats = [all_cat_in, all_cat_in_cf][k]

            z_shared_k = z_shared

            for i in range(self.zs_num + 1):

                if i == 0:
                    x_decoder_input = z_shared_k
                else:
                    x_decoder_input = self.z_to_zcf_nn[i-1](zs[i-1], cat_in[i-1], cat_in_cf[i-1])

                size_factor = [library, library_cf][k]

                x_decoder = self.x_decoders_list[i]
                dec_covs = cats if i == 0 else all_cats[i - 1]

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

                if k == 0:
                    output_dict["px"] += [px]
                else:
                    output_dict["px_cf"] += [px]

        output_dict["ps|z"] = []
        output_dict["ps"] = []

        for i in range(self.zs_num):
            # p(s|z)
            zs_i = zs[i]
            s_i_classifier = self.s_classifiers_list[i]

            ps_i_given_z_i = s_i_classifier(zs_i)
            ps_i = self.s_prior[i].repeat(ps_i_given_z_i.size(0), 1)

            output_dict["ps|z"].append(ps_i_given_z_i)
            output_dict["ps"].append(ps_i)

        return output_dict

    def classification_loss(self, labelled_dataset):
        x = labelled_dataset[REGISTRY_KEYS.X_KEY].to(device)

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key].to(device) if cat_key in labelled_dataset.keys() else None

        if cat_covs is not None:
            cat_in = torch.split(cat_covs, 1, dim=1)
        else:
            cat_in = ()

        ce_loss = []
        logits = []
        for i in range(self.zs_num):
            zs_i = self.z_encoders_list[i+1](x)[1].to(device)

            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(zs_i)
            logits += [logits_i]

            s_i = one_hot_cat([self.n_cat_list[i]], cat_in[i]).to(device)

            ce_loss += [
                F.cross_entropy(
                    logits_i,
                    s_i
                )
            ]

        return ce_loss, cat_in, logits

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
            labelled_tensors=None,
            classification_ratio=None,
            mode=0
    ):
        indices = inference_outputs["indices"]
        indices_cf = inference_outputs["indices_cf"]
        x_tot = tensors[REGISTRY_KEYS.X_KEY].to(device)
        x = torch.index_select(x_tot, dim=dim_indices, index=indices).to(device)
        x_cf = torch.index_select(x_tot, dim=dim_indices, index=indices_cf).to(device)

        # reconstruction losses

        reconst_x_losses = [-torch.mean(px.log_prob(x).sum(-1)) for px in generative_outputs["px"]]
        reconst_x_cf_losses = [-torch.mean(px_cf.log_prob(x_cf).sum(-1)) for px_cf in generative_outputs["px_cf"]]

        reconst_loss_x = torch.mean(reconst_x_losses)
        reconst_loss_x_cf = torch.mean(reconst_x_cf_losses)

        reconst_loss = reconst_loss_x + reconst_loss_x_cf

        cat_covs = inference_outputs["cat_covs"]
        cat_input = torch.split(cat_covs, 1, dim=1)

        # KL divergence S

        kl_s = [kl(Categorical(generative_outputs["ps"][i]),
                   Categorical(generative_outputs["ps|z"][i])).sum() * self.alpha[i]
                for i in range(self.zs_num)]

        kl_s_sum = sum(torch.mean(kl_s[i]) for i in range(self.zs_num))

        # Cross Entropy

        ce_losses, true_labels, logits = self.classification_loss(tensors)
        ce_loss = sum(torch.mean(ce_losses[i]) for i in range(self.zs_num))

        # compute other metrics (accuracy, F1) and log

        accuracy_scores = []
        f1_scores = []
        for i in range(self.zs_num):
            kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
            predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
            acc = Accuracy(**kwargs).to(device)
            accuracy_scores.append(acc(predicted_labels, true_labels[i]).to(device))
            F1 = F1Score(**kwargs).to(device)
            f1_scores.append(F1(predicted_labels, true_labels[i]).to(device))

        accuracy = sum(accuracy_scores)
        f1 = sum(f1_scores)
        
        # KL divergence Z

        px_cf_mean_list = [px_cf.mean for px_cf in generative_outputs["px_cf"]]

        px_cf_mean = reduce(
            torch.Tensor.add_,
            px_cf_mean_list,
            torch.zeros_like(px_cf_mean_list[0])
        ) / len(px_cf_mean_list)

        cont_covs_cf = inference_outputs["cont_covs_cf"]
        cat_covs_cf = inference_outputs["cat_covs_cf"]

        new_inference_out = self.inference(px_cf_mean, px_cf_mean,
                                           cont_covs_cf, cont_covs_cf,
                                           cat_covs_cf, cat_covs_cf,
                                           indices_cf, indices_cf)

        kl_z_shared = kl(inference_outputs["qz_shared"], new_inference_out["qz_shared_cf"]).sum(dim=1)
        kl_zs = [kl(qzs, qzs_cf).sum(dim=1) for qzs, qzs_cf in
                            zip(inference_outputs["qzs"], new_inference_out["qzs_cf"])]

        kl_zs_sum = sum([torch.mean(kl_z_i) for kl_z_i in kl_zs])
        kl_z = torch.mean(kl_z_shared) + kl_zs_sum

        # total loss
        loss = reconst_loss
        if mode >= TRAIN_MODE.KL_Z:
            loss += kl_z
        # if mode >= :
        #     loss += torch.mean(sum(kl_s)) * kl_weight
        if mode >= TRAIN_MODE.CLASSIFICATION:
            loss += ce_loss * classification_ratio

        loss_dict = {
            LOSS_KEYS.LOSS: loss,
            LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x,
            LOSS_KEYS.RECONST_LOSS_X_CF: reconst_loss_x_cf,
            LOSS_KEYS.KL_S: kl_s_sum,
            LOSS_KEYS.KL_Z_SHARED: torch.mean(kl_z_shared),
            LOSS_KEYS.KL_ZS: kl_zs_sum,
            LOSS_KEYS.CLASSIFICATION_LOSS: ce_loss,
            LOSS_KEYS.ACCURACY: accuracy,
            LOSS_KEYS.F1: f1
        }

        return loss_dict
