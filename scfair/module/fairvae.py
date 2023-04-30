from typing import Callable, Iterable, Literal, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Bernoulli
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Decoder, Encoder, _base_components, one_hot

torch.backends.cudnn.benchmark = True

# from scvi_dev.nn._base_components_utils import *
from scvi_dev.nn._utils import *

dim_indices = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'

# all tensors are samples_count x cov_count


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
            alpha: int = 1,  # coef for P(Si|Zi)
            beta: int = 1,  # coef for TC term
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

        self.z_shared_encoder = Encoder(
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

        self.zs_num = len(self.n_cat_list) + n_continuous_cov
        # TODO: should be changed (e.g. some cont_covs (pc_i) might be grouped to 1 zs)

        self.zs_encoders_list = torch.nn.ModuleList(
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
                for _ in range(self.zs_num)
            ]
        )

        # Decoders
        # covs are not used in decoders input

        self.n_latent = n_latent_shared + n_latent_attribute * self.zs_num
        self.x_decoder = DecoderSCVI(
            self.n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
        ).to(device)
        self.n_cov_list = [1 if self.is_index_for_cont_cov(i) else self.n_cat_list[i] for i in range(self.zs_num)]
        # TODO: should be changed (e.g. some cont_covs (pc_i) might be grouped to 1 zs)

        self.s_decoders_list = torch.nn.ModuleList([])
        for i in range(self.zs_num):
            if self.is_index_for_cont_cov(i):
                self.s_decoders_list.append(
                    Decoder(
                        n_input=n_latent_attribute,
                        n_output=self.n_cov_list[i],
                        n_hidden=n_hidden,
                        n_layers=n_layers,
                        use_batch_norm=use_batch_norm_decoder,
                        use_layer_norm=use_layer_norm_decoder,
                        use_activation=True,
                    ).to(device)
                )
            else:
                self.s_decoders_list.append(
                    DecoderSCVI(
                        n_latent_attribute,
                        self.n_cov_list[i],
                        n_layers=n_layers,
                        n_hidden=n_hidden,
                        inject_covariates=deeply_inject_covariates,
                        use_batch_norm=use_batch_norm_decoder,
                        use_layer_norm=use_layer_norm_decoder,
                        scale_activation="softmax",
                    ).to(device)
                )

        self.ps_r = [torch.nn.Parameter(torch.randn(self.n_cov_list[i])).to(device) for i in range(self.zs_num)]

    def is_index_for_cont_cov(self, i):
        return i >= len(self.n_cat_list)

    def _get_inference_input(self, tensors, for_train=True):
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
            indices_cf = torch.tensor(list(range(batch_size // 2, batch_size))).to(device)

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
                  for_train=True):

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
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
            categorical_input_cf = torch.split(cat_covs_cf, 1, dim=1)
            library_s.extend(torch.log(c.sum(1)).unsqueeze(1) for c in categorical_input)
        else:
            categorical_input = ()
            categorical_input_cf = ()

        qz_shared, z_shared = self.z_shared_encoder(encoder_input, *categorical_input)
        qz_shared_cf, z_shared_cf = self.z_shared_encoder(encoder_input_cf, *categorical_input_cf)
        z_shared = z_shared.to(device)
        z_shared_cf = z_shared_cf.to(device)

        encoders_outputs = [zs_encoder(encoder_input, *categorical_input) for zs_encoder in self.zs_encoders_list]
        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]

        encoders_outputs_cf = [zs_encoder(encoder_input_cf, *categorical_input_cf) for zs_encoder in
                               self.zs_encoders_list]
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

    def construct_zs_cf(self, zs, cont_covs, cat_covs, cont_covs_cf, cat_covs_cf):
        # construct counterfactual zs with the same covs as (cont_covs_cf, cat_covs_cf)
        # approach: take mean of zs that have the same covs as (cont_covs_cf, cat_covs_cf)
        # if no such zs exists at a point we fill with 0
        # not: must have dim(covs_cf) << dim(covs)

        zs_cf = []

        for cov_idx in range(len(zs)):
            zs_cf_cov = []
            # for sample_idx in range(len(zs[0])):
            for sample_idx in range(len(cat_covs_cf)):

                if self.is_index_for_cont_cov(cov_idx):
                    cont_cov_idx = cov_idx - len(self.n_cat_list)
                    zs_cf_i_all = [zs[cov_idx][j] for j in range(len(zs[cov_idx]))
                                   if cont_covs[j][cont_cov_idx] == cont_covs_cf[sample_idx][cont_cov_idx]]
                else:
                    zs_cf_i_all = [zs[cov_idx][j] for j in range(len(zs[cov_idx]))
                                   if cat_covs[j][cov_idx] == cat_covs_cf[sample_idx][cov_idx]]

                zs_cf_i_tensor = torch.tensor([list(z) for z in zs_cf_i_all]) if len(zs_cf_i_all) > 0 \
                    else torch.zeros(len(zs[cov_idx][0]))

                # zs_cf_i_tensor = torch.cat(zs_cf_i_all, dim=0) if len(zs_cf_i_all) > 0 \
                #     else torch.zeros(1, len(zs[cov_idx][0]))
                    # else torch.zeros_like(zs[cov_idx][0])
                zs_cf_i_tensor = zs_cf_i_tensor.to(device)

                # print(zs_cf_i_tensor)

                zs_cf_i = torch.mean(zs_cf_i_tensor, dim=0) if len(zs_cf_i_all) > 0 \
                    else zs_cf_i_tensor
                zs_cf_i = zs_cf_i.to(device)

                # print(zs_cf_i)
                # print(zs_cf_i.size())

                zs_cf_cov.append(zs_cf_i)

            # print(zs_cf_cov)

            zs_cf.append(torch.tensor([list(z) for z in zs_cf_cov]).to(device))
            # zs_cf.append(torch.cat(zs_cf_cov, dim=0))

        return zs_cf

    @auto_move_data
    def generative(self, z_shared, z_shared_cf,
                   zs, zs_cf,
                   library, library_cf, library_s,
                   cont_covs, cont_covs_cf,
                   cat_covs, cat_covs_cf,
                   indices, indices_cf
                   ):
        output_dict = {}
        for i in [0, 1]:
            # p(x|z), p(x|z')

            z_shared_i = z_shared

            if i == 0:
                zs_i = zs
            else:
                zs_i = self.construct_zs_cf(zs, cont_covs, cat_covs, cont_covs_cf, cat_covs_cf)

            x_decoder_input = torch.cat([z_shared_i, *zs_i], dim=-1)
            size_factor = [library, library_cf][i]

            px_scale, px_r, px_rate, px_dropout = self.x_decoder(
                self.dispersion,
                x_decoder_input,
                size_factor,
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
                output_dict["px"] = px
            else:
                output_dict["px_cf"] = px

        output_dict["ps|z"] = []
        output_dict["ps"] = []

        for i in range(self.zs_num):
            zs_i = zs[i]
            s_i_decoder = self.s_decoders_list[i]
            if self.is_index_for_cont_cov(i):
                # p(s|z)
                ps_mean, ps_v = s_i_decoder(x=zs_i)
                ps = Normal(loc=ps_mean, scale=ps_v.sqrt())
            else:
                # p(s|z)
                size_factor = library_s[i].to(device)
                ps_scale, ps_r, ps_rate, ps_dropout = s_i_decoder(
                    self.dispersion,
                    zs_i,
                    size_factor,
                )
                ps_r = torch.exp(self.ps_r[i]).to(device)
                ps = NegativeBinomial(mu=ps_rate, theta=ps_r, scale=ps_scale)

            output_dict["ps"].append(ps)

        return output_dict

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
    ):
        indices = inference_outputs["indices"]
        indices_cf = inference_outputs["indices_cf"]
        x_tot = tensors[REGISTRY_KEYS.X_KEY].to(device)
        x = torch.index_select(x_tot, dim=dim_indices, index=indices).to(device)
        x_cf = torch.index_select(x_tot, dim=dim_indices, index=indices_cf).to(device)

        # reconstruction losses

        reconst_loss_x = -generative_outputs["px"].log_prob(x).sum(-1)
        reconst_loss_x_cf = -generative_outputs["px_cf"].log_prob(x_cf).sum(-1)

        cont_covs = inference_outputs["cont_covs"]
        cont_input = torch.split(cont_covs, 1, dim=1) if cont_covs is not None else None

        cat_covs = inference_outputs["cat_covs"]
        cat_input = torch.split(cat_covs, 1, dim=1)

        reconst_loss_s = 0
        for i in range(self.zs_num):
            s_i = cont_input[i - len(self.n_cat_list)] if self.is_index_for_cont_cov(i) \
                else one_hot_cat([self.n_cov_list[i]], cat_input[i]).to(device)

            # print(s_i.device)

            reconst_loss_s_i = -generative_outputs["ps"][i].log_prob(s_i).sum(-1)
            reconst_loss_s += reconst_loss_s_i

        reconst_loss_s *= self.alpha

        reconst_loss = torch.mean(reconst_loss_x) + torch.mean(reconst_loss_x_cf) + torch.mean(reconst_loss_s)

        # KL divergence

        # TODO: check why VCI used a different approach to compute p(Z | X', S') in KL divergence
        #  by calculating X' as generative_outputs["px_cf"].mean() instead of original X' from data

        kl_divergence_z_shared = kl(inference_outputs["qz_shared"], inference_outputs["qz_shared_cf"]).sum(dim=1)
        kl_divergence_zs = [kl(qzs, qzs_cf).sum(dim=1) for qzs, qzs_cf in
                            zip(inference_outputs["qzs"], inference_outputs["qzs_cf"])]

        weighted_kl_local = kl_weight * (kl_divergence_z_shared + sum(kl_divergence_zs))

        # total loss

        loss = reconst_loss + torch.mean(weighted_kl_local)

        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss_x, kl_local=weighted_kl_local
        )
