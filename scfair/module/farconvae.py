from typing import Callable, Iterable, Literal, Optional

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
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data, BaseModuleClass
from scvi.nn import DecoderSCVI, Encoder, one_hot, _base_components

from scvi.module._vae import VAE

from scvi_dev.nn._base_components_utils import *
from scvi_dev.nn._utils import *

torch.backends.cudnn.benchmark = True


class FarconVAE(VAE, BaseMinifiedModeModuleClass):
    """
    FAir Representation via distributional CONtrastive Variational AutoEncoder (FarconVAE) model.
    sensitives attributes are only categorical
    """

    def __init__(
            self,
            n_input: int,
            n_batch: int = 0,
            n_labels: int = 0,
            n_hidden: Tunable[int] = 128,
            n_latent: Tunable[int] = 10,
            n_latent_sensitive: Tunable[int] = 10,                       # new ( must be = n_latent)
            n_layers: Tunable[int] = 1,
            n_continuous_cov: int = 0,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            dispersion: Tunable[Literal["gene", "gene-batch", "gene-label", "gene-cell"]] = "gene",
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            encode_covariates: Tunable[bool] = True,                            ###new###
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            use_size_factor_key: bool = False,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            var_activation: Optional[Callable] = None,
            sensitive_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",   # new
            alpha=1,                                                                    # new
            beta=0.2,                                                                   # new
            gamma=1,                                                                    # new
            kernel: Tunable[Literal["student-t", "gaussian"]] = "student-t",            # new
            **kw
    ):
        super().__init__(n_input)
        self.n_input = n_input
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_latent_sensitive = n_latent_sensitive
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.sensitive_likelihood = sensitive_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kernel = kernel

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        if n_batch > 1:
            self.cat_list = [n_batch] + self.cat_list

        self.x_decoder_out = n_input
        self.s_decoder_out = int(np.sum(self.cat_list))
        self.y_decoder_out = n_labels

        self.p_r = []

        for i in range(10):
            dec_out_size = [self.x_decoder_out, self.s_decoder_out][i % 2]
            if i >= 8:
                dec_out_size = self.y_decoder_out
            if self.dispersion == "gene":
                pr = torch.nn.Parameter(torch.randn(dec_out_size))
            elif self.dispersion == "gene-batch":
                pr = torch.nn.Parameter(torch.randn(dec_out_size, n_batch))
            elif self.dispersion == "gene-label":
                pr = torch.nn.Parameter(torch.randn(dec_out_size, n_labels))
            elif self.dispersion == "gene-cell":
                pass
            else:
                raise ValueError(
                    "dispersion must be one of ['gene', 'gene-batch',"
                    " 'gene-label', 'gene-cell'], but input was "
                    "{}.format(self.dispersion)"
                )
            self.p_r += [pr]

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates

        encoder_cat_list = self.cat_list if encode_covariates else None

        self.zs_encoder = Encoder(
            n_input_encoder,
            n_latent_sensitive,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        self.zx_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data

        self.decoders = []

        for i in range(10):
            n_input_decoder = n_latent + n_latent_sensitive + n_continuous_cov
            n_output_decoder = [self.x_decoder_out, self.s_decoder_out][i % 2]
            if i >= 8:
                n_input_decoder = n_latent + n_continuous_cov
                n_output_decoder = self.y_decoder_out
            decoder = DecoderSCVI(
                n_input_decoder,
                n_output_decoder,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=False,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
            )
            self.decoders += [decoder]


    def _get_inference_input(
            self,
            tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.minified_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            input_dict = dict(
                x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
            )
        else:
            if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                observed_lib_size = tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE]
                input_dict = dict(qzm=qzm, qzv=qzv, observed_lib_size=observed_lib_size)
            else:
                raise NotImplementedError(
                    f"Unknown minified-data type: {self.minified_data_type}"
                )

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        zs = inference_outputs["zs"]
        zs_c = inference_outputs["zs_c"]
        zx = inference_outputs["zx"]
        zx_c = inference_outputs["zx_c"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        input_dict = dict(
            zs=zs,
            zs_c=zs_c,
            zx=zx,
            zx_c=zx_c,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict

    @auto_move_data
    def _regular_inference(
            self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1
    ):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
            cat_c = get_counterfactual_cat(self.cat_list, cat_covs, batch_index)
            cat_c_input = torch.split(cat_c, 1, dim=1)
        else:
            categorical_input = tuple()
            cat_c_input = tuple()

        qzs, zs = self.zs_encoder(encoder_input, batch_index, *categorical_input)
        qzs_c, zs_c = self.zs_encoder(encoder_input, *cat_c_input)
        qzx, zx = self.zx_encoder(encoder_input, batch_index, *categorical_input)
        qzx_c, zx_c = self.zx_encoder(encoder_input, *cat_c_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            untran_zs = qzs.sample((n_samples,))
            zs = self.zs_encoder.z_transformation(untran_zs)
            untran_zs_c = qzs_c.sample((n_samples,))
            zs_c = self.zs_encoder.z_transformation(untran_zs_c)
            untran_zx = qzx.sample((n_samples,))
            zx = self.zx_encoder.z_transformation(untran_zx)
            untran_zx_c = qzx_c.sample((n_samples,))
            zx_c = self.zx_encoder.z_transformation(untran_zx_c)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))

        outputs = dict(zs=zs, qzs=qzs, zs_c=zs_c, qzs_c=qzs_c,
                       zx=zx, qzx=qzx, zx_c=zx_c, qzx_c=qzx_c,
                       z=zx, qz=qzx,
                       ql=ql,
                       library=library)
        return outputs

    # TODO: implement _cached_inference

    @auto_move_data
    def generative(
            self,
            zs,
            zs_c,
            zx,
            zx_c,
            library,
            batch_index,
            cont_covs=None,
            cat_covs=None,
            size_factor=None,
            y=None,
            transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        # Likelihood distribution

        decoder_inputs = []
        y_decoder_inputs = []

        for i in range(4):
            z_x, z_s = [(zx, zs), (zx_c, zs), (zx_c, zs_c), (zx, zs_c)][i]
            if cont_covs is None:
                decoder_input = torch.cat([z_s, z_x], dim=-1)
            elif z_x.dim() != cont_covs.dim():
                decoder_input = torch.cat(
                    [z_s, z_x, cont_covs.unsqueeze(0).expand(z_x.size(0), -1, -1)], dim=-1
                )
            else:
                decoder_input = torch.cat([z_s, z_x, cont_covs], dim=-1)
            decoder_inputs += [decoder_input]

        for i in range(2):
            z_x = [zx, zx_c][i]
            if cont_covs is None:
                decoder_input_zx = z_x
            elif z_x.dim() != cont_covs.dim():
                decoder_input_zx = torch.cat(
                    [z_x, cont_covs.unsqueeze(0).expand(z_x.size(0), -1, -1)], dim=-1
                )
            else:
                decoder_input_zx = torch.cat([z_x, cont_covs], dim=-1)
            y_decoder_inputs += [decoder_input_zx]

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        p = []

        for i in range(10):
            decoder = self.decoders[i]
            dec_input = decoder_inputs[i // 2] if i < 8 else y_decoder_inputs[i - 8]
            px_scale, px_r, px_rate, px_dropout = decoder(
                self.dispersion,
                dec_input,
                size_factor,
            )
            if self.dispersion == "gene-label":
                px_r = F.linear(
                    one_hot(y, self.n_labels), self.p_r[i]
                )  # px_r gets transposed - last dimension is nb genes
            elif self.dispersion == "gene-batch":
                px_r = F.linear(one_hot(batch_index, self.n_batch), self.p_r[i])
            elif self.dispersion == "gene":
                px_r = self.p_r[i]

            px_r = torch.exp(px_r)

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

            p += [px]

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pzs = Normal(torch.zeros_like(zs), torch.ones_like(zs))
        pzx = Normal(torch.zeros_like(zx), torch.ones_like(zx))
        pzs_c = Normal(torch.zeros_like(zs_c), torch.ones_like(zs_c))
        pzx_c = Normal(torch.zeros_like(zx_c), torch.ones_like(zx_c))
        return dict(
            px_given_x_s=p[0],
            ps_given_x_s=p[1],
            px_given_xc_s=p[2],
            ps_given_xc_s=p[3],
            px_given_xc_sc=p[4],
            ps_given_xc_sc=p[5],
            px_given_x_sc=p[6],
            ps_given_x_sc=p[7],
            py_given_x=p[8],
            py_given_xc=p[9],
            px=p[0],
            pz=pzx,
            pl=pl,
            pzs=pzs,
            pzx=pzx,
            pzs_c=pzs_c,
            pzx_c=pzx_c
        )

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        # compute ELBO
        recon_loss = -generative_outputs["px_given_x_s"].log_prob(x).sum(-1)
        s_cat = one_hot_cat(self.cat_list, cat_covs, batch_index)
        recon_loss += -generative_outputs["ps_given_x_s"].log_prob(s_cat).sum(-1)
        pred_loss = -generative_outputs["py_given_x"].log_prob(y).sum(-1)

        recon_loss_c = -generative_outputs["px_given_xc_sc"].log_prob(x).sum(-1)
        s_cat_c = get_counterfactual_cat(self.cat_list, cat_covs, batch_index, onehot=True)
        recon_loss_c += -generative_outputs["ps_given_xc_sc"].log_prob(s_cat_c).sum(-1)
        pred_loss_c = -generative_outputs["py_given_xc"].log_prob(y).sum(-1)

        # zs=zs, qzs=qzs, zs_c=zs_c, qzs_c=qzs_c, zx=zx, qzx=qzx, zx_c=zx_c, qzx_c=qzx_c, ql=ql, library=library

        kl_divergence_z = kl(inference_outputs["qzx"], generative_outputs["pzx"]).sum(dim=1) + \
                          kl(inference_outputs["qzs"], generative_outputs["pzs"]).sum(dim=1)
        kl_divergence_zc = kl(inference_outputs["qzx_c"], generative_outputs["pzx_c"]).sum(dim=1) + \
                           kl(inference_outputs["qzs_c"], generative_outputs["pzs_c"]).sum(dim=1)

        elbo = recon_loss + pred_loss + self.beta * kl_divergence_z
        elbo_c = recon_loss_c + pred_loss_c + self.beta * kl_divergence_zc
        avg_elbo = (elbo + elbo_c) / 2

        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        # kl_local_for_warmup = kl_divergence_z
        # kl_local_no_warmup = kl_divergence_l

        # weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        # loss = torch.mean(recon_loss + weighted_kl_local)

        kl_avg = lambda p, q: (kl(p, q) + kl(q, p)) / 2

        if self.kernel == "student-t":
            kernel_func = lambda kl: 1 / (1 + kl)
        elif self.kernel == "gaussian":
            kernel_func = lambda kl: torch.exp(-kl)

        qzx = inference_outputs["qzx"]
        qzs = inference_outputs["qzs"]
        qzx_c = inference_outputs["qzx_c"]
        qzs_c = inference_outputs["qzs_c"]

        L_DC = kl_avg(qzx, qzx_c).sum(dim=1) + \
               kernel_func(kl_avg(qzs, qzs_c).sum(dim=1)) + \
               kernel_func(kl_avg(qzx, qzs).sum(dim=1)) + \
               kernel_func(kl_avg(qzx_c, qzs_c).sum(dim=1))

        L_SR = (generative_outputs["px_given_xc_s"].log_prob(x).sum(-1) +
                generative_outputs["ps_given_xc_s"].log_prob(s_cat).sum(-1)) / 2 + \
               generative_outputs["px_given_x_sc"].log_prob(x).sum(-1) + \
               generative_outputs["ps_given_x_sc"].log_prob(s_cat_c).sum(-1)

        loss = torch.mean(avg_elbo - self.alpha * L_DC - self.gamma * L_SR)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        return LossOutput(
            loss=loss, reconstruction_loss=recon_loss, kl_local=kl_local
        )
