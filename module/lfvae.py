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


class LFVAE(VAE, BaseMinifiedModeModuleClass):
    """
    Lagrangian Fair Variational auto-encoder model.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[Literal["gene", "gene-batch", "gene-label", "gene-cell"]] = "gene",
        log_variational: bool = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = True,                                            ###new###
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        sensitive_likelihood_cat: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",    # new
        sensitive_likelihood_cont: Tunable[Literal["normal"]] = "normal",                # new
        sensitive_prior_cat: Tunable[Literal["bernoulli"]] = "bernoulli",                # new
        **kw
    ):
        super().__init__(n_input)
        self.n_input = n_input
        self.dispersion = dispersion
        self.encode_covariates = encode_covariates
        self.sensitive_likelihood_cat = sensitive_likelihood_cat
        self.sensitive_likelihood_cont = sensitive_likelihood_cont
        self.sensitive_prior_cat = sensitive_prior_cat
        
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # self.cat_list  = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)  
        self.cat_list  = list([] if n_cats_per_cov is None else n_cats_per_cov)
        if n_batch > 1:
            self.cat_list  = [n_batch] + self.cat_list

        self.encoder_cat_list = self.cat_list if encode_covariates else None
        
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.x_decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=self.encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )
        
        self.u_cat_decoder_out = int(np.sum(self.cat_list))
        
        if self.dispersion == "gene":
            self.puz_cat_r = torch.nn.Parameter(torch.randn(self.u_cat_decoder_out))
        elif self.dispersion == "gene-batch":
            self.puz_cat_r = torch.nn.Parameter(torch.randn(self.u_cat_decoder_out, n_batch))
        elif self.dispersion == "gene-label":
            self.puz_cat_r = torch.nn.Parameter(torch.randn(self.u_cat_decoder_out, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )
        
        self.u_cat_decoder = DecoderSCVI(
            n_input_decoder,
            self.u_cat_decoder_out,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=False,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )
        self.puz_cont_m_decoder = torch.nn.Linear(n_input_decoder, n_continuous_cov)
        self.puz_cont_v_decoder = torch.nn.Linear(n_input_decoder, n_continuous_cov)
        

    @auto_move_data
    def generative(
        self,
        z,
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
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        # p(x | z,u)
        px_scale, px_r, px_rate, px_dropout = self.x_decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

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

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        # p(u|z)  u: cat
        puz_cat_scale, puz_cat_r, puz_cat_rate, puz_cat_dropout = self.u_cat_decoder(
            self.dispersion,
            decoder_input,
            size_factor
        )
        if self.dispersion == "gene-label":
            puz_cat_r = F.linear(
                one_hot(y, self.n_labels), self.puz_cat_r
            )  # puz_cat_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            puz_cat_r = F.linear(one_hot(batch_index, self.n_batch), self.puz_cat_r)
        elif self.dispersion == "gene":
            puz_cat_r = self.puz_cat_r
            
        puz_cat_r = torch.exp(puz_cat_r)
                
        if self.sensitive_likelihood_cat == "zinb":
            puz_cat = ZeroInflatedNegativeBinomial(
                mu=puz_cat_rate,
                theta=puz_cat_r,
                zi_logits=puz_cat_dropout,
                scale=puz_cat_scale,
            )
        elif self.sensitive_likelihood_cat == "nb":
            puz_cat = NegativeBinomial(mu=puz_cat_rate, theta=puz_cat_r, scale=puz_cat_scale)
        elif self.sensitive_likelihood_cat == "poisson":
            puz_cat = Poisson(puz_cat_rate, scale=puz_cat_scale)

        # p(u)  u: cat
        if self.sensitive_prior_cat == "bernoulli":
            pu_cat = Bernoulli(torch.full([self.u_cat_decoder_out], 0.5))

        # p(u|z)  u: cont
        puz_cont_m = self.puz_cont_m_decoder(z)
        puz_cont_v = torch.exp(self.puz_cont_v_decoder(z))
        puz_cont = Normal(puz_cont_m, puz_cont_v.sqrt())

        # p(u)  u: cont
        pu_cont = Normal(torch.zeros_like(puz_cont_m), torch.ones_like(puz_cont_v.sqrt()))

        # TODO: calculate p(u) using prior distribution in dataset
        # TODO: implement other famous distributions for priors and posteriors

        return dict(
            px=px,
            pl=pl,
            pz=pz,
            puz_cat=puz_cat,
            pu_cat=pu_cat,
            puz_cont=puz_cont,
            pu_cont=pu_cont
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
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=1
        )
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        Lr = reconst_loss
        C1 = weighted_kl_local    # elbo

        u_cat = one_hot_cat(self.cat_list, cat_covs, batch_index)
        
        # compute C2_cat
        log_puz_cat = generative_outputs["puz_cat"].log_prob(u_cat)
        log_pu_cat = generative_outputs["pu_cat"].log_prob(u_cat)
        C2_cat = (log_puz_cat - log_pu_cat).sum(-1)

        # compute u_cont
        # TODO: compute C2_cont (we should incorporate u_cont in encoder and decoder)

        mode_sign = 1 if self.opt_mode == "min" else -1

        loss = torch.mean(Lr + self.l1 * C1 + self.l2 * C2_cat)

        kl_local = dict(
            kl_divergence_l=mode_sign * kl_divergence_l, kl_divergence_z=mode_sign * kl_divergence_z
        )
        return LossOutput(
            loss=mode_sign * loss, reconstruction_loss=mode_sign * reconst_loss, kl_local=kl_local
        )


    def set_requires_grad_encoder(self, req_grad):
        Encoder_set_requires_grad(self.z_encoder, req_grad)
        Encoder_set_requires_grad(self.l_encoder, req_grad)
        
    def set_requires_grad_x_decoder(self, req_grad):
        DecoderSCVI_set_requires_grad(self.x_decoder, req_grad)

    def set_requires_grad_u_decoder(self, req_grad):
        DecoderSCVI_set_requires_grad(self.u_cat_decoder, req_grad)
        set_requires_grad(self.puz_cont_m_decoder, req_grad)
        set_requires_grad(self.puz_cont_v_decoder, req_grad)
        
