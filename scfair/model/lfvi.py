import logging
from typing import List, Literal, Optional, Union

import numpy as np
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi._types import MinifiedDataType
from scvi.data import AnnDataManager
from scvi.data._constants import _ADATA_MINIFY_TYPE_UNS_KEY, ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.model.utils import get_minified_adata_scrna
from scvi_dev.module._lfvae import LFVAE
from scvi.utils import setup_anndata_dsp

from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin, BaseModelClass

from scvi.model._scvi import SCVI
from scvi_dev.module.lf import LFVAE

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"
_SCVI_OBSERVED_LIB_SIZE = "_scvi_observed_lib_size"

logger = logging.getLogger(__name__)


class LFVI(SCVI):
    """
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.LFVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.LFVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_LFVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_LFVI"] = vae.get_normalized_expression()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        sensitive_likelihood_cat: Literal["zinb", "nb", "poisson"] = "zinb",    # new
        sensitive_likelihood_cont: Literal["normal"] = "normal",                # new
        sensitive_prior_cat: Literal["bernoulli"] = "bernoulli",                # new
        **model_kwargs,
    ):
        super().__init__(adata)
        
        self.n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        self.n_batch = self.summary_stats.n_batch
        self.use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        self.library_log_means, self.library_log_vars = None, None
        if not self.use_size_factor_key and self.minified_data_type is None:
            self.library_log_means, self.library_log_vars = _init_library_size(
                self.adata_manager, self.n_batch
            )

        self.module = LFVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=self.n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=self.n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=self.use_size_factor_key,
            library_log_means=self.library_log_means,
            library_log_vars=self.library_log_vars,
            sensitive_likelihood_cat=sensitive_likelihood_cat,
            sensitive_likelihood_cont=sensitive_likelihood_cont,
            sensitive_prior_cat=sensitive_prior_cat,
            **model_kwargs,
        )
               

    def train(
            self,
            max_epochs: int = 300,
            max_inner_epochs: int = 10,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            plan_kwargs: Optional[dict] = None,
            **trainer_kwargs,
    ):
        """
        Trains the model in an adversarial fashion by fixed l1, l2
        """
#         self.module.l1 = l1
#         self.module.l2 = l2
        # TODO: find optimal l1, l2 by a hyperparameter-tuning technique
        
#         for l1 in [0.1, 0.5, 1, 2, 5]:
#             for l2 in [0.1, 0.5, 1, 2, 5]:

        for l1 in [1]:
            for l2 in [1]:
                
                opt_mode_idx = 0
                self.module.l1 = l1
                self.module.l2 = l2
        
                for epoch in range(max_epochs):
                    self.module.opt_mode = ['min', 'max'][opt_mode_idx]

                    if opt_mode_idx % 2 == 0:
                        self.module.set_requires_grad_encoder(True)
                        self.module.set_requires_grad_x_decoder(True)
                        self.module.set_requires_grad_u_decoder(False)
                    else:
                        self.module.set_requires_grad_encoder(False)
                        self.module.set_requires_grad_x_decoder(False)
                        self.module.set_requires_grad_u_decoder(True)

                    runner = super().train(
                        max_epochs=max_inner_epochs,
                        use_gpu=use_gpu,
                        train_size=train_size,
                        validation_size=validation_size,
                        batch_size=batch_size,
                        plan_kwargs=plan_kwargs,
                        **trainer_kwargs,
                    )
                    opt_mode_idx = 1 - opt_mode_idx

            
        return runner
