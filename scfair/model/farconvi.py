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
from scvi_dev.module.farcon import FarconVAE

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"
_SCVI_OBSERVED_LIB_SIZE = "_scvi_observed_lib_size"

logger = logging.getLogger(__name__)


class FarconVI(SCVI):
    """
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.FarconVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.FarconVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_FarconVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_FarconVI"] = vae.get_normalized_expression()
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
        sensitive_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",        # new
        alpha=1,                                                                # new
        beta=0.2,                                                               # new
        gamma=1,                                                                # new
        kernel: Literal["student-t", "gaussian"] = "student-t",                 # new
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

        self.module = FarconVAE(
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
            sensitive_likelihood=sensitive_likelihood,
            alpha=alpha, 
            beta=beta,
            gamma=gamma,
            kernel=kernel,
            **model_kwargs,
        )
               
