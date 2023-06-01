import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from scvi.train import TrainingPlan, TrainRunner
# from scvi.utils._docstrings import devices_dsp

from scvi.model.base import RNASeqMixin, VAEMixin, BaseModelClass

logger = logging.getLogger(__name__)


from .fairvae import fairVAE
from .trainingplan import FairVITrainingPlan



from .fairvi_datasplitter import FairVIDataSplitter


# for hyperparameter tuning
from ray import tune
from scvi._decorators import classproperty
from scvi.autotune._types import Tunable, TunableMixin


class FairVI(
    RNASeqMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
    TunableMixin
):
    """
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    """

    _module_cls = fairVAE
    _data_splitter_cls = FairVIDataSplitter
    _training_plan_cls = FairVITrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent_shared: int = 10,
        n_latent_attribute: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        alpha: Tunable[Union[List[float], Tuple[float], float]] = 1.0,  # coef for P(Si|Zi)
        beta: Tunable[float] = 1.0,   # coef for TC term
        **model_kwargs,
    ):
        super().__init__(adata)

        self.alpha = alpha
        self.beta = beta

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent_shared=n_latent_shared,
            n_latent_attribute=n_latent_attribute,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            alpha=alpha,
            beta=beta,
            **model_kwargs,
        )
        self._model_summary_string = (
            "FairVI Model with the following params: \nn_hidden: {}, n_latent_shared: {}, n_latent_attribute: {}"
            ", n_layers: {}, dropout_rate: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent_shared,
            n_latent_attribute,
            n_layers,
            dropout_rate,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """%(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        adata_cf: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,   # must be as large as possible (default(None) = adata.n_obs)
    ) -> AnnData:
        """
        must have adata.X = adata_cf.X (same gene expression)
        return AnnData with AnnData.covs = adata_cf.covs and predicted X

        Parameters
        ----------
        adata
        adata_cf
        indices
        batch_size

        Returns
        -------

        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)

        batch_size = adata.n_obs

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        adata_cf = self._validate_anndata(adata_cf)
        scdl_cf = self._make_data_loader(
            adata=adata_cf, indices=indices, batch_size=batch_size, shuffle=False
        )

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY

        zs, z_shared = [], []
        cont_covs, cat_covs = [], []
        inference_outputs = dict()
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors, for_train=False)
            inference_outputs = self.module.inference(**inference_inputs, for_train=False)
            zs += [inference_outputs["zs"]]
            z_shared += [inference_outputs["z_shared"]]

            cont_covs += [tensors[cont_key] if cont_key in tensors.keys() else None]
            cat_covs += [tensors[cat_key] if cat_key in tensors.keys() else None]

        zs = list(torch.cat(x) for x in zip(*zs))
        z_shared = torch.cat(z_shared)
        cont_covs = torch.cat(cont_covs) if None not in cont_covs else None
        cat_covs = torch.cat(cat_covs) if None not in cat_covs else None

        pred_adata = AnnData()

        for tensors in scdl_cf:
            # zs_cf
            cont_covs_cf = tensors[cont_key] if cont_key in tensors.keys() else None
            cat_covs_cf = tensors[cat_key] if cat_key in tensors.keys() else None
            zs_cf = self.module.construct_zs_cf(zs, cont_covs, cat_covs, cont_covs_cf, cat_covs_cf)
            # decode
            generative_inputs = {
                "z_shared": z_shared,
                "z_shared_cf": z_shared,
                "zs": zs_cf,
                "zs_cf": zs_cf,
                "library": inference_outputs["library"],
                "library_cf": inference_outputs["library_cf"],
                "library_s": inference_outputs["library_s"],
                "cont_covs": cont_covs_cf,
                "cont_covs_cf": cont_covs_cf,
                "cat_covs": cat_covs_cf,
                "cat_covs_cf": cat_covs_cf,
                "indices": inference_outputs["indices"],
                "indices_cf": inference_outputs["indices_cf"]
            }
            generative_outputs = self.module.generative(generative_inputs)
            pred_x = generative_outputs["px"].mean
            pred_adata = AnnData(X=pred_x)
            pred_adata.layers["counts"] = pred_adata.X.copy()
            pred_adata.obs = adata.obs
            pred_adata.var_names = adata.var_names

        return pred_adata

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        nullify_cat_covs_indices: Optional[List[int]] = None,
        nullify_cont_covs_indices: Optional[List[int]] = None,
        nullify_shared: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.
        batch_size
            Batch size to use.
        nullify_cat_covs_indices
            Categorical attributes to nullify in the latent space.
        nullify_cont_covs_indices
            Continuous attributes to nullify in the latent space.
        nullify_shared
            nullify Z_shared
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors, for_train=False)
            outputs = self.module.inference(**inference_inputs,
                                            nullify_cat_covs_indices=nullify_cat_covs_indices,
                                            nullify_cont_covs_indices=nullify_cont_covs_indices,
                                            nullify_shared=nullify_shared,
                                            for_train=False)

            latent += [outputs["z_concat"].cpu()]

        return torch.cat(latent).numpy()


    # @devices_dsp.dedent
        def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        train_size: float = 0.8,
        validation_size: Optional[float] = None,
        batch_size: int = 512,
        early_stopping: bool = True,
        plan_kwargs: Optional[dict] = None,
        mode: int = 0,
        adv_period: int = 1,
        **trainer_kwargs,
    ):
        """Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_use_gpu)s
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        self.module.set_require_grad(mode)

        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = FairVIDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=True,
            batch_size=batch_size,
        )
        training_plan = self._training_plan_cls(self.module, beta=self.beta, mode=mode, adv_period=adv_period, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs['early_stopping_monitor'] = "loss_validation"
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            # accelerator=accelerator,
            # devices=devices,
            **trainer_kwargs,
        )
        return runner()
