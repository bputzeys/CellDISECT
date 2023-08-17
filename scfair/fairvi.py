import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
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
from .fairvi_dataloader import FairVIDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for hyperparameter tuning
# from ray import tune
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
    >>> FairVI.setup_anndata(adata, batch_key="batch")
    >>> vae = FairVI(adata)
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
        idx_cf_tensor_path: str = '',
        cell_to_idx: torch.Tensor = None,
        idx_to_cell: torch.Tensor = None,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.idx_cf_tensor_path = idx_cf_tensor_path

        if cell_to_idx is None:
            cell_to_idx = torch.tensor(list(range(adata.n_obs))).to(device)
        self.cell_to_idx = cell_to_idx
        if idx_to_cell is None:
            idx_to_cell = torch.tensor(list(range(adata.n_obs))).to(device)
        self.idx_to_cell = idx_to_cell

        self._data_loader_cls = FairVIDataLoader

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent_shared=n_latent_shared,
            n_latent_attribute=n_latent_attribute,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
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

    # call this method after training the model with this held-out:
    # covs[cov_idx] = cov_value_cf, covs[others_idx] = adata.obs[others_idx]
    @torch.no_grad()
    def predict_given_covs(
        self,
        adata: AnnData,  # source anndata with fixed cov values
        cats: List[str],
        cov_idx: int,  # index in cats starting from 0
        cov_value_cf,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:

        self._check_if_trained(warn=False)

        adata_cf = adata.copy()
        cov_name = cats[cov_idx]
        adata_cf.obs[cov_name] = pd.Categorical([cov_value_cf for _ in adata_cf.obs[cov_name]])

        FairVI.setup_anndata(
            adata_cf,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )

        adata_cf = self._validate_anndata(adata_cf)

        scdl = self._make_data_loader(
            adata=adata_cf, batch_size=batch_size
        )

        px_cf_mean_list = []

        for tensors in scdl:

            px_cf = self.module.sub_forward(idx=cov_idx+1, x=tensors[REGISTRY_KEYS.X_KEY][0].to(device),
                                            cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY][0].to(device))

            px_cf_mean_list.append(px_cf.mean)

        px_cf_mean_tensor = torch.cat(px_cf_mean_list, dim=0)
        px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)

        return px_cf_mean_pred

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        nullify_cat_covs_indices: Optional[List[int]] = None,
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
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs,
                                            nullify_cat_covs_indices=nullify_cat_covs_indices,
                                            nullify_shared=nullify_shared)

            latent += [outputs["z_concat"].cpu()]

        return torch.cat(latent).numpy()

    def _make_data_loader(
            self,
            adata: AnnData,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
            shuffle: bool = False,
            data_loader_class=None,
            **data_loader_kwargs,
    ):
        """Create a AnnDataLoader object for data iteration.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        shuffle
            Whether observations are shuffled each iteration though
        data_loader_class
            Class to use for data loader
        data_loader_kwargs
            Kwargs to the class-specific data loader class
        """
        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            raise AssertionError(
                "AnnDataManager not found. Call `self._validate_anndata` prior to calling this function."
            )

        adata_manager.idx_cf_tensor_path = self.idx_cf_tensor_path
        adata_manager.cell_to_idx = self.cell_to_idx
        adata_manager.idx_to_cell = self.idx_to_cell

        adata = adata_manager.adata

        if batch_size is None:
            batch_size = settings.batch_size
        if indices is None:
            indices = np.arange(adata.n_obs)
        if data_loader_class is None:
            data_loader_class = self._data_loader_cls

        if "num_workers" not in data_loader_kwargs:
            data_loader_kwargs.update({"num_workers": settings.dl_num_workers})

        dl = data_loader_class(
            adata_manager,
            shuffle=shuffle,
            indices=indices,
            batch_size=batch_size,
            **data_loader_kwargs,
        )
        return dl

    # @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = True,
        train_size: float = 0.8,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        plan_kwargs: Optional[dict] = None,
        cf_weight: Tunable[Union[float, int]] = 1,  # RECONST_LOSS_X_CF weight
        alpha: Tunable[Union[float, int]] = 1,  # KL Zi weight
        clf_weight: Tunable[Union[float, int]] = 50,  # Si classifier weight
        adv_clf_weight: Tunable[Union[float, int]] = 10,  # adversarial classifier weight
        adv_period: Tunable[int] = 1,  # adversarial training period
        mode: Tunable[Tuple[int]] = (0,),  # training mode
        cf_mode: Tunable[int] = 0,  # counterfactual mode (semi auto-encoding + detach options)
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
        cf_weight
            RECONST_LOSS_X_CF weight
        alpha
            KL Zi weight
        clf_weight
            Si classifier weight
        adv_clf_weight
            adversarial classifier weight
        adv_period
            adversarial training period
        mode
            training mode
        cf_mode
            counterfactual mode (semi auto-encoding + detach options)
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        self.module.set_require_grad(mode)

        self.adata_manager.idx_cf_tensor_path = self.idx_cf_tensor_path
        self.adata_manager.cell_to_idx = self.cell_to_idx
        self.adata_manager.idx_to_cell = self.idx_to_cell

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
        training_plan = self._training_plan_cls(self.module,
                                                cf_weight=cf_weight,
                                                alpha=alpha,
                                                clf_weight=clf_weight,
                                                adv_clf_weight=adv_clf_weight,
                                                adv_period=adv_period,
                                                mode=mode,
                                                cf_mode=cf_mode,
                                                **plan_kwargs)

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
            **trainer_kwargs,
        )
        return runner()
