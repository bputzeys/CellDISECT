import random
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Literal, Optional, Union, List
from enum import Enum

import optax
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi.autotune._types import Tunable, TunableMixin
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, LossOutput
JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from scvi.train import TrainingPlan

from scvi_dev.nn._utils import *

from scvi.train._metrics import ElboMetric

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for hyperparameter tuning
from ray import tune
from scvi._decorators import classproperty
from scvi.autotune._types import Tunable, TunableMixin


class FairVITrainingPlan(TrainingPlan):
    """Train vaes with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
    scale_adversarial_loss
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following opposite of
        kl warmup.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 3 * 1e-4,
        weight_decay: Tunable[float] = 1e-4,
        n_steps_kl_warmup: Tunable[int] = None,
        n_epochs_kl_warmup: Tunable[int] = 400,
        reduce_lr_on_plateau: Tunable[bool] = True,
        lr_factor: Tunable[float] = 0.4,
        lr_patience: Tunable[int] = 20,
        lr_threshold: Tunable[float] = 0.0,
        lr_scheduler_metric: Literal["loss_validation"] = "loss_validation",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = True,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        beta: Tunable[float] = 1.0,  # coef for TC term
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        self.beta = beta

        if adversarial_classifier is True:
            self.n_output_classifier = 2
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            ).to(device)
        else:
            self.adversarial_classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss

        self.automatic_optimization = False

        # self.adversarial_attribute_decoder =

        # self.epoch_keys = LOSS_KEYS_LIST
        # self.epoch_history = {"mode": [], "epoch": []}
        # for key in self.epoch_keys:
        #     self.epoch_history[key] = []

    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: Optional[int] = None):
        """Initialize metrics and the metric collection."""
        metrics_list = [ElboMetric(met_name, mode, "obs") for met_name in LOSS_KEYS_LIST]
        collection = OrderedDict([(metric.name, metric) for metric in metrics_list])
        return metrics_list, collection

    def initialize_train_metrics(self):
        """Initialize train related metrics."""
        self.elbo_metrics_list_train, self.train_metrics = \
            self._create_elbo_metric_components(mode="train", n_total=self.n_obs_training)

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        self.elbo_metrics_list_val, self.val_metrics = \
            self._create_elbo_metric_components(mode="validation", n_total=self.n_obs_validation)

    @torch.inference_mode()
    def compute_and_log_metrics(
            self,
            loss_output: dict,
            metrics: Dict[str, ElboMetric],
            mode: str,
    ):
        """Computes and logs metrics.

        Parameters
        ----------
        loss_output
            LossOutput dict from scvi-tools module
        metrics
            Dictionary of metrics to update
        mode
            Postfix string to add to the metric name of
            extra metrics
        """

        for met_name in LOSS_KEYS_LIST:
            metrics[f"{met_name}_{mode}"] = loss_output[met_name]
            if isinstance(loss_output[met_name], dict):
                self.log_dict(
                    loss_output[met_name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True
                )
            else:
                self.log(
                    f"{met_name}_{mode}",
                    loss_output[met_name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True
                )

    # @property
    # def epoch_keys(self):
    #     """Epoch keys getter."""
    #     return self._epoch_keys
    #
    # @epoch_keys.setter
    # def epoch_keys(self, epoch_keys: List):
    #     self._epoch_keys.extend(epoch_keys)

    # def permute_z(self, z_list):
    #     z_perm_list = []
    #     idx = torch.randperm(len(z_list))
    #     z_perm_list = z_list[idx]
    #     # for j in range(len(z_list)):
    #     #     idx = torch.randperm(z_list[j].size(0)).to(device)
    #     #     z_perm_list.append(z_list[j][idx, :].detach())
    #     return z_perm_list

    def loss_adversarial_classifier(self, z_shared, zs, compute_for_classifier=True):
        """Loss for adversarial classifier."""
        if compute_for_classifier:
            # detach z
            zs = [zs_i.detach() for zs_i in zs]
            z_shared = z_shared.detach()
            zs_concat = torch.cat(zs, dim=-1).to(device)
            z_concat = torch.cat([z_shared, zs_concat], dim=-1).to(device)
            # permute z
            z_list_perm = [z_shared] + zs
            # z_list_perm = self.permute_z(z_list_perm)
            random.shuffle(z_list_perm)
            z_shared_perm = z_list_perm[0]
            zs_perm = z_list_perm[1:]
            zs_concat_perm = torch.cat(zs_perm, dim=-1).to(device)
            z_concat_perm = torch.cat([z_shared_perm, zs_concat_perm], dim=-1).to(device)
            # mix permuted z and unpermuted z
            z_concat_mixed = torch.cat([z_concat, z_concat_perm], dim=0).to(device)
            perm_batch_idx = torch.randperm(z_concat_mixed.size(0)).to(device)
            z_concat_mixed = z_concat_mixed[perm_batch_idx, :]
            # give to adversarial_classifier and compute loss
            cls_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat_mixed)).to(device)
            true_idx = torch.tensor([i for i in range(int(z_concat_mixed.size(0))) if perm_batch_idx[i] < z_concat.size(0)]).to(device)
            false_idx = torch.tensor([i for i in range(int(z_concat_mixed.size(0))) if perm_batch_idx[i] >= z_concat.size(0)]).to(device)
            true_pred = torch.index_select(cls_pred, dim=0, index=true_idx).to(device)
            false_pred = torch.index_select(cls_pred, dim=0, index=false_idx).to(device)
            loss = -(torch.mean(true_pred[:, 0]) + torch.mean(false_pred[:, 1])) / 2
        else:
            zs_concat = torch.cat(zs, dim=-1).to(device)
            z_concat = torch.cat([z_shared, zs_concat], dim=-1).to(device)
            cls_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat)).to(device)
            loss = torch.mean(cls_pred[:, 0]) - torch.mean(cls_pred[:, 1])

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, losses = self.forward(
            batch, loss_kwargs=self.loss_kwargs
        )
        z_shared = inference_outputs["z_shared"]
        zs = inference_outputs["zs"]
        loss = losses[LOSS_KEYS.LOSS]
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z_shared, zs, False) * self.beta
            loss += fool_loss * kappa

            self.log("adversarial_loss_train", fool_loss, on_epoch=True, prog_bar=True)

        # log metrics
        # for key in self.epoch_keys:
        #     if isinstance(losses[key], dict):
        #         self.log_dict(losses[key], on_epoch=True, prog_bar=True)
        #     else:
        #         self.log(f"train_{key}", losses[key], on_epoch=True, prog_bar=True)
        # self.log("train_loss", loss, on_epoch=True)

        self.compute_and_log_metrics(losses, self.train_metrics, "train")

        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z_shared, zs, True)

            # tune
            # tune.report({"loss_adversarial": loss})

            # self.log("train_adversarial_loss", on_epoch=True, prog_bar=True)

            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

        results = {}
        for key in LOSS_KEYS_LIST:
            results.update({key: losses[key]})
        return results

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inf_outputs, gen_outputs, losses = self.forward(batch)

        self.compute_and_log_metrics(losses, self.val_metrics, "validation")

        results = {}
        for key in losses:
            results.update({key: losses[key]})

        # add new metrics:
        # r2_mean, r2_var = self.module.r2_metric(batch, gen_outputs)
        # results.update({"generative_mean_accuracy": r2_mean})
        # results.update({"generative_var_accuracy": r2_var})
        # results.update({"biolord_metric": biolord_metric(r2_mean, r2_var)})

        # log metrics
        # for key in self.epoch_keys:
        #     if isinstance(results[key], dict):
        #         self.log_dict(results[key], on_epoch=True, prog_bar=True)
        #     else:
        #         self.log(f"val_{key}", results[key], on_epoch=True, prog_bar=True)

        return results

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        # Update the learning rate via scheduler steps.
        if (
            not self.reduce_lr_on_plateau
            or "validation" not in self.lr_scheduler_metric
        ):
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(
                lambda p: p.requires_grad, self.adversarial_classifier.parameters()
            )
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1

    # @classproperty
    # def _tunables(cls):
    #     return [cls.__init__, cls.training_step]
    #
    # @classproperty
    # def _metrics(cls):
    #     return ["loss_adversarial"]

