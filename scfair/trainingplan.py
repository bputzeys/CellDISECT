import random
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Literal, Optional, Union, List, Tuple

import optax
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, LossOutput
JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from scvi.train import TrainingPlan

from .utils import *

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
        classification_ratio: int = 50,
        lr: Tunable[float] = 1e-3,
        weight_decay: Tunable[float] = 1e-5,
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
        mode: Tuple[int] = (0,),
        adv_period: int = 1,
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
        self.mode = mode
        self.adv_period = adv_period

        self.loss_kwargs.update({"classification_ratio": classification_ratio})
        self.loss_kwargs.update({"mode": mode})

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
            random.shuffle(z_list_perm)
            z_shared_perm = z_list_perm[0]
            zs_perm = z_list_perm[1:]
            zs_concat_perm = torch.cat(zs_perm, dim=-1).to(device)
            z_concat_perm = torch.cat([z_shared_perm, zs_concat_perm], dim=-1).to(device)
            # give to adversarial_classifier and compute loss
            true_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat)).to(device)
            false_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat_perm)).to(device)
            loss = -(torch.mean(true_pred[:, 0]) + torch.mean(false_pred[:, 1]))
        else:
            zs_concat = torch.cat(zs, dim=-1).to(device)
            z_concat = torch.cat([z_shared, zs_concat], dim=-1).to(device)
            cls_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat)).to(device)
            loss = torch.mean(cls_pred[:, 0]) - torch.mean(cls_pred[:, 1])

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        full_dataset = batch[0]
        labelled_dataset = batch[1]

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

        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)

        inference_outputs, _, losses = self.forward(
            full_dataset, loss_kwargs=input_kwargs
        )
        z_shared = inference_outputs["z_shared"]
        zs = inference_outputs["zs"]

        if self.current_epoch % self.adv_period == 0 or (TRAIN_MODE.ADVERSARIAL not in self.mode):

            loss = losses[LOSS_KEYS.LOSS]
            # fool classifier if doing adversarial training
            if (TRAIN_MODE.ADVERSARIAL in self.mode) and kappa > 0 and (self.adversarial_classifier is not False):
                fool_loss = self.loss_adversarial_classifier(z_shared, zs, False) * self.beta
                loss += fool_loss * kappa

                self.log("adv_fool_loss_train", fool_loss, on_step=False, on_epoch=True, prog_bar=True)

            self.compute_and_log_metrics(losses, self.train_metrics, "train")

            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()

        if self.adv_period == 1 or (self.current_epoch % self.adv_period != 0 and (TRAIN_MODE.ADVERSARIAL in self.mode)):

            # train adversarial classifier
            if (TRAIN_MODE.ADVERSARIAL in self.mode) and (opt2 is not None):
                loss = self.loss_adversarial_classifier(z_shared, zs, True)

                self.log("adv_loss_train", loss, on_step=False, on_epoch=True, prog_bar=True)

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
        full_dataset = batch[0]
        labelled_dataset = batch[1]

        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)

        inf_outputs, gen_outputs, losses = self.forward(
            full_dataset, loss_kwargs=input_kwargs
        )

        self.compute_and_log_metrics(losses, self.val_metrics, "validation")

        # log adversarial metrics
        z_shared = inf_outputs["z_shared"].detach()
        zs = [z.detach() for z in inf_outputs["zs"]]
        fool_loss = self.loss_adversarial_classifier(z_shared, zs, False)
        adv_loss = self.loss_adversarial_classifier(z_shared, zs, True)
        self.log("adv_fool_loss_validation", fool_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("adv_loss_validation", adv_loss, on_step=False, on_epoch=True, prog_bar=True)


        results = {}
        for key in losses:
            results.update({key: losses[key]})

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
