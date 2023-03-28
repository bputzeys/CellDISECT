from typing import Callable, Dict, Iterable, Literal, Optional, Union

import optax
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi.autotune._types import Tunable, TunableMixin
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass
JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from scvi.train import TrainingPlan


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
        lr: Tunable[float] = 1e-3,
        weight_decay: Tunable[float] = 1e-6,
        n_steps_kl_warmup: Tunable[int] = None,
        n_epochs_kl_warmup: Tunable[int] = 400,
        reduce_lr_on_plateau: Tunable[bool] = False,
        lr_factor: Tunable[float] = 0.6,
        lr_patience: Tunable[int] = 30,
        lr_threshold: Tunable[float] = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = True,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
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
        if adversarial_classifier is True:
            self.n_output_classifier = 2
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
        else:
            self.adversarial_classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def permute_z(self, z_list):
        z_perm_list = []
        for j in range(len(z_list)):
            idx = torch.randperm(z_list[j].size(0))
            z_perm_list.append(z_list[j][idx, :].detach())
        return z_perm_list

    def loss_adversarial_classifier(self, z_shared, zs, compute_for_classifier=True):
        """Loss for adversarial classifier."""
        if compute_for_classifier:
            # detach z
            zs = [zs_i.detach() for zs_i in zs]
            z_shared = z_shared.detach()
            zs_concat = torch.cat(zs, dim=-1)
            z_concat = torch.cat([z_shared, zs_concat], dim=-1)
            # permute z
            z_list = [z_shared] + zs
            z_list_perm = self.permute_z(z_list)
            z_shared_perm = z_list_perm[0]
            zs_perm = z_list_perm[1:]
            zs_concat_perm = torch.cat(zs_perm, dim=-1)
            z_concat_perm = torch.cat([z_shared_perm, zs_concat_perm], dim=-1)
            # mix permuted z and unpermuted z
            z_concat_mixed = torch.cat([z_concat, z_concat_perm], dim=0)
            perm_batch_idx = torch.randperm(z_concat_mixed.size(0))
            z_concat_mixed = z_concat_mixed[perm_batch_idx, :]
            # give to adversarial_classifier and compute loss
            cls_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat_mixed))
            true_idx = torch.Tensor([i for i in range(z_concat_mixed.size(0)) if perm_batch_idx[i] < z_concat.size(0)])
            false_idx = torch.Tensor([i for i in range(z_concat_mixed.size(0)) if perm_batch_idx[i] >= z_concat.size(0)])
            loss = -(torch.mean(cls_pred[true_idx, 0]) + torch.mean(cls_pred[false_idx, 1])) / 2
        else:
            zs_concat = torch.cat(zs, dim=-1)
            z_concat = torch.cat([z_shared, zs_concat], dim=-1)
            cls_pred = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z_concat))
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

        inference_outputs, _, scvi_loss = self.forward(
            batch, loss_kwargs=self.loss_kwargs
        )
        z_shared = inference_outputs["z_shared"]
        zs = inference_outputs["zs"]
        loss = scvi_loss.loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z_shared, zs, False)
            loss += fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z_shared, zs, True)
            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
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

