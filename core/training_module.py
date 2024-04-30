from typing import Any

import lightning.pytorch as pl
from torch import nn, optim
import torch


class TrainingModule(pl.LightningModule):
    """
    Training module (based on Lightning) that initializes training and implements
    training, validation and test steps.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    loss_function : nn.Module
        Function to compute accuracy between true vs model output
    optimizer : optim.Optimizer
        Training optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer

    def training_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for training datasets."""

        I, Xp, Xn, Y, (N_t, N_rp, N_rn) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred, Cp, Cn, (jp, jn) = self.model(I, Xp, Xn, N_t)
        training_loss = self.loss_fn(Y_pred, Y)

        self.log("training_loss", training_loss)
        return training_loss

    def validation_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for validation datasets."""

        I, Xp, Xn, Y, (N_t, N_rp, N_rn) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred, Cp, Cn, (jp, jn) = self.model(I, Xp, Xn, N_t)
        validation_accuracy = self.loss_fn(Y_pred, Y)

        self.log("validation_accuracy", validation_accuracy)
        return validation_accuracy

    def test_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for test datasets."""

        I, Xp, Xn, Y, (N_t, N_rp, N_rn) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred, Cp, Cn, (jp, jn) = self.model(I, Xp, Xn, N_t)
        test_accuracy = self.loss_fn(Y_pred, Y)

        self.log("test_accuracy", test_accuracy)
        return test_accuracy

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer for training."""

        return self.optimizer

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
