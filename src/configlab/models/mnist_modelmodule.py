import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, MetricCollection, MetricTracker
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
)

from configlab.types.model import EncoderProto, HeadProto


class LitMNIST(LightningModule):
    """LitMNIST module."""

    def __init__(
        self,
        encoder: EncoderProto,
        head: HeadProto,
    ) -> None:
        """Initialize a `LitMNIST` module."""
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.save_hyperparameters(logger=False, ignore=["encoder", "head"])
        self._init_metrics()

    def _init_metrics(self) -> None:
        # loss metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # performance metrics
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=10),
                "f1": F1Score(task="multiclass", num_classes=10),
                "auroc": AUROC(task="multiclass", num_classes=10),
                "aupr": AveragePrecision(task="multiclass", num_classes=10),
            },
            prefix="val/",
        )
        self.val_metrics_tracker = MetricTracker(self.val_metrics)
        self.test_metrics = self.val_metrics.clone(prefix="test/")

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass."""
        x = self.encoder(data)
        return self.head(x)

    def model_step(self, data_label: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Shared step for training and validation."""
        data, labels = data_label
        logits = self(data)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, labels

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Increment the validation metrics tracker at the start of each validation epoch."""
        self.val_metrics_tracker.increment()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, logits, labels = self.model_step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        self.val_metrics_tracker.update(logits, labels.int())

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics at the end of each validation epoch."""
        self.log("val/loss", self.val_loss.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.val_loss.reset()

        epoch_metrics = self.val_metrics_tracker.compute()
        best_merics = self.val_metrics_tracker.best_metric()
        self.log_dict(epoch_metrics, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {"/best/".join(k.rsplit("/", 1)): v for k, v in best_merics.items()},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        data, labels = batch
        logits = self(data)

        # update and log metrics
        self.test_metrics.update(logits, labels.int())
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
        lr_scheduler = {"scheduler": scheduler, "monitor": "val/aupr", "interval": "epoch", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
