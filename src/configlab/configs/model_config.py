from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class CNNEncoderConfig:
    """Configuration for the CNN encoder."""

    _target_: str = "configlab.models.comps.encoder.CNNEncoder"
    output_dim: int = 128


@dataclass
class MLPEncoderConfig:
    """Configuration for the MLP encoder."""

    _target_: str = "configlab.models.comps.encoder.MLPEncoder"
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 128


@dataclass
class MLPHeadConfig:
    """Configuration for the head."""

    _target_: str = "configlab.models.comps.head.MLPHead"
    input_dim: int = 128
    output_dim: int = 10


@dataclass
class AdamOptimizerConfig:
    """Configuration for the Adam optimizer."""

    _target_: str = "torch.optim.Adam"
    _partial_: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class ReduceLROnPlateauSchedulerConfig:
    """Configuration for the ReduceLROnPlateau scheduler."""

    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    _partial_: bool = True
    mode: str = "max"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6


@dataclass
class LitMNISTConfig:
    """Configuration for the LitMNIST model."""

    encoder: Any
    head: Any
    # Mostly fixed; only lr and similar parameters are exposed for hyperparameter search.
    optimizer_factory: Any = AdamOptimizerConfig
    # move optimizer and scheduler out of model for hyperparameter tuning
    scheduler_factory: Any = ReduceLROnPlateauSchedulerConfig
    _target_: str = "configlab.models.mnist_modelmodule.LitMNIST"


def register_model_configs() -> None:
    """Register models in the config."""
    cs = ConfigStore.instance()
    cs.store(group="model/encoder", name="cnn", node=CNNEncoderConfig)
    cs.store(group="model/encoder", name="mlp", node=MLPEncoderConfig)
    cs.store(group="model/head", name="mlp", node=MLPHeadConfig)
    cs.store(group="model", name="lit_mnist", node=LitMNISTConfig)
