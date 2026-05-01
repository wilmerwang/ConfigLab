from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from .base import TargetConfig


@dataclass
class CNNEncoderConfig(TargetConfig):
    """Configuration for the CNN encoder."""

    _target_: str = "configlab.models.comps.encoder.CNNEncoder"
    output_dim: int = 128


@dataclass
class MLPEncoderConfig(TargetConfig):
    """Configuration for the MLP encoder."""

    _target_: str = "configlab.models.comps.encoder.MLPEncoder"
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 128


@dataclass
class MLPHeadConfig(TargetConfig):
    """Configuration for the head."""

    _target_: str = "configlab.models.comps.head.MLPHead"
    input_dim: int = 128
    output_dim: int = 10


@dataclass
class LitMNISTConfig(TargetConfig):
    """Configuration for the LitMNIST model."""

    _target_: str = "configlab.models.mnist_modelmodule.LitMNIST"
    encoder: Any = "${encoder}"
    head: Any = "${head}"


def register_model_configs() -> None:
    """Register models in the config."""
    cs = ConfigStore.instance()
    cs.store(group="encoder", name="cnn", node=CNNEncoderConfig)
    cs.store(group="encoder", name="mlp", node=MLPEncoderConfig)
    cs.store(group="head", name="mlp", node=MLPHeadConfig)
    cs.store(group="model", name="lit_mnist", node=LitMNISTConfig)
