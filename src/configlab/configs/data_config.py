from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from .base import TargetConfig


@dataclass
class DataPrepareConfig(TargetConfig):
    """Data preparation configuration object that contains all the necessary information for data preparation."""

    _target_: str = "configlab.data.data_prepare.mnist_prepare"
    _partial_: bool = True
    root: str = "data"


@dataclass
class DataConfig(TargetConfig):
    """Data configuration object that contains all the necessary information of the data."""

    _target_: str = "configlab.data.mnist_datamodule.MNISTDataModule"
    data_prepare_func: Any = "${data.data_prepare_func}"
    batch_size: int = 512
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = True


def register_data_configs() -> None:
    """Register data configs in the config."""
    cs = ConfigStore.instance()
    cs.store(group="data/data_prepare_func", name="default", node=DataPrepareConfig)
    cs.store(group="data", name="mnist", node=DataConfig)
