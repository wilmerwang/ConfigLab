from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Data configuration object that contains all the necessary information of the data."""

    _target_: str = "configlab.data.mnist_datamodule.MNISTDataModule"
    batch_size: int = 512
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = True


def register_data_configs() -> None:
    """Register data configs in the config."""
    cs = ConfigStore.instance()
    cs.store(group="data", name="mnist", node=DataConfig)
