from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def build_data_module(config: DictConfig, **kwargs) -> LightningDataModule:
    """Build the data module from the configuration."""
    return instantiate(config.data, **kwargs)


def build_model_module(config: DictConfig, **kwargs) -> LightningModule:
    """Build the model module from the configuration."""
    return instantiate(config.model, **kwargs)


def build_components(config: DictConfig, key: str) -> list:
    """Generic builder for components like callbacks, loggers, etc."""
    cfg = config.get(key)

    if not cfg:
        return []

    if isinstance(cfg, DictConfig) and "_target_" in cfg:
        return [instantiate(cfg)]

    if isinstance(cfg, DictConfig):
        return [instantiate(v) for v in cfg.values() if isinstance(v, DictConfig) and "_target_" in v]
    if isinstance(cfg, list):
        return [instantiate(item) for item in cfg if isinstance(item, DictConfig) and "_target_" in item]

    raise ValueError(f"Invalid configuration for {key}: {cfg}")


def build_callbacks(config: DictConfig) -> list[Callback]:
    """Build the callbacks from the configuration."""
    return build_components(config, "callbacks")


def build_loggers(config: DictConfig) -> list[Logger]:
    """Build the loggers from the configuration."""
    return build_components(config, "logger")


def build_trainer(config: DictConfig, **kwargs) -> Trainer:
    """Build the trainer from the configuration."""
    return instantiate(config.trainer, **kwargs)
