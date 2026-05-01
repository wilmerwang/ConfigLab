from hydra import compose, initialize

from configlab.configs.data_config import register_data_configs
from configlab.configs.model_config import register_model_configs
from configlab.configs.training_config import register_training_configs

register_data_configs()
register_model_configs()
register_training_configs()


def test_config() -> None:
    """Test that the config can be composed."""
    with initialize(version_base="1.1", config_path="../configs"):
        cfg = compose(config_name="config")

    assert "data" in cfg
    assert "model" in cfg
    assert "trainer" in cfg
    assert "callbacks" in cfg
    assert "logger" not in cfg  # logger is set to null by default, so it should not be in the config

    with initialize(version_base="1.1", config_path="../configs"):
        cfg = compose(config_name="config", overrides=["task=train"])
    assert cfg.task_name == "train"
    assert "logger" in cfg
