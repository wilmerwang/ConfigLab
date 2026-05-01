from configlab.configs.data_config import register_data_configs
from configlab.configs.model_config import register_model_configs
from configlab.configs.training_config import register_training_configs


def register_configs() -> None:
    """Register all configs."""
    register_data_configs()
    register_model_configs()
    register_training_configs()
