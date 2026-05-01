import hydra
from omegaconf import DictConfig

from configlab.pipelines.runner import run_pipeline
from configlab.utils.hydra_utils import register_configs
from configlab.utils.utils import get_metric_value


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """The main entry point."""
    results = run_pipeline(cfg)

    metric_results = {**results.get("train_metrics", {}), **results.get("test_metrics", {})}

    return get_metric_value(metric_results, cfg.get("optimized_metric"))


if __name__ == "__main__":
    register_configs()
    main()
