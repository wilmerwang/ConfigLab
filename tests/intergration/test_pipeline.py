import pytest
from _pytest.monkeypatch import MonkeyPatch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import Dataset

from configlab.pipelines.runner import run_pipeline
from configlab.utils.hydra_utils import register_configs


@pytest.fixture
def cfg(train_dataset: Dataset, test_dataset: Dataset, tmp_path_factory: pytest.TempPathFactory) -> DictConfig:
    """Fixture to prepare the configuration for the training pipeline."""
    register_configs()

    def _cfg(overrides: list[str]) -> DictConfig:
        with initialize(version_base="1.1", config_path="../../configs"):
            return compose(
                config_name="config",
                overrides=[
                    f"paths.data_dir={tmp_path_factory.mktemp('data')}",
                    f"paths.output_dir={tmp_path_factory.mktemp('output')}",
                    f"paths.work_dir={tmp_path_factory.mktemp('work')}",
                    f"paths.cache_dir={tmp_path_factory.mktemp('cache')}",
                    f"paths.log_dir={tmp_path_factory.mktemp('log')}",
                    "data.persistent_workers=False",
                    "debug=default",
                    *overrides,
                ],
            )

    return _cfg


@pytest.fixture(autouse=True)
def mock_mnist(monkeypatch: MonkeyPatch, train_dataset: Dataset, test_dataset: Dataset) -> None:
    """Fixture to mock the MNIST dataset preparation."""

    def _fake_mnist_prepare(root: str = "data") -> tuple[Dataset, Dataset]:
        return train_dataset, test_dataset

    monkeypatch.setattr("configlab.pipelines.runner.mnist_prepare", _fake_mnist_prepare)


@pytest.mark.slow
def test_train(cfg: DictConfig) -> None:
    """Test the training pipeline."""
    config = cfg(["task=train"])
    results = run_pipeline(config)

    assert "train_metrics" in results


@pytest.mark.slow
def test_test(cfg: DictConfig) -> None:
    """Test the testing pipeline."""
    config = cfg(["task=test"])
    results = run_pipeline(config)

    assert "test_metrics" in results


@pytest.mark.slow
def test_predict(cfg: DictConfig) -> None:
    """Test the prediction pipeline."""
    config = cfg(["task=predict"])
    results = run_pipeline(config)

    assert "predict_output" in results
