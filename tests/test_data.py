import pytest
from torch.utils.data import DataLoader, Dataset

from configlab.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize(("batch_size", "batch_size_expected"), [(16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("loader_name", ["train", "val", "test", "predict"])
def test_dataloaders(
    train_dataset: Dataset, test_dataset: Dataset, batch_size: int, batch_size_expected: int, loader_name: str
) -> None:
    """Test the dataloaders of the MNISTDataModule."""
    def data_prepare_func() -> tuple[Dataset, Dataset]:
        return train_dataset, test_dataset

    dm = MNISTDataModule(data_prepare_func, batch_size=batch_size, num_workers=0, pin_memory=False)
    dm.setup()

    assert dm.num_classes == 10

    loader = getattr(dm, f"{loader_name}_dataloader")()

    def _check_loader(loader: DataLoader) -> None:
        assert len(loader) > 0
        assert loader.batch_size == batch_size_expected

        x, y = next(iter(loader))
        assert x.shape[1:] == (1, 28, 28)
        assert y.max() < 10

    _check_loader(loader)
