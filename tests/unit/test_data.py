import pytest
from torch.utils.data import DataLoader, Dataset

from configlab.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize(("batch_size", "batch_size_expected"), [(16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("loader_name", ["train", "val", "test", "predict"])
def test_dataloaders(
    train_dataset: Dataset, test_dataset: Dataset, batch_size: int, batch_size_expected: int, loader_name: str
) -> None:
    """Test the dataloaders of the MNISTDataModule."""
    dm = MNISTDataModule(train_dataset, test_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
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


def test_setup_is_idempotent(train_dataset: Dataset, test_dataset: Dataset) -> None:
    """Test repeated setup calls do not split the train dataset again."""
    dm = MNISTDataModule(train_dataset, test_dataset, batch_size=16, num_workers=0, pin_memory=False)

    dm.setup()
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)

    dm.setup()

    assert len(dm.train_dataset) == train_len
    assert len(dm.val_dataset) == val_len
