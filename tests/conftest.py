import pytest
import torch
from torch.utils.data import Dataset


class DummyMNISTDataset(Dataset):
    """A dummy dataset that mimics MNIST shape and labels."""

    def __init__(self, num_samples: int = 100) -> None:
        """Initialize the dummy dataset with random data and labels."""
        self.x = torch.randn(num_samples, 1, 28, 28)  # MNIST shape
        self.y = torch.randint(0, 10, (num_samples,))  # 10 classes

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sample and label at the given index."""
        return self.x[idx], self.y[idx]


@pytest.fixture
def train_dataset() -> Dataset:
    """Fixture for the training dataset."""
    return DummyMNISTDataset(num_samples=200)


@pytest.fixture
def test_dataset() -> Dataset:
    """Fixture for the test dataset."""
    return DummyMNISTDataset(num_samples=50)
