from collections.abc import Callable

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class MNISTDataModule(LightningDataModule):
    """LightningDataModule for MNIST dataset."""

    def __init__(
        self,
        data_prepare_func: Callable[[], tuple[Dataset, Dataset]],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize the MNISTDataModule with the given datasets and parameters."""
        super().__init__()
        self.train_dataset, self.test_dataset = data_prepare_func()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 10

    def setup(self, stage: str | None = None) -> None:
        """Setup the datasets for training and testing."""
        # No additional setup is needed since the datasets are already prepared
        num_train_samples = len(self.train_dataset)
        num_val_samples = int(0.1 * num_train_samples)
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [num_train_samples - num_val_samples, num_val_samples]
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test dataset (used for prediction)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
