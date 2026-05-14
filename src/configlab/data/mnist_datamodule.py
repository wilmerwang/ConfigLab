from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split


class MNISTDataModule(LightningDataModule):
    """LightningDataModule for MNIST dataset."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_seed: int = 42,
    ) -> None:
        """Initialize the MNISTDataModule with the given datasets and parameters."""
        super().__init__()
        self.full_train_dataset = train_dataset
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.split_seed = split_seed

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 10

    def setup(self, stage: str | None = None) -> None:
        """Setup the datasets for training and testing."""
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        num_train_samples = len(self.full_train_dataset)
        num_val_samples = int(0.1 * num_train_samples)
        self.train_dataset, self.val_dataset = random_split(
            self.full_train_dataset,
            [num_train_samples - num_val_samples, num_val_samples],
            generator=Generator().manual_seed(self.split_seed),
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
