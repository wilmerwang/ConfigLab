from torch.utils.data import Dataset
from torchvision import datasets, transforms


def mnist_prepare(root: str = "data") -> tuple[Dataset, Dataset]:
    """Prepare the MNIST dataset for training and testing."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset
