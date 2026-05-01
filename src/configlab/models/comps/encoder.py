import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class CNNEncoder(nn.Module):
    """A simple CNN encoder for image data."""

    def __init__(self, output_dim: int) -> None:
        """A simple CNN encoder for image data."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.fc = nn.Linear(64 * 5 * 5, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN encoder."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.2)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MLPEncoder(nn.Module):
    """A simple MLP encoder for tabular data."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """A simple MLP encoder for tabular data."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP encoder."""
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)

        return self.fc2(x)
