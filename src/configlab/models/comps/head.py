import torch
from torch import nn


class MLPHead(nn.Module):
    """A simple MLP head for tabular data."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """A simple MLP head for tabular data."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP head."""
        return self.fc1(x)
