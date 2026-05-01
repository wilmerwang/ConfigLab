from typing import Protocol

from jaxtyping import Float
from torch import Tensor


class EncoderProto(Protocol):
    """Protocol for a model that can be trained and evaluated."""

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d"]:
        """Encode an input tensor and return the resulting tensor."""
        ...


class HeadProto(Protocol):
    """Protocol for a model that can be trained and evaluated."""

    def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b o"]:
        """Encode an input tensor and return the resulting tensor."""
        ...
