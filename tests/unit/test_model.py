import torch

from configlab.models.comps.encoder import CNNEncoder, MLPEncoder
from configlab.models.comps.head import MLPHead


def test_cnn_encoder() -> None:
    """Test the CNNEncoder."""
    encoder = CNNEncoder(output_dim=128)
    x = torch.randn(16, 1, 28, 28)
    output = encoder(x)
    assert output.shape == (16, 128)


def test_mlp_encoder() -> None:
    """Test the MLPEncoder."""
    encoder = MLPEncoder(input_dim=784, hidden_dim=128, output_dim=128)
    x = torch.randn(16, 784)
    output = encoder(x)
    assert output.shape == (16, 128)


def test_encoders_are_deterministic_in_eval_mode() -> None:
    """Test dropout is disabled when encoders are in eval mode."""
    x_cnn = torch.randn(16, 1, 28, 28)
    cnn_encoder = CNNEncoder(output_dim=128).eval()
    assert torch.equal(cnn_encoder(x_cnn), cnn_encoder(x_cnn))

    x_mlp = torch.randn(16, 784)
    mlp_encoder = MLPEncoder(input_dim=784, hidden_dim=128, output_dim=128).eval()
    assert torch.equal(mlp_encoder(x_mlp), mlp_encoder(x_mlp))


def test_mlp_head() -> None:
    """Test the MLPHead."""
    head = MLPHead(input_dim=128, output_dim=10)
    x = torch.randn(16, 128)
    output = head(x)
    assert output.shape == (16, 10)
