import numpy as np
import pytest
import torch
from c1resnet.model import ResNet1D
from c1resnet.engine import environment, seed_all


def test_reference_architecture_full_length_forward_backward(config):
    environment("cpu", cpu_threads=2)
    model = ResNet1D(config["architecture"])
    x = torch.randn(2, 6, 8000)
    logits = model(x)
    assert logits.shape == (2, 12)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 11]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    description = model.description()
    assert description["main_path_convolutions"] == 9
    assert description["receptive_field_samples"] == 1023
    assert description["receptive_field_ms"] == 25.575
    assert description["temporal_downsampling"] is False


def test_seeded_initialization_reproducible(config):
    seed_all(42)
    a = ResNet1D(config["architecture"])
    seed_all(42)
    b = ResNet1D(config["architecture"])
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_evaluation_does_not_update_batchnorm(config):
    model = ResNet1D(config["architecture"]).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model(torch.randn(2, 6, 128))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_wrong_channel_count_rejected(config):
    with pytest.raises(ValueError):
        ResNet1D(config["architecture"])(torch.randn(2, 3, 128))


def test_even_kernel_requires_explicit_architecture_change(config):
    config["architecture"]["kernels"][0] = 8
    with pytest.raises(ValueError):
        ResNet1D(config["architecture"])


def test_cuda_request_never_silently_uses_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA is present")
    with pytest.raises(ValueError, match="CUDA requested"):
        environment("cuda")
