import numpy as np
import pytest
import torch
from c1inception.model import InceptionTime1D
from c1inception.engine import environment, seed_all


def test_reference_architecture_full_length_forward_backward(config):
    environment("cpu", cpu_threads=2)
    model = InceptionTime1D(config["architecture"])
    x = torch.randn(2, 6, 8000)
    logits = model(x)
    assert logits.shape == (2, 12)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 11]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    description = model.description()
    assert description["inception_modules"] == 6
    assert description["convolution_layers"] == 32
    assert description["parameters"] == 276012
    assert description["receptive_field_samples"] == 1081
    assert description["receptive_field_ms"] == 27.025
    assert description["temporal_downsampling"] is False


def test_seeded_initialization_reproducible(config):
    seed_all(42)
    a = InceptionTime1D(config["architecture"])
    seed_all(42)
    b = InceptionTime1D(config["architecture"])
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_evaluation_does_not_update_batchnorm(config):
    model = InceptionTime1D(config["architecture"]).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model(torch.randn(2, 6, 128))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_wrong_channel_count_rejected(config):
    with pytest.raises(ValueError):
        InceptionTime1D(config["architecture"])(torch.randn(2, 3, 128))


def test_even_kernel_requires_explicit_architecture_change(config):
    config["architecture"]["kernels"][0] = 8
    with pytest.raises(ValueError):
        InceptionTime1D(config["architecture"])


def test_cuda_request_never_silently_uses_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA is present")
    with pytest.raises(ValueError, match="CUDA requested"):
        environment("cuda")
