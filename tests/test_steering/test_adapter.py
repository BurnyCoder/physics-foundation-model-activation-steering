import pytest
import torch

from gphyt.models.transformer.model import PhysicsTransformer
from gphyt.steering.adapters import HookBackend, PyveneBackend, list_layer_ids


def _build_transformer() -> PhysicsTransformer:
    return PhysicsTransformer(
        num_fields=5,
        hidden_dim=16,
        mlp_dim=32,
        num_heads=4,
        num_layers=2,
        patch_size=(1, 4, 4),
        img_size=(2, 8, 8),
        pos_enc_mode="absolute",
        tokenizer_mode="linear",
        detokenizer_mode="linear",
        dropout=0.0,
        stochastic_depth_rate=0.0,
        use_derivatives=False,
    )


def test_list_layer_ids():
    model = _build_transformer()
    assert list_layer_ids(model) == ["block_out:0", "block_out:1"]


def test_hook_backend_collect_and_noop():
    model = _build_transformer()
    backend = HookBackend()
    x = torch.randn(2, 2, 8, 8, 5)

    activation = backend.collect_activation(model, "block_out:0", x)
    output = model(x)
    noop_output = backend.run_with_noop(model, "block_out:0", x)

    assert activation.shape == (2, 2, 2, 2, 16)
    assert output.shape == noop_output.shape
    assert torch.allclose(output, noop_output, atol=1e-6, rtol=1e-5)


def test_hook_backend_direction_changes_output():
    model = _build_transformer()
    backend = HookBackend()
    x = torch.randn(2, 2, 8, 8, 5)
    baseline = model(x)
    steered = backend.run_with_direction(
        model,
        "block_out:0",
        x,
        direction=torch.ones(16),
        scale=0.5,
    )

    assert not torch.allclose(baseline, steered)


def test_pyvene_backend_noop_parity():
    pytest.importorskip("pyvene")
    model = _build_transformer()
    backend = PyveneBackend()
    x = torch.randn(2, 2, 8, 8, 5)

    baseline = model(x)
    noop = backend.run_with_noop(model, "block_out:0", x)
    activation = backend.collect_activation(model, "block_out:0", x)

    assert activation.shape[-1] == 16
    assert torch.allclose(baseline, noop, atol=1e-6, rtol=1e-5)
