"""TorchOpt parity checks for the Torch optimizer runtime."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adamw

torchopt = pytest.importorskip("torchopt")


@pytest.fixture
def params() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        {"lr": 0.1, "betas": (0.85, 0.99), "eps": 1e-6, "weight_decay": 0.0},
        {"lr": 5e-4, "weight_decay": 0.1},
    ],
    ids=["default", "high_lr_no_wd", "heavy_wd"],
)
def test_adamw_matches_torchopt(params: dict[str, torch.Tensor], kwargs: dict) -> None:
    step, state = adamw(params, **kwargs)
    reference = torchopt.adamw(**kwargs)
    reference_state = reference.init(params)

    torch.manual_seed(42)
    for _ in range(10):
        grads = {name: torch.randn_like(value) for name, value in params.items()}
        updates, state = step(grads, state, params=params)
        expected, reference_state = reference.update(
            grads, reference_state, params=params
        )
        for name in params:
            torch.testing.assert_close(updates[name], expected[name])
