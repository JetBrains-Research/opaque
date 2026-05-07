"""Tests for opaque.denoising.disk (DiSK)."""

import dataclasses

import pytest
import torch

from opaque.denoising import DenoiserState, DiskDenoiserState, disk_denoiser
from opaque.utils.per_group import PerGroup


def _scalar_kalman_step(
    y: float, est: float, var: float, r: float, q: float
) -> tuple[float, float]:
    prior_var = var + q
    k = prior_var / (prior_var + r)
    new_est = est + k * (y - est)
    new_var = (1.0 - k) * prior_var
    return new_est, new_var


def test_known_sequence_1d():
    # R=1, Q=0.01  <=>  noise_stddev=1.0 (sigma^2=R)
    r, q = 1.0, 0.01
    template = torch.zeros(1)
    observations = [0.5, -0.2, 0.1, 0.0, 0.3]
    est_f, var_f = 0.0, r + q
    expected = []
    for y in observations:
        est_f, var_f = _scalar_kalman_step(y, est_f, var_f, r, q)
        expected.append(est_f)

    denoise, st = disk_denoiser(
        template,
        noise_stddev=1.0,
        process_stddev=q**0.5,
        dtype=torch.float64,
    )
    actual = []
    for y in observations:
        y_t = torch.tensor([y], dtype=torch.float64)
        filt, st = denoise(y_t, st)
        actual.append(filt.item())

    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert abs(a - b) < 1e-10


def test_state_shape_matches_template():
    template = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    denoise, st = disk_denoiser(template, noise_stddev=1.0, process_stddev=0.1**0.5)
    assert isinstance(st, DiskDenoiserState)
    assert isinstance(st, DenoiserState)
    assert set(st._estimate.keys()) == {"a", "b"}
    assert st._estimate["a"].shape == (2, 3)
    assert st._error_var["a"].shape == (2, 3)
    noisy = {"a": torch.randn(2, 3), "b": torch.randn(4)}
    out, st2 = denoise(noisy, st)
    assert out["a"].shape == (2, 3)
    assert out["b"].shape == (4,)
    assert st2._step_counter == 1


def test_large_q_passthrough():
    """When Q >> R, gain is near 1 and filtered value tracks observation."""
    template = torch.zeros(1)
    # R = 1e-6  =>  noise_stddev = 1e-3
    denoise, st = disk_denoiser(
        template,
        noise_stddev=1e-3,
        process_stddev=(1e6) ** 0.5,
        dtype=torch.float64,
    )
    y = torch.tensor([3.14], dtype=torch.float64)
    out, _ = denoise(y, st)
    assert abs(out.item() - 3.14) < 0.01


def test_small_q_heavy_smoothing():
    """When Q << R, first update pulls estimate only partway toward y."""
    template = torch.zeros(1)
    r, q = 1.0, 1e-6
    denoise, st = disk_denoiser(
        template,
        noise_stddev=1.0,
        process_stddev=q**0.5,
        dtype=torch.float64,
    )
    y = torch.tensor([10.0], dtype=torch.float64)
    out, _ = denoise(y, st)
    init_var = r + q
    prior_var = init_var + q
    k = prior_var / (prior_var + r)
    expected = k * 10.0
    assert abs(out.item() - expected) < 1e-8


def test_pytree_independent_leaves():
    template = {"w": torch.zeros(1), "b": torch.zeros(1)}
    denoise, st = disk_denoiser(
        template, noise_stddev=1.0, process_stddev=0.1, dtype=torch.float64
    )
    noisy = {"w": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    out, _ = denoise(noisy, st)
    assert out["w"].item() != out["b"].item()


def test_noise_stddev_override_changes_gain():
    template = torch.zeros(1)
    denoise, st = disk_denoiser(
        template, noise_stddev=1.0, process_stddev=0.1, dtype=torch.float64
    )
    y = torch.tensor([5.0], dtype=torch.float64)
    out1, st1 = denoise(y, st)
    # Larger sigma => larger R => different gain
    out2, _ = denoise(y, st1, noise_stddev=10.0)
    assert out1.item() != out2.item()


def test_per_group_noise_stddev():
    groups = {"w": "g", "b": "g"}
    pg_std = PerGroup(groups=groups, values={"g": 1.0})
    template = {"w": torch.zeros(1), "b": torch.zeros(1)}
    denoise, st = disk_denoiser(
        template, noise_stddev=pg_std, process_stddev=0.1, dtype=torch.float64
    )
    noisy = {"w": torch.tensor([1.0]), "b": torch.tensor([1.0])}
    out, _ = denoise(noisy, st)
    assert abs(out["w"].item() - out["b"].item()) < 1e-10


def test_frozen_state():
    template = torch.zeros(1)
    _, st = disk_denoiser(template, noise_stddev=1.0, process_stddev=0.1**0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st._step_counter = 99  # type: ignore[misc]


def test_invalid_noise_stddev():
    template = torch.zeros(1)
    with pytest.raises(ValueError, match="positive"):
        disk_denoiser(template, noise_stddev=0.0, process_stddev=0.1)


def test_invalid_process_stddev():
    template = torch.zeros(1)
    with pytest.raises(ValueError, match="non-negative"):
        disk_denoiser(template, noise_stddev=1.0, process_stddev=-1.0)
