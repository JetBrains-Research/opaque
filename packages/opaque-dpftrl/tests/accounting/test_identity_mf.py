"""Tests for :class:`~opaque.dpftrl.accounting.mechanisms.IdentityMf`."""

import math

import pytest

import opaque.dpftrl.accounting as ftrl_acc


@pytest.mark.parametrize(
    ("nm", "sample_rate", "num_steps"),
    [
        (1.0, 0.05, 100),
        (1.8, 0.01, 500),
        (5.4, 0.12, 1),
    ],
)
def test_mf_identity_epsilon_positive(nm, sample_rate, num_steps):
    proc = ftrl_acc.mf_identity(nm, sample_rate=sample_rate, num_steps=num_steps)
    eps = proc.epsilon_at(1e-5)
    assert eps > 0
    assert eps < 5000


def test_mf_identity_zero_noise_is_non_private():
    proc = ftrl_acc.mf_identity(
        0.0, sample_rate=0.05, num_steps=300
    )
    assert math.isinf(proc.epsilon_at(1e-5))


def test_mf_identity_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        ftrl_acc.mf_identity(1.0, sample_rate=1.5, num_steps=10)
    with pytest.raises(ValueError, match="sample_rate"):
        ftrl_acc.mf_identity(1.0, sample_rate=0.0, num_steps=10)


def test_mf_identity_rejects_invalid_num_steps():
    with pytest.raises(ValueError, match="num_steps"):
        ftrl_acc.mf_identity(1.0, sample_rate=0.1, num_steps=0)
