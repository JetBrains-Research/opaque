"""Tests for mf_noise with PerGroup clipping bounds."""

from __future__ import annotations

import math

import pytest
import torch

from opaque.noise_allocation import per_group_noise_stddev
from opaque.types import PerGroup, clipped
from opaque.types import NoisedPytree, SecondMomentClippingOutput
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.random import key


def _make_pg_two_groups() -> PerGroup:
    """w -> attn (B=1), b -> mlp (B=2)."""
    return PerGroup(
        groups={"w": "attn", "b": "mlp"},
        values={"attn": 1.0, "mlp": 2.0},
    )


class TestMfNoisePerGroupSingleStream:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_returns_per_group_noise_stddev(self, grad_template):
        strategy = band_mf_strategy(n_steps=20, bands=4, momentum=0.9)
        nm = 1.5
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=nm,
            key=key(7),
        )
        pg = _make_pg_two_groups()
        grads = clipped({"w": torch.randn(4, 3), "b": torch.randn(4)}, max_norm=pg)
        out, st = noise_fn(grads, state)
        assert isinstance(out, NoisedPytree)
        assert isinstance(out.noise_stddev, PerGroup)
        expected = per_group_noise_stddev(pg, nm)
        assert out.noise_stddev.groups == expected.groups
        for g in expected.values:
            assert out.noise_stddev.values[g] == pytest.approx(expected.values[g])

    def test_mlp_group_has_larger_noise_stddev_than_attn(self, grad_template):
        strategy = identity_strategy()
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
        )
        pg = _make_pg_two_groups()
        grads = clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        assert isinstance(out.noise_stddev, PerGroup)
        s_attn = out.noise_stddev.values["attn"]
        s_mlp = out.noise_stddev.values["mlp"]
        assert s_mlp == pytest.approx(s_attn * math.sqrt(2.0), rel=1e-9)

    def test_constant_max_norm_latch_pergroup_mismatch(self, grad_template):
        strategy = band_mf_strategy(n_steps=10, bands=3, momentum=0.9)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(0),
        )
        pg1 = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 1.0},
        )
        pg2 = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 2.0},
        )
        _, state = noise_fn(
            clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg1),
            state,
        )
        with pytest.raises(ValueError, match="varying"):
            noise_fn(
                clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg2),
                state,
            )

    def test_constant_max_norm_latch_kind_mismatch(self, grad_template):
        strategy = band_mf_strategy(n_steps=10, bands=3, momentum=0.9)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(1),
        )
        _, state = noise_fn(
            clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=0.5),
            state,
        )
        pg = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 1.0},
        )
        with pytest.raises(ValueError, match="varying"):
            noise_fn(
                clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg),
                state,
            )

    def test_per_group_non_dict_grads_raises(self, grad_template):
        strategy = identity_strategy()
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(2),
        )
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        # tree of tuples is not a flat dict — PerGroup stddev cannot resolve keys
        bad = clipped((torch.zeros(3), torch.zeros(3)), max_norm=pg)
        with pytest.raises(TypeError, match="flat dict"):
            noise_fn(bad, state)

    @pytest.mark.parametrize(
        "make_strategy",
        [
            lambda: band_mf_strategy(n_steps=15, bands=3, momentum=0.9),
            lambda: blt_strategy(n_steps=15, min_sep=15, momentum=0.9),
            lambda: bisr_strategy(bandwidth=3, n_steps=15, min_sep=15, momentum=0.9),
            lambda: bsr_strategy(
                bandwidth=4, n_steps=15, min_sep=15, alpha=1.0, beta=0.9
            ),
            lambda: lambda_cgd_strategy(0.85, n_steps=15, min_sep=15),
            lambda: identity_strategy(),
        ],
    )
    def test_per_group_runs_all_strategies(self, grad_template, make_strategy):
        strategy = make_strategy()
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=0.8,
            key=key(99),
        )
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": 0.5, "b": 1.5},
        )
        grads = clipped({"w": torch.randn(4, 3), "b": torch.randn(4)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        assert isinstance(out.noise_stddev, PerGroup)
        assert not torch.allclose(out.pytree["w"], grads.pytree["w"])


class TestMfNoisePerGroupPairedStream:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_paired_returns_per_group_stddevs(self, grad_template):
        strategy = band_mf_strategy(n_steps=20, bands=4, momentum=0.9)
        second = band_mf_strategy(n_steps=20, bands=4, momentum=0.99)
        nm = 1.2
        c1 = float(strategy._max_column_norm)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=nm,
            key=key(11),
            second_moment_strategy=second,
        )
        zeta = 0.05
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": zeta, "b": zeta * 2},
        )
        sq = {k: v * v for k, v in {"w": torch.ones(4, 3), "b": torch.ones(4)}.items()}
        sq_pg = pg * pg
        paired = SecondMomentClippingOutput(
            grads=clipped({"w": torch.ones(4, 3), "b": torch.ones(4)}, max_norm=pg),
            squared_grads=clipped(sq, max_norm=sq_pg),
        )
        out, _ = noise_fn(paired, state)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        # Joint Mahalanobis on encoded sensitivities equals (c1 / nm)²
        c2 = float(second._max_column_norm)
        s1 = out.noisy_grads.noise_stddev
        s2 = out.noisy_squared_grads.noise_stddev
        mahal = 0.0
        for param_key in ("w", "b"):
            b1 = pg.for_key(param_key)
            b2 = sq_pg.for_key(param_key)
            d1 = b1 * c1
            d2 = b2 * c2
            mahal += (d1 / s1.for_key(param_key)) ** 2 + (
                d2 / s2.for_key(param_key)
            ) ** 2
        assert mahal == pytest.approx((c1 / nm) ** 2, rel=1e-9)


class TestMfNoisePerGroupMahalanobisSingleStream:
    """Encoded Mahalanobis budget with per-group IID base stddev."""

    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(3), "b": torch.zeros(3)}

    def test_mahalanobis_equals_c1_over_nm_squared(self, grad_template):
        strategy = band_mf_strategy(n_steps=12, bands=3, momentum=0.9)
        nm = 0.75
        c1 = float(strategy._max_column_norm)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=nm,
            key=key(5),
        )
        zeta = 0.08
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": zeta, "b": zeta * 3},
        )
        grads = clipped({"w": torch.zeros(3), "b": torch.zeros(3)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        sigma = out.noise_stddev
        assert isinstance(sigma, PerGroup)
        acc = 0.0
        for param_key in ("w", "b"):
            b_g = pg.for_key(param_key)
            sens = b_g * c1
            acc += (sens / sigma.for_key(param_key)) ** 2
        assert acc == pytest.approx((c1 / nm) ** 2, rel=1e-9)
