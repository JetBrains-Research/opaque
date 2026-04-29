"""Tests for opaque.scheduling.

Cross-checks each public curve against HuggingFace's reference lambdas
pointwise — the multiplier returned by HF's LambdaLR scaled by
``BASE_LR`` must match our schedule at every step in a sweep.
"""

import math

import pytest
import torch

transformers = pytest.importorskip(
    "transformers", reason="transformers required for HF cross-check"
)

from torch.optim import SGD  # noqa: E402
from transformers.optimization import (  # noqa: E402
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from opaque.scheduling import (  # noqa: E402
    constant_schedule,
    cosine_schedule,
    exponential_decay,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    with_restarts,
    with_warmup,
)


BASE_LR = 1e-3


def _hf_lambda(make_scheduler):
    """Extract HF's LambdaLR multiplier function (step -> multiplier)."""
    p = torch.zeros(1, requires_grad=True)
    return make_scheduler(SGD([p], lr=BASE_LR)).lr_lambdas[0]


def _assert_pointwise(ours, hf_lam, steps, *, tol=1e-9):
    """Our schedule must equal hf_lam(step) * BASE_LR at every step."""
    for s in steps:
        assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=tol), (
            f"step={s}: ours={ours(s)} hf={hf_lam(s) * BASE_LR}"
        )


# ---------------------------------------------------------------------------
# constant_schedule
# ---------------------------------------------------------------------------


class TestConstantSchedule:
    def test_returns_value_at_every_step(self):
        sched = constant_schedule(0.5)
        for s in (0, 1, 100, 10_000):
            assert sched(s) == 0.5

    def test_matches_hf_get_constant_schedule(self):
        sched = constant_schedule(BASE_LR)
        hf = _hf_lambda(get_constant_schedule)
        _assert_pointwise(sched, hf, range(0, 1100, 13))


# ---------------------------------------------------------------------------
# linear_schedule
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    def test_starts_at_init(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_reaches_end(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(100) == pytest.approx(0.0, abs=1e-12)

    def test_midpoint(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100)
        assert s(50) == pytest.approx(0.5)

    def test_clamps_after_transition(self):
        s = linear_schedule(1.0, 0.2, transition_steps=100)
        assert s(100) == pytest.approx(0.2)
        assert s(500) == pytest.approx(0.2)

    def test_transition_begin_holds_init(self):
        s = linear_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert s(0) == pytest.approx(1.0)
        assert s(49) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# polynomial_schedule
# ---------------------------------------------------------------------------


class TestPolynomialSchedule:
    def test_power_one_matches_linear(self):
        poly = polynomial_schedule(1.0, 0.0, power=1.0, transition_steps=100)
        lin = linear_schedule(1.0, 0.0, transition_steps=100)
        for s in (0, 25, 50, 75, 100, 200):
            assert poly(s) == pytest.approx(lin(s), abs=1e-12)

    def test_power_two_quadratic(self):
        # frac = 1 - count/T; value = (init - end) * frac^2 + end.
        s = polynomial_schedule(1.0, 0.0, power=2.0, transition_steps=100)
        # At progress=0.5: frac=0.5, frac^2 = 0.25.
        assert s(50) == pytest.approx(0.25)
        # At progress=1: frac=0, value=end.
        assert s(100) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# exponential_decay
# ---------------------------------------------------------------------------


class TestExponentialDecay:
    def test_starts_at_init_value(self):
        s = exponential_decay(1.0, decay_rate=0.5, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_decays_geometrically(self):
        # decayed = init * decay_rate^(step / transition_steps)
        s = exponential_decay(1.0, decay_rate=0.5, transition_steps=100)
        assert s(100) == pytest.approx(0.5)
        assert s(200) == pytest.approx(0.25)
        assert s(300) == pytest.approx(0.125)

    def test_transition_begin_holds_init(self):
        s = exponential_decay(
            1.0,
            decay_rate=0.5,
            transition_begin=50,
            transition_steps=100,
        )
        assert s(0) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.5)

    def test_staircase(self):
        # With staircase=True, exponent is floored.
        s = exponential_decay(
            1.0,
            decay_rate=0.5,
            transition_steps=100,
            staircase=True,
        )
        # step=99: floor(0.99) = 0 -> decay_rate^0 = 1.
        assert s(99) == pytest.approx(1.0)
        # step=100: floor(1.0) = 1 -> 0.5.
        assert s(100) == pytest.approx(0.5)
        # step=199: floor(1.99) = 1 -> still 0.5.
        assert s(199) == pytest.approx(0.5)
        # step=200: floor(2.0) = 2 -> 0.25.
        assert s(200) == pytest.approx(0.25)

    def test_end_value_clamps(self):
        # decay_rate < 1 -> clamp at max(decayed, end_value).
        s = exponential_decay(
            1.0,
            decay_rate=0.5,
            transition_steps=100,
            end_value=0.1,
        )
        # Without clamp at step=400: 0.5^4 = 0.0625 < 0.1.
        assert s(400) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# cosine_schedule
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    def test_starts_at_init_value(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100)
        assert sched(0) == pytest.approx(1.0)

    def test_reaches_end_value_at_transition_steps(self):
        # num_cycles=0.5: half cosine, bottoms at progress=1.
        sched = cosine_schedule(1.0, 0.0, transition_steps=100, num_cycles=0.5)
        assert sched(100) == pytest.approx(0.0, abs=1e-12)

    def test_monotone_decreasing_for_default_cycles(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100)
        prev = sched(0)
        for s in range(1, 101):
            cur = sched(s)
            assert cur <= prev + 1e-12, f"non-monotone at step {s}"
            prev = cur

    def test_transition_begin_holds_init_before_offset(self):
        sched = cosine_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert sched(0) == pytest.approx(1.0)
        assert sched(49) == pytest.approx(1.0)
        assert sched(50) == pytest.approx(1.0)
        assert sched(150) == pytest.approx(0.0, abs=1e-12)

    def test_matches_hf_no_warmup(self):
        sched = cosine_schedule(BASE_LR, 0.0, transition_steps=1000)
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(
                o, num_warmup_steps=0, num_training_steps=1000
            )
        )
        _assert_pointwise(sched, hf, range(0, 1100, 7))

    def test_matches_hf_num_cycles_one(self):
        # Multi-cycle: progress past 1 keeps oscillating.
        sched = cosine_schedule(BASE_LR, 0.0, transition_steps=1000, num_cycles=1.0)
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(
                o, num_warmup_steps=0, num_training_steps=1000, num_cycles=1.0
            )
        )
        _assert_pointwise(sched, hf, range(0, 1500, 11))


# ---------------------------------------------------------------------------
# inverse_sqrt_schedule
# ---------------------------------------------------------------------------


class TestInverseSqrtSchedule:
    def test_starts_at_init_value(self):
        sched = inverse_sqrt_schedule(1.0, transition_steps=100)
        assert sched(0) == pytest.approx(1.0)

    def test_decays_as_one_over_sqrt(self):
        T = 100
        sched = inverse_sqrt_schedule(1.0, transition_steps=T)
        # At s=T, multiplier = 1/sqrt(2).
        assert sched(T) == pytest.approx(1.0 / math.sqrt(2))
        # At s=3T, multiplier = 1/sqrt(4) = 1/2.
        assert sched(3 * T) == pytest.approx(0.5)

    def test_transition_begin_holds_init(self):
        sched = inverse_sqrt_schedule(1.0, transition_steps=100, transition_begin=50)
        assert sched(0) == pytest.approx(1.0)
        assert sched(50) == pytest.approx(1.0)
        assert sched(150) == pytest.approx(1.0 / math.sqrt(2))


# ---------------------------------------------------------------------------
# one_minus_sqrt_schedule
# ---------------------------------------------------------------------------


class TestOneMinusSqrtSchedule:
    def test_starts_at_init(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        assert s(0) == pytest.approx(1.0)

    def test_reaches_end_at_transition_steps(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        assert s(100) == pytest.approx(0.0, abs=1e-12)

    def test_concave_decreasing(self):
        # Drops faster early than late.
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100)
        # At 25% progress: 1 - sqrt(0.25) = 0.5
        assert s(25) == pytest.approx(0.5)
        # At 50% progress: 1 - sqrt(0.5) ≈ 0.293
        assert s(50) == pytest.approx(1.0 - math.sqrt(0.5))

    def test_clamps_after_transition_steps(self):
        s = one_minus_sqrt_schedule(1.0, 0.2, transition_steps=100)
        assert s(100) == pytest.approx(0.2)
        assert s(500) == pytest.approx(0.2)

    def test_transition_begin_holds_init(self):
        s = one_minus_sqrt_schedule(1.0, 0.0, transition_steps=100, transition_begin=50)
        assert s(0) == pytest.approx(1.0)
        assert s(50) == pytest.approx(1.0)
        assert s(150) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# with_warmup
# ---------------------------------------------------------------------------


class TestWithWarmup:
    def test_zero_warmup_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_warmup(constant_schedule(0.5), transition_steps=0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_warmup(constant_schedule(0.5), transition_steps=-1)

    def test_warmup_ramp_starts_at_zero(self):
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        assert sched(0) == pytest.approx(0.0)

    def test_warmup_reaches_inner_value_at_transition_steps(self):
        # At step==transition_steps, ramp = 1, so result = inner(step).
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        assert sched(10) == pytest.approx(1.0)

    def test_multiplicative_composition(self):
        # Inner returns float(step); ramp(step) = step/10 for step < 10.
        # So sched(step) = (step/10) * step for step < 10, step otherwise.
        def inner(s):
            return float(s)

        sched = with_warmup(inner, transition_steps=10)
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.5 * 5)  # 0.5 * 5.0 = 2.5
        assert sched(10) == pytest.approx(10.0)  # ramp == 1, inner(10) = 10.0
        assert sched(100) == pytest.approx(100.0)

    def test_warmup_ramp_linear_with_constant_inner(self):
        sched = with_warmup(constant_schedule(1.0), transition_steps=10)
        for k in range(10):
            assert sched(k) == pytest.approx(k / 10)

    def test_accepts_float_as_constant(self):
        # `with_warmup(0.5, ...)` is shorthand for `with_warmup(constant_schedule(0.5), ...)`.
        sched = with_warmup(0.5, transition_steps=10)
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.25)  # 0.5 ramp * 0.5 const
        assert sched(10) == pytest.approx(0.5)
        assert sched(100) == pytest.approx(0.5)

    def test_cosine_ramp(self):
        # ramp(p) = 0.5 * (1 - cos(pi*p)). At p=0: 0; p=0.5: 0.5; p=1: 1.
        sched = with_warmup(constant_schedule(1.0), transition_steps=10, ramp="cosine")
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.5, abs=1e-12)
        assert sched(10) == pytest.approx(1.0)

    def test_one_minus_sqrt_ramp(self):
        # ramp(p) = 1 - sqrt(1 - p). At p=0: 0; p=1: 1.
        sched = with_warmup(constant_schedule(1.0), transition_steps=100, ramp="1-sqrt")
        assert sched(0) == pytest.approx(0.0)
        assert sched(100) == pytest.approx(1.0)
        # Concave (drops slowly early, fast late).
        # At p=0.5: 1 - sqrt(0.5) ~= 0.293
        assert sched(50) == pytest.approx(1.0 - math.sqrt(0.5))

    def test_callable_ramp(self):
        # Quadratic ramp: progress**2.
        sched = with_warmup(
            constant_schedule(1.0), transition_steps=10, ramp=lambda p: p * p
        )
        assert sched(0) == pytest.approx(0.0)
        assert sched(5) == pytest.approx(0.25)  # (5/10)**2 = 0.25
        assert sched(10) == pytest.approx(1.0)

    def test_unknown_ramp_string_raises(self):
        with pytest.raises(ValueError, match="Unknown ramp"):
            with_warmup(constant_schedule(1.0), transition_steps=10, ramp="bogus")


class TestWithRestarts:
    def test_zero_cycles_raises(self):
        with pytest.raises(ValueError, match="num_cycles > 0"):
            with_restarts(constant_schedule(1.0), transition_steps=10, num_cycles=0)

    def test_zero_transition_raises(self):
        with pytest.raises(ValueError, match="transition_steps > 0"):
            with_restarts(constant_schedule(1.0), transition_steps=0, num_cycles=2)

    def test_single_cycle_is_identity_within_window(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=100)
        sched = with_restarts(decay, transition_steps=100, num_cycles=1)
        for s in (0, 25, 50, 75, 100):
            assert sched(s) == pytest.approx(decay(s), abs=1e-12)

    def test_two_cycles_replays_at_half(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(decay, transition_steps=100, num_cycles=2)
        # Cycle length = 50; second cycle starts at step 50 from peak.
        assert sched(0) == pytest.approx(1.0)  # first cycle peak
        assert sched(50) == pytest.approx(1.0)  # second cycle peak (restart)
        assert sched(25) == pytest.approx(decay(25))
        assert sched(75) == pytest.approx(decay(25))  # same point in cycle 2

    def test_transition_begin_holds_at_init(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(
            decay, transition_steps=100, num_cycles=2, transition_begin=10
        )
        assert sched(0) == pytest.approx(1.0)  # before begin -> schedule(0)
        assert sched(9) == pytest.approx(1.0)
        assert sched(10) == pytest.approx(1.0)  # cycle starts here

    def test_after_window_returns_cycle_end(self):
        decay = cosine_schedule(1.0, 0.0, transition_steps=50)
        sched = with_restarts(decay, transition_steps=100, num_cycles=2)
        assert sched(100) == pytest.approx(
            0.0
        )  # cycle_length = 50; schedule(50) = end_value
        assert sched(500) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Composed: pointwise HF parity for each supported scheduler type.
# ---------------------------------------------------------------------------


class TestHFParity:
    """Cross-check each composed schedule pointwise vs HF for >1000 steps."""

    @pytest.fixture(params=[(0, 1000), (50, 1000), (100, 1000), (500, 2000)])
    def warmup_total(self, request):
        return request.param

    def _check(self, ours, hf, num_total, *, beyond=200):
        steps = range(0, num_total + beyond, max(1, (num_total + beyond) // 200))
        _assert_pointwise(ours, hf, steps)

    def test_linear(self, warmup_total):
        W, N = warmup_total
        decay = linear_schedule(BASE_LR, 0.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_cosine(self, warmup_total):
        W, N = warmup_total
        decay = cosine_schedule(BASE_LR, 0.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_constant_with_warmup(self, warmup_total):
        W, _ = warmup_total
        if W == 0:
            pytest.skip("constant_with_warmup is just constant; tested elsewhere")
        ours = with_warmup(BASE_LR, transition_steps=W)
        hf = _hf_lambda(lambda o: get_constant_schedule_with_warmup(o, W))
        self._check(ours, hf, max(W * 3, 100))

    def test_polynomial(self, warmup_total):
        W, N = warmup_total
        # HF defaults: lr_end=1e-7, power=1.0.
        decay = polynomial_schedule(BASE_LR, 1e-7, 1.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(lambda o: get_polynomial_decay_schedule_with_warmup(o, W, N))
        self._check(ours, hf, N)

    def test_polynomial_custom_power(self):
        W, N = 100, 1000
        decay = polynomial_schedule(BASE_LR, 1e-7, 2.0, N - W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W)
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(o, W, N, power=2.0)
        )
        self._check(ours, hf, N)

    def test_inverse_sqrt(self, warmup_total):
        W, _ = warmup_total
        # HF default: timescale = num_warmup_steps (or 10000 if W==0; but HF
        # raises ValueError for W==0 since timescale defaults None -> int(None)).
        if W == 0:
            pytest.skip("HF inverse_sqrt requires num_warmup_steps > 0 by default")
        decay = inverse_sqrt_schedule(BASE_LR, transition_steps=W, transition_begin=W)
        ours = with_warmup(decay, transition_steps=W)
        hf = _hf_lambda(lambda o: get_inverse_sqrt_schedule(o, num_warmup_steps=W))
        self._check(ours, hf, W * 5)

    @pytest.mark.parametrize("num_cycles", [1, 2, 4])
    def test_cosine_with_restarts(self, warmup_total, num_cycles):
        W, N = warmup_total
        if (N - W) % num_cycles != 0:
            pytest.skip(
                f"with_restarts requires num_cycles | (N-W); "
                f"got N-W={N - W}, num_cycles={num_cycles}"
            )
        cycle_length = (N - W) // num_cycles
        inner = cosine_schedule(BASE_LR, 0.0, transition_steps=cycle_length)
        decay = with_restarts(
            inner,
            transition_steps=N - W,
            num_cycles=num_cycles,
            transition_begin=W,
        )
        ours = with_warmup(decay, transition_steps=W) if W > 0 else decay
        hf = _hf_lambda(
            lambda o: get_cosine_with_hard_restarts_schedule_with_warmup(
                o,
                W,
                N,
                num_cycles=num_cycles,
            )
        )
        self._check(ours, hf, N)
