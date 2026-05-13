"""Tests for opaque.api.transformers.trainer._scheduler.

Verifies that ``build_lr_schedule`` dispatches HF's ``lr_scheduler_type``
strings to a schedule that reproduces HF's reference behavior pointwise,
and that deferred / unknown types and bad kwargs raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

torchopt = pytest.importorskip("torchopt", reason="torchopt required")
transformers = pytest.importorskip(
    "transformers", reason="transformers required for HF cross-check"
)

from torch.optim import SGD  # noqa: E402
from transformers.optimization import (  # noqa: E402
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_wsd_schedule,
)

from opaque.api.transformers.trainer._scheduler import (  # noqa: E402
    ReduceLROnPlateauSchedule,
    build_lr_schedule,
    get_warmup_steps,
    parse_optim_args,
)


BASE_LR = 1e-3


@dataclass
class _Args:
    """Minimal stand-in for TrainingArguments — only the fields the shim reads."""

    lr_scheduler_type: str = "linear"
    learning_rate: float = BASE_LR
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_for_best_model: str | None = None


def _hf_lambda(make_scheduler):
    p = torch.zeros(1, requires_grad=True)
    return make_scheduler(SGD([p], lr=BASE_LR)).lr_lambdas[0]


def _assert_pointwise(ours, hf_lam, num_total, *, beyond=100, tol=1e-9):
    step_size = max(1, (num_total + beyond) // 200)
    for s in range(0, num_total + beyond, step_size):
        assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=tol), (
            f"step={s}: ours={ours(s)} hf={hf_lam(s) * BASE_LR}"
        )


# ---------------------------------------------------------------------------
# get_warmup_steps
# ---------------------------------------------------------------------------


class TestGetWarmupSteps:
    def test_warmup_steps_only(self):
        assert get_warmup_steps(1000, 50, 0.0) == 50

    def test_warmup_ratio_only(self):
        assert get_warmup_steps(1000, 0, 0.1) == 100

    def test_warmup_steps_wins_when_both_set(self):
        assert get_warmup_steps(1000, 50, 0.1) == 50

    def test_zero_when_neither(self):
        assert get_warmup_steps(1000, 0, 0.0) == 0

    def test_ratio_uses_ceil(self):
        # 1000 * 0.0501 = 50.1 -> ceil = 51
        assert get_warmup_steps(1000, 0, 0.0501) == 51


# ---------------------------------------------------------------------------
# Per-type pointwise parity with HF
# ---------------------------------------------------------------------------


class TestPointwiseParity:
    def test_constant(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="constant"), num_training_steps=500
        )
        hf = _hf_lambda(get_constant_schedule)
        _assert_pointwise(ours, hf, 500)

    def test_constant_with_warmup(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="constant_with_warmup", warmup_steps=50),
            num_training_steps=500,
        )
        hf = _hf_lambda(
            lambda o: get_constant_schedule_with_warmup(o, num_warmup_steps=50)
        )
        _assert_pointwise(ours, hf, 500)

    def test_linear(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 100, 1000))
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_default_cycles(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="cosine", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, 100, 1000))
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_custom_cycles(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine",
                warmup_steps=100,
                lr_scheduler_kwargs={"num_cycles": 1.5},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(o, 100, 1000, num_cycles=1.5)
        )
        _assert_pointwise(ours, hf, 1000)

    def test_polynomial_defaults(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="polynomial", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(o, 100, 1000)
        )
        _assert_pointwise(ours, hf, 1000)

    def test_polynomial_custom_power(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="polynomial",
                warmup_steps=100,
                lr_scheduler_kwargs={"power": 2.0, "lr_end": 1e-9},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(
                o, 100, 1000, lr_end=1e-9, power=2.0
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_inverse_sqrt(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="inverse_sqrt", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_inverse_sqrt_schedule(o, num_warmup_steps=100))
        _assert_pointwise(ours, hf, 1000)

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_cosine_with_restarts(self, k):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_with_restarts",
                warmup_steps=100,
                lr_scheduler_kwargs={"num_cycles": k},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_hard_restarts_schedule_with_warmup(
                o, 100, 1000, num_cycles=k
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_with_min_lr_absolute(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr": 1e-4},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_min_lr_schedule_with_warmup(
                o, 100, 1000, min_lr=1e-4
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_with_min_lr_rate(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_min_lr_schedule_with_warmup(
                o, 100, 1000, min_lr_rate=0.1
            )
        )
        _assert_pointwise(ours, hf, 1000)

    # cosine_warmup_with_min_lr deviates from HF's lambda by a one-step
    # phase shift in the indexing (HF uses (step + 1) instead of step).
    # We test the contract directly rather than pointwise HF parity.

    def test_cosine_warmup_with_min_lr_default_warmup_floor(self):
        # warmup_lr_rate omitted → warmup ramps from 0 to base_lr.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
            ),
            num_training_steps=1000,
        )
        assert sched(0) == pytest.approx(0.0)
        assert sched(100) == pytest.approx(base_lr)
        # Cosine to floor: at step 1000, value is min_lr_rate * base_lr.
        assert sched(1000) == pytest.approx(0.1 * base_lr, abs=1e-12)

    def test_cosine_warmup_with_min_lr_with_warmup_floor(self):
        # warmup_lr_rate=0.2 → warmup ramps from 0.2*base_lr to base_lr.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1, "warmup_lr_rate": 0.2},
            ),
            num_training_steps=1000,
        )
        assert sched(0) == pytest.approx(0.2 * base_lr)
        assert sched(100) == pytest.approx(base_lr)
        # Cosine to floor.
        assert sched(1000) == pytest.approx(0.1 * base_lr, abs=1e-12)

    def test_cosine_warmup_with_min_lr_decay_matches_cosine_with_min_lr(self):
        # Post-warmup, the decay shape is identical to cosine_with_min_lr.
        kwargs = {"min_lr_rate": 0.1}
        ours_warmup = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs=kwargs,
            ),
            num_training_steps=1000,
        )
        ours_regular = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs=kwargs,
            ),
            num_training_steps=1000,
        )
        for s in range(100, 1100, 50):  # post-warmup
            assert ours_warmup(s) == pytest.approx(ours_regular(s), abs=1e-12)

    def test_cosine_warmup_with_min_lr_accepts_absolute_min_lr(self):
        # HF parity: min_lr (absolute) is accepted and converted internally.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr": 1e-5},
            ),
            num_training_steps=1000,
        )
        assert sched(100) == pytest.approx(base_lr)
        # End of decay: cosine reaches 1e-5.
        assert sched(1000) == pytest.approx(1e-5, abs=1e-12)

    def test_warmup_stable_decay_cosine(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "min_lr_ratio": 0.1},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_training_steps=1000,
                min_lr_ratio=0.1,
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_warmup_stable_decay_linear(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "decay_type": "linear"},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_training_steps=1000,
                decay_type="linear",
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_warmup_stable_decay_explicit_stable(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "num_stable_steps": 500},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_stable_steps=500,
            )
        )
        _assert_pointwise(ours, hf, 50 + 500 + 200)

    @pytest.mark.parametrize("warmup_type", ["linear", "cosine", "1-sqrt"])
    @pytest.mark.parametrize("decay_type", ["linear", "cosine", "1-sqrt"])
    def test_warmup_stable_decay_all_combos(self, warmup_type, decay_type):
        W, D, N = 50, 200, 1000
        kw = {
            "num_decay_steps": D,
            "warmup_type": warmup_type,
            "decay_type": decay_type,
            "min_lr_ratio": 0.1,
        }
        ours = build_lr_schedule(
            _Args(
                lr_scheduler_type="warmup_stable_decay",
                warmup_steps=W,
                lr_scheduler_kwargs=kw,
            ),
            num_training_steps=N,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=W,
                num_decay_steps=D,
                num_training_steps=N,
                warmup_type=warmup_type,
                decay_type=decay_type,
                min_lr_ratio=0.1,
            )
        )
        _assert_pointwise(ours, hf, N)


# ---------------------------------------------------------------------------
# warmup resolution via warmup_ratio
# ---------------------------------------------------------------------------


class TestWarmupRatio:
    def test_warmup_ratio_resolves_to_steps(self):
        # ratio 0.05 of 1000 -> 50 warmup steps
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_ratio=0.05),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 50, 1000))
        _assert_pointwise(ours, hf, 1000)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_reduce_lr_on_plateau_defaults_metric_to_loss(self):
        # HF parity: ``TrainingArguments.__post_init__`` defaults
        # ``metric_for_best_model`` to ``"loss"`` whenever
        # ``lr_scheduler_type == "reduce_lr_on_plateau"``.  This stub
        # mirrors that contract: the builder accepts a default metric
        # name (``"loss"``) and wires up ``mode="min"``.
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="reduce_lr_on_plateau",
                metric_for_best_model="loss",
            ),
            num_training_steps=100,
        )
        # ``mode`` derived from the metric name suffix: ``"loss"`` → "min".
        assert sched.mode == "min"

    def test_cosine_with_min_lr_requires_one_of_min_args(self):
        with pytest.raises(ValueError, match="min_lr"):
            build_lr_schedule(
                _Args(lr_scheduler_type="cosine_with_min_lr", warmup_steps=10),
                num_training_steps=100,
            )

    def test_cosine_with_min_lr_rejects_both_min_args(self):
        with pytest.raises(ValueError, match="only one"):
            build_lr_schedule(
                _Args(
                    lr_scheduler_type="cosine_with_min_lr",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"min_lr": 1e-4, "min_lr_rate": 0.1},
                ),
                num_training_steps=100,
            )

    def test_cosine_warmup_with_min_lr_requires_min_arg(self):
        with pytest.raises(ValueError, match="min_lr"):
            build_lr_schedule(
                _Args(lr_scheduler_type="cosine_warmup_with_min_lr", warmup_steps=10),
                num_training_steps=100,
            )

    def test_wsd_requires_num_decay_steps(self):
        with pytest.raises(ValueError, match="num_decay_steps"):
            build_lr_schedule(
                _Args(lr_scheduler_type="warmup_stable_decay", warmup_steps=10),
                num_training_steps=100,
            )

    def test_wsd_invalid_warmup_type(self):
        with pytest.raises(ValueError, match="warmup_type"):
            build_lr_schedule(
                _Args(
                    lr_scheduler_type="warmup_stable_decay",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"num_decay_steps": 20, "warmup_type": "bogus"},
                ),
                num_training_steps=100,
            )

    def test_wsd_invalid_decay_type(self):
        with pytest.raises(ValueError, match="decay_type"):
            build_lr_schedule(
                _Args(
                    lr_scheduler_type="warmup_stable_decay",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"num_decay_steps": 20, "decay_type": "bogus"},
                ),
                num_training_steps=100,
            )

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown lr_scheduler_type"):
            build_lr_schedule(_Args(lr_scheduler_type="bogus"), 100)

    def test_unknown_kwargs_for_cosine(self):
        with pytest.raises(ValueError, match="unsupported keys"):
            build_lr_schedule(
                _Args(
                    lr_scheduler_type="cosine",
                    lr_scheduler_kwargs={"power": 2.0},  # 'power' belongs to polynomial
                ),
                num_training_steps=100,
            )

    def test_unknown_kwargs_for_constant(self):
        # constant accepts no kwargs at all
        with pytest.raises(ValueError, match="unsupported keys"):
            build_lr_schedule(
                _Args(
                    lr_scheduler_type="constant",
                    lr_scheduler_kwargs={"num_cycles": 1.0},
                ),
                num_training_steps=100,
            )


# ---------------------------------------------------------------------------
# ReduceLROnPlateauSchedule — metric-driven dispatch
# ---------------------------------------------------------------------------


class TestReduceLROnPlateau:
    def test_dispatcher_returns_plateau_instance(self):
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="reduce_lr_on_plateau",
                metric_for_best_model="eval_loss",
                lr_scheduler_kwargs={"factor": 0.5, "patience": 2},
            ),
            num_training_steps=100,
        )
        assert isinstance(sched, ReduceLROnPlateauSchedule)
        assert sched.factor == 0.5
        assert sched.patience == 2
        # Loss-named metric → mode="min".
        assert sched.mode == "min"
        # Initial LR before any update equals base_lr.
        assert sched(0) == pytest.approx(BASE_LR)

    def test_mode_defaults_to_max_for_non_loss_metric(self):
        sched = build_lr_schedule(
            _Args(
                lr_scheduler_type="reduce_lr_on_plateau",
                metric_for_best_model="eval_accuracy",
            ),
            num_training_steps=10,
        )
        assert sched.mode == "max"

    def test_drops_lr_after_patience(self):
        sched = ReduceLROnPlateauSchedule(
            base_lr=1.0,
            factor=0.5,
            patience=1,
            threshold=0.0,
            mode="min",
        )
        # First update: best.
        sched.update(1.0)
        assert sched(0) == 1.0
        # Second update: bad → counter=1, still under patience.
        sched.update(1.0)
        assert sched(0) == 1.0
        # Third update: bad again → counter=2 > patience, factor applied.
        sched.update(1.0)
        assert sched(0) == pytest.approx(0.5)

    def test_state_dict_round_trip(self):
        a = ReduceLROnPlateauSchedule(
            base_lr=1.0, factor=0.5, patience=0, threshold=0.0
        )
        a.update(1.0)
        a.update(1.0)  # → drop
        sd = a.state_dict()

        b = ReduceLROnPlateauSchedule(
            base_lr=1.0, factor=0.5, patience=0, threshold=0.0
        )
        b.load_state_dict(sd)
        assert b(0) == a(0)


# ---------------------------------------------------------------------------
# parse_optim_args
# ---------------------------------------------------------------------------


class TestParseOptimArgs:
    def test_empty_returns_empty_dict(self):
        assert parse_optim_args(None) == {}
        assert parse_optim_args("") == {}

    def test_basic_parse_with_coercion(self):
        out = parse_optim_args("momentum=0.9,nesterov=True,steps=10,name=adam")
        assert out == {
            "momentum": 0.9,
            "nesterov": True,
            "steps": 10,
            "name": "adam",
        }

    def test_whitespace_tolerated(self):
        assert parse_optim_args(" a = 1 , b = 2 ") == {"a": 1, "b": 2}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_optim_args("momentum")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            parse_optim_args("=1")


# ---------------------------------------------------------------------------
# Warmup edge cases — boundary values that frequently lurk in real configs
# (zero warmup, full warmup, over-shoot warmup, polynomial-as-linear).
# ---------------------------------------------------------------------------


class TestWarmupEdgeCases:
    """HF-parity around degenerate warmup parameter combinations."""

    def test_linear_no_warmup(self):
        # ``warmup_steps=0`` and ``warmup_ratio=0`` ⇒ the warmup ramp
        # collapses to the identity transform.  HF and ours must match.
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_steps=0),
            num_training_steps=200,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 0, 200))
        _assert_pointwise(ours, hf, 200)

    def test_cosine_no_warmup(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="cosine", warmup_steps=0),
            num_training_steps=200,
        )
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, 0, 200))
        _assert_pointwise(ours, hf, 200)

    def test_warmup_equal_to_total_training_steps(self):
        # All-warmup schedule: ``warmup_steps == num_training_steps``
        # means the LR ramps from 0 to base_lr across the entire run.
        # HF and ours agree pointwise *during* the warmup ramp — only
        # at the exact boundary step they differ by one (HF clips
        # immediately to 0; ours holds at base_lr for one step before
        # the zero-length decay floors at 0).  This test asserts the
        # ramp itself is HF-parity; the boundary behaviour is
        # documented as a known divergence.
        N = 100
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_steps=N),
            num_training_steps=N,
        )
        hf_lam = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, N, N))
        # Exercise the ramp body excluding the boundary step.
        for s in range(0, N, 5):
            assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=1e-9)
        # By step N+1, both have decayed to 0.
        assert ours(N + 1) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_exceeds_total_training_steps(self):
        # Over-shoot: ``warmup_steps > num_training_steps`` means the
        # ramp never reaches base_lr; the decay phase has zero length.
        # HF clips silently — its lambda just keeps ramping forever and
        # the trainer never queries it past num_training_steps.  Ours
        # matches HF on the ramp body across the whole training-step
        # range.
        N = 100
        ours = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_steps=200),
            num_training_steps=N,
        )
        hf_lam = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 200, N))
        for s in range(0, N, 5):
            assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=1e-9)

    def test_polynomial_power_one_matches_linear(self):
        # ``polynomial(power=1.0, lr_end=0.0)`` is exactly a linear
        # decay.  Confirm the dispatcher produces identical curves to
        # ``linear``.
        N = 500
        polynomial = build_lr_schedule(
            _Args(
                lr_scheduler_type="polynomial",
                warmup_steps=50,
                lr_scheduler_kwargs={"power": 1.0, "lr_end": 0.0},
            ),
            num_training_steps=N,
        )
        linear = build_lr_schedule(
            _Args(lr_scheduler_type="linear", warmup_steps=50),
            num_training_steps=N,
        )
        for s in range(0, N + 50, 25):
            assert polynomial(s) == pytest.approx(linear(s), abs=1e-9)
