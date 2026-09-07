"""Unit tests for ``trainer/_checkpoint.py`` helpers."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch

import opaque.api.transformers.trainer._checkpoint as ckpt
from opaque.api.engine.clipping.types import FixedClipState
from opaque.api.transformers.trainer._participation import (
    ResolvedParticipationPlan,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpsgd.noise import gaussian_noise
from opaque.exceptions import CheckpointError
from opaque.random import key
from opaque.random.types import RngKey
from opaque.serialization import from_state_dict as opaque_from_state_dict
from opaque.types import (
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)


class TestParseCheckpointStep:
    def test_valid(self):
        assert ckpt.parse_checkpoint_step("checkpoint-500") == 500
        assert ckpt.parse_checkpoint_step("/tmp/output/checkpoint-12") == 12

    def test_invalid(self):
        assert ckpt.parse_checkpoint_step("model") is None
        assert ckpt.parse_checkpoint_step("checkpoint-abc") is None
        assert ckpt.parse_checkpoint_step("checkpoint-500-extra") is None


class TestListAndLastCheckpoint:
    def test_empty_dir_returns_empty_and_none(self, tmp_path):
        assert ckpt.list_checkpoints(str(tmp_path)) == []
        assert ckpt.get_last_checkpoint(str(tmp_path)) is None

    def test_nonexistent_dir(self, tmp_path):
        missing = str(tmp_path / "nope")
        assert ckpt.list_checkpoints(missing) == []
        assert ckpt.get_last_checkpoint(missing) is None

    def test_sorted_ascending(self, tmp_path):
        for step in (100, 10, 50):
            (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / "model").mkdir()  # ignored
        listing = ckpt.list_checkpoints(str(tmp_path))
        assert [Path(p).name for p in listing] == [
            "checkpoint-10",
            "checkpoint-50",
            "checkpoint-100",
        ]
        assert Path(ckpt.get_last_checkpoint(str(tmp_path))).name == "checkpoint-100"

    def test_files_with_checkpoint_prefix_ignored(self, tmp_path):
        (tmp_path / "checkpoint-99").touch()  # file, not dir
        assert ckpt.list_checkpoints(str(tmp_path)) == []


class TestRotateCheckpoints:
    def _make(self, path, step):
        d = path / f"checkpoint-{step}"
        d.mkdir()
        (d / "marker").touch()

    def test_no_op_when_limit_none(self, tmp_path):
        for s in (1, 2, 3):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=None)
        assert len(ckpt.list_checkpoints(str(tmp_path))) == 3

    def test_no_op_when_under_limit(self, tmp_path):
        for s in (1, 2):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=5)
        assert len(ckpt.list_checkpoints(str(tmp_path))) == 2

    def test_keeps_n_most_recent(self, tmp_path):
        for s in (1, 2, 3, 4, 5):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=2)
        names = [Path(p).name for p in ckpt.list_checkpoints(str(tmp_path))]
        assert names == ["checkpoint-4", "checkpoint-5"]

    def test_protects_best_when_outside_window(self, tmp_path):
        for s in (1, 2, 3, 4, 5):
            self._make(tmp_path, s)
        best = str(tmp_path / "checkpoint-1")
        ckpt.rotate_checkpoints(
            str(tmp_path), save_total_limit=2, best_model_checkpoint=best
        )
        names = sorted(Path(p).name for p in ckpt.list_checkpoints(str(tmp_path)))
        # Both most-recent (5) and best (1) survive; total kept = max(2, 2) = 2
        assert "checkpoint-1" in names
        assert "checkpoint-5" in names

    def test_save_total_limit_one_with_best(self, tmp_path):
        # HF parity: limit=1 with a different best keeps both.
        for s in (1, 2, 3):
            self._make(tmp_path, s)
        best = str(tmp_path / "checkpoint-2")
        ckpt.rotate_checkpoints(
            str(tmp_path), save_total_limit=1, best_model_checkpoint=best
        )
        names = sorted(Path(p).name for p in ckpt.list_checkpoints(str(tmp_path)))
        assert names == ["checkpoint-2", "checkpoint-3"]


class TestRngSnapshot:
    def test_roundtrip_python(self):
        import random as r

        r.seed(123)
        snap = ckpt.snapshot_rng_state()
        a = r.random()

        r.seed(999)  # disturb
        ckpt.restore_rng_state(snap)
        b = r.random()
        assert a == b

    def test_roundtrip_numpy(self):
        import numpy as np

        np.random.seed(7)
        snap = ckpt.snapshot_rng_state()
        a = np.random.rand(5)

        np.random.seed(0)
        ckpt.restore_rng_state(snap)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_roundtrip_torch_cpu(self):
        torch.manual_seed(42)
        snap = ckpt.snapshot_rng_state()
        a = torch.rand(5)

        torch.manual_seed(0)
        ckpt.restore_rng_state(snap)
        b = torch.rand(5)
        assert torch.equal(a, b)


class TestDpRuntimeBundle:
    def test_roundtrip(self, tmp_path):
        clip = FixedClipState()
        _, noise = gaussian_noise(noise_multiplier=1.0, key=key(11))

        path = str(tmp_path / "dp_runtime.pt")
        participation = ResolvedParticipationPlan.resolve(
            mechanism_kind="gaussian",
            sampling_mode="k_out_of_t",
            sampling_kwargs={"k": 3, "allocation": "block"},
            population_size=100,
            expected_batch_size=10,
            sample_rate=0.1,
            total_steps=30,
            num_bins=10,
            world_size=1,
        )
        ckpt.save_dp_runtime_state(
            path,
            clip_state=clip,
            noise_state=noise,
            sampler_state={
                "key_seed": 5,
                "key_impl": "opaque_threefry_like",
                "consumed": 2,
                "num_samples": 100,
                "k": 3,
                "t": 30,
                "allocation": "block",
            },
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.1,
            expected_steps_per_epoch=10,
            expected_batch_size=10,
            total_steps=30,
            participation_plan=participation.to_state_dict(),
            is_horizon_process=True,
            calibration_source="calibrated",
            target_epsilon=5.0,
            horizon_process_state={"type": "ExampleHorizon", "n_steps": 30},
        )
        loaded = ckpt.load_dp_runtime_state(path)

        assert isinstance(loaded, ckpt.RuntimeCheckpoint)
        assert opaque_from_state_dict(clip, loaded.clip_state) == clip
        assert opaque_from_state_dict(noise, loaded.noise_state) == noise
        assert loaded.version == ckpt.DP_STATE_BUNDLE_VERSION
        assert loaded.sampler_state["consumed"] == 2
        assert loaded.sample_rate == pytest.approx(0.1)
        assert loaded.target_delta == pytest.approx(1e-5)
        assert loaded.noise_multiplier == pytest.approx(1.1)
        assert loaded.expected_steps_per_epoch == 10
        assert loaded.total_steps == 30
        assert loaded.participation_plan == participation.to_state_dict()
        assert loaded.is_horizon_process is True
        assert loaded.calibration_source == "calibrated"
        assert loaded.target_epsilon == pytest.approx(5.0)
        assert loaded.horizon_process_state == {
            "type": "ExampleHorizon",
            "n_steps": 30,
        }

    @staticmethod
    def _legacy_band_runtime(sampler_state):
        runtime = ckpt.RuntimeCheckpoint(
            version=ckpt.DP_STATE_BUNDLE_VERSION,
            clip_state={},
            noise_state={},
            sampler_state=sampler_state,
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=10,
            expected_batch_size=10,
            total_steps=20,
            mechanism_kind="mf_band",
        )
        runtime.__dict__.pop("participation_plan")
        runtime.__dict__.pop("sampler_cursor_origin")
        return runtime

    @classmethod
    def _coherent_legacy_b_min_sep_runtime(cls, sampler_state):
        runtime = cls._legacy_band_runtime(sampler_state)
        runtime.is_horizon_process = True
        runtime.mf_n_steps = 20
        runtime.mf_min_sep = 4
        runtime.mf_max_participations = 5
        return runtime

    @staticmethod
    def _issue_776_fixture():
        path = (
            Path(__file__).parents[1]
            / "fixtures"
            / "issue_776_legacy_sampler_states_v7.json"
        )
        with path.open() as fixture_file:
            return json.load(fixture_file)

    def test_legacy_band_poisson_is_inspectable_but_not_resumable(self, tmp_path):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        assert set(runtime.__dict__) == set(
            fixture["_fixture"]["runtime_instance_fields"]
        )
        path = tmp_path / "legacy.pt"
        torch.save(runtime, path)

        loaded = ckpt.load_dp_runtime_state(str(path))
        assert getattr(loaded, "participation_plan", None) is None
        with pytest.raises(
            CheckpointError,
            match=r"Changing the sampler cannot repair.*mf_identity",
        ):
            ckpt.validate_dp_runtime_for_resume(loaded)

    def test_legacy_band_b_min_sep_remains_resumable(self):
        fixture = self._issue_776_fixture()
        runtime = self._coherent_legacy_b_min_sep_runtime(fixture["b_min_sep"])
        ckpt.validate_dp_runtime_for_resume(runtime)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("bands", 5, "contradicts its saved MF"),
            ("sampling_prob", 0.2, "probability contradicts"),
            ("n_steps", 19, "contradicts its saved MF"),
            ("consumed", 21, "cursor is invalid"),
        ],
    )
    def test_legacy_band_b_min_sep_rejects_contradictions(self, field, value, message):
        fixture = self._issue_776_fixture()
        sampler_state = dict(fixture["b_min_sep"])
        sampler_state[field] = value
        runtime = self._coherent_legacy_b_min_sep_runtime(sampler_state)
        with pytest.raises(CheckpointError, match=message):
            ckpt.validate_dp_runtime_for_resume(runtime)

    def test_sampler_cursor_must_match_trainer_progress(self):
        fixture = self._issue_776_fixture()
        runtime = self._coherent_legacy_b_min_sep_runtime(fixture["b_min_sep"])
        runtime.sampler_state = dict(runtime.sampler_state, consumed=4)

        ckpt.validate_sampler_cursor_for_resume(
            runtime,
            global_step=4,
            ignore_data_skip=False,
        )
        with pytest.raises(CheckpointError, match="does not match trainer progress"):
            ckpt.validate_sampler_cursor_for_resume(
                runtime,
                global_step=3,
                ignore_data_skip=False,
            )

    def test_sampler_cursor_origin_supports_chained_poisson_resume(self):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        runtime.mechanism_kind = "gaussian"
        runtime.sampler_state = dict(runtime.sampler_state, consumed=2)
        runtime.sampler_cursor_origin = 5

        ckpt.validate_sampler_cursor_for_resume(
            runtime,
            global_step=7,
            ignore_data_skip=False,
        )
        with pytest.raises(CheckpointError, match="does not match trainer progress"):
            ckpt.validate_sampler_cursor_for_resume(
                runtime,
                global_step=8,
                ignore_data_skip=False,
            )

    def test_non_poisson_sampler_rejects_nonzero_cursor_origin(self):
        fixture = self._issue_776_fixture()
        runtime = self._coherent_legacy_b_min_sep_runtime(fixture["b_min_sep"])
        runtime.sampler_cursor_origin = 1

        with pytest.raises(CheckpointError, match="whole-dataset Poisson"):
            ckpt.validate_sampler_cursor_for_resume(
                runtime,
                global_step=1,
                ignore_data_skip=False,
            )

    @pytest.mark.parametrize("global_step", [True, 1.5])
    def test_sampler_cursor_rejects_non_integer_trainer_progress(self, global_step):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        with pytest.raises(CheckpointError, match="global_step is invalid"):
            ckpt.validate_sampler_cursor_for_resume(
                runtime,
                global_step=global_step,
                ignore_data_skip=True,
            )

    def test_sampler_cursor_rejects_non_boolean_skip_policy(self):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        with pytest.raises(CheckpointError, match="must be a bool"):
            ckpt.validate_sampler_cursor_for_resume(
                runtime,
                global_step=0,
                ignore_data_skip=1,
            )

    def test_ignore_data_skip_explicitly_discards_saved_cursor(self):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        ckpt.validate_sampler_cursor_for_resume(
            runtime,
            global_step=7,
            ignore_data_skip=True,
        )

    def test_legacy_band_ambiguous_sampler_state_fails_closed(self):
        fixture = self._issue_776_fixture()
        ambiguous = dict(fixture["plain_poisson"])
        ambiguous["bands"] = 4
        runtime = self._legacy_band_runtime(ambiguous)
        with pytest.raises(CheckpointError, match="refusing to guess"):
            ckpt.validate_dp_runtime_for_resume(runtime)

    def test_current_plan_rejects_contradictory_sampler_state(self):
        fixture = self._issue_776_fixture()
        plan = ResolvedParticipationPlan.resolve(
            mechanism_kind="mf_band",
            sampling_mode="b_min_sep",
            sampling_kwargs={},
            population_size=100,
            expected_batch_size=10,
            sample_rate=0.1,
            total_steps=20,
            num_bins=10,
            world_size=1,
        )
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        runtime.participation_plan = plan.to_state_dict()
        with pytest.raises(CheckpointError, match="Changing the sampler cannot repair"):
            ckpt.validate_dp_runtime_for_resume(runtime)

    def test_current_plan_accepts_rank_local_sampler_population(self):
        fixture = self._issue_776_fixture()
        sampler_state = dict(fixture["plain_poisson"])
        sampler_state["num_samples"] = 50
        plan = ResolvedParticipationPlan.resolve(
            mechanism_kind="gaussian",
            sampling_mode="poisson",
            sampling_kwargs={},
            population_size=100,
            expected_batch_size=10,
            sample_rate=0.1,
            total_steps=20,
            num_bins=10,
            world_size=2,
        )
        runtime = self._legacy_band_runtime(sampler_state)
        runtime.mechanism_kind = "gaussian"
        runtime.participation_plan = plan.to_state_dict()

        ckpt.validate_dp_runtime_for_resume(runtime)

    def test_current_plan_rejects_malformed_sampler_scalar_type(self):
        fixture = self._issue_776_fixture()
        sampler_state = dict(fixture["plain_poisson"])
        sampler_state["n_steps"] = True
        plan = ResolvedParticipationPlan.resolve(
            mechanism_kind="gaussian",
            sampling_mode="poisson",
            sampling_kwargs={},
            population_size=100,
            expected_batch_size=10,
            sample_rate=0.1,
            total_steps=20,
            num_bins=10,
            world_size=1,
        )
        runtime = self._legacy_band_runtime(sampler_state)
        runtime.mechanism_kind = "gaussian"
        runtime.participation_plan = plan.to_state_dict()

        with pytest.raises(CheckpointError, match="horizon has an invalid type"):
            ckpt.validate_dp_runtime_for_resume(runtime)

    def test_legacy_non_band_poisson_is_not_rejected(self):
        fixture = self._issue_776_fixture()
        runtime = self._legacy_band_runtime(fixture["plain_poisson"])
        runtime.mechanism_kind = "gaussian"
        ckpt.validate_dp_runtime_for_resume(runtime)

    def test_unsupported_clip_state_type_raises(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        _, noise = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(CheckpointError, match="clip_state must be a ClipState"):
            ckpt.save_dp_runtime_state(
                path,
                clip_state="not_a_clip_state",
                noise_state=noise,
                sampler_state=None,
                sample_rate=0.1,
                target_delta=1e-5,
                noise_multiplier=1.0,
                expected_steps_per_epoch=1,
                expected_batch_size=32,
                total_steps=1,
            )

    def test_unsupported_noise_state_type_raises(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        clip = FixedClipState()
        with pytest.raises(CheckpointError, match="noise_state must be a NoiseState"):
            ckpt.save_dp_runtime_state(
                path,
                clip_state=clip,
                noise_state="not_noise",
                sampler_state=None,
                sample_rate=0.1,
                target_delta=1e-5,
                noise_multiplier=1.0,
                expected_steps_per_epoch=1,
                expected_batch_size=32,
                total_steps=1,
            )

    def test_rejects_unknown_bundle_version(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        fake = ckpt.RuntimeCheckpoint(
            version=1,  # wrong; current is DP_STATE_BUNDLE_VERSION (see _checkpoint)
            clip_state={},
            noise_state={},
            sampler_state=None,
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=1,
            expected_batch_size=32,
            total_steps=1,
        )
        torch.save(fake, path)
        with pytest.raises(CheckpointError, match="unsupported dp_state"):
            ckpt.load_dp_runtime_state(path)


_NOISE_STEPS = 6
_NOISE_CHECKPOINT_STEP = 3


def _noise_tree() -> dict[str, Any]:
    return {
        "fallback": torch.zeros(3, dtype=torch.float64),
        "nested": {"head": torch.zeros((2, 2), dtype=torch.float64)},
    }


def _per_group_norm(*, fallback: float, head: float) -> PerGroup:
    return PerGroup(
        groups={
            ("fallback",): "fallback",
            ("nested", "head"): "head",
        },
        values={"fallback": fallback, "head": head},
    )


def _mf_noise_factory(strategy_factory: Callable[[], Any]) -> Callable[[RngKey], Any]:
    def make_noise(rng_key: RngKey):
        return mf_gaussian_noise(
            _noise_tree(),
            strategy_factory(),
            n_steps=_NOISE_STEPS,
            min_sep=_NOISE_STEPS,
            max_participations=1,
            noise_multiplier=1.0,
            key=rng_key,
            compute_dtype=torch.float64,
        )

    return make_noise


def _paired_noise_factory(rng_key: RngKey):
    return mf_gaussian_noise(
        _noise_tree(),
        band_mf_strategy(bands=3, momentum=0.9),
        n_steps=_NOISE_STEPS,
        min_sep=_NOISE_STEPS,
        max_participations=1,
        noise_multiplier=1.0,
        key=rng_key,
        compute_dtype=torch.float64,
        second_moment_strategy=lambda_cgd_strategy(lambda_=0.5),
    )


@dataclasses.dataclass(frozen=True)
class _NoiseContinuityCase:
    make_noise: Callable[[RngKey], Any]
    make_input: Callable[[], Any]
    make_poison_input: Callable[[], Any]
    mechanism_kind: str
    buffered: bool = False


def _scalar_input(max_norm: float):
    return clipped(_noise_tree(), max_norm=max_norm)


def _per_group_input(*, fallback: float, head: float):
    return clipped(
        _noise_tree(),
        max_norm=_per_group_norm(fallback=fallback, head=head),
    )


def _paired_input(*, max_norm: float, squared_max_norm: float):
    return SecondMomentClippingOutput(
        grads=_scalar_input(max_norm),
        squared_grads=_scalar_input(squared_max_norm),
    )


_NOISE_CASES = [
    pytest.param(
        _NoiseContinuityCase(
            make_noise=lambda rng_key: gaussian_noise(
                noise_multiplier=1.0,
                key=rng_key,
                compute_dtype=torch.float64,
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="gaussian",
        ),
        id="gaussian",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(identity_strategy),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_identity",
        ),
        id="mf-identity",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: band_mf_strategy(bands=3, momentum=0.9)
            ),
            make_input=lambda: _per_group_input(fallback=1.0, head=2.0),
            make_poison_input=lambda: _per_group_input(
                fallback=7.0,
                head=8.0,
            ),
            mechanism_kind="mf_band",
            buffered=True,
        ),
        id="mf-band-per-group",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: blt_strategy(max_buffers=2, momentum=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_blt",
            buffered=True,
        ),
        id="mf-blt",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: bisr_strategy(bandwidth=3, momentum=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_bisr",
            buffered=True,
        ),
        id="mf-bisr",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: bsr_strategy(bandwidth=3, alpha=1.0, beta=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_bsr",
            buffered=True,
        ),
        id="mf-bsr",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(lambda: lambda_cgd_strategy(lambda_=0.5)),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_lambda_cgd",
        ),
        id="mf-lambda-cgd",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_paired_noise_factory,
            make_input=lambda: _paired_input(max_norm=1.0, squared_max_norm=1.0),
            make_poison_input=lambda: _paired_input(
                max_norm=7.0,
                squared_max_norm=49.0,
            ),
            mechanism_kind="mf_band",
            buffered=True,
        ),
        id="mf-paired-band-lambda-cgd",
    ),
]


def _assert_nested_equal(actual: Any, expected: Any, path: str = "root") -> None:
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype, path
        assert actual.device == expected.device, path
        assert torch.equal(actual, expected), path
        return
    if dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            _assert_nested_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(expected, Mapping):
        assert set(actual) == set(expected), path
        for key_ in expected:
            _assert_nested_equal(actual[key_], expected[key_], f"{path}[{key_!r}]")
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    assert actual == expected, path


def _nested_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _nested_tensors(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _nested_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_tensors(child)


def _noise_streams(output: Any):
    if isinstance(output, SecondMomentNoiseOutput):
        return output.noisy_grads, output.noisy_squared_grads
    return (output,)


class TestNoiseStreamContinuity:
    """DP noise state and outputs continue exactly across a runtime checkpoint."""

    @pytest.mark.parametrize("case", _NOISE_CASES)
    def test_continues_after_resume(self, tmp_path, case):
        uninterrupted_fn, uninterrupted_state = case.make_noise(key(42))
        uninterrupted_outputs = []
        for _ in range(_NOISE_STEPS):
            output, uninterrupted_state = uninterrupted_fn(
                case.make_input(),
                uninterrupted_state,
            )
            uninterrupted_outputs.append(output)

        interrupted_fn, interrupted_state = case.make_noise(key(42))
        interrupted_outputs = []
        for _ in range(_NOISE_CHECKPOINT_STEP):
            output, interrupted_state = interrupted_fn(
                case.make_input(),
                interrupted_state,
            )
            interrupted_outputs.append(output)

        for actual, expected in zip(
            interrupted_outputs,
            uninterrupted_outputs[:_NOISE_CHECKPOINT_STEP],
            strict=True,
        ):
            _assert_nested_equal(actual, expected)
        if case.buffered:
            assert any(
                bool(torch.count_nonzero(value))
                for value in _nested_tensors(interrupted_state)
            )

        path = str(tmp_path / "dp_runtime.pt")
        is_mf = case.mechanism_kind.startswith("mf_")
        ckpt.save_dp_runtime_state(
            path,
            clip_state=FixedClipState(),
            noise_state=interrupted_state,
            sampler_state=None,
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=1,
            expected_batch_size=1,
            total_steps=_NOISE_STEPS,
            mechanism_kind=case.mechanism_kind,
            mf_n_steps=_NOISE_STEPS if is_mf else None,
            mf_min_sep=_NOISE_STEPS if is_mf else None,
            mf_max_participations=1 if is_mf else None,
        )
        checkpoint = ckpt.load_dp_runtime_state(path)

        for restore_mode in ("fresh", "poisoned"):
            poisoned = restore_mode == "poisoned"
            restore_key = (
                RngKey(seed=999, impl="poison-template") if poisoned else key(42)
            )
            resumed_fn, state_template = case.make_noise(restore_key)
            if poisoned:
                _, state_template = resumed_fn(
                    case.make_poison_input(),
                    state_template,
                )
            resumed_state = opaque_from_state_dict(
                state_template,
                checkpoint.noise_state,
            )
            _assert_nested_equal(
                resumed_state,
                interrupted_state,
                path=f"{restore_mode}.restored_state",
            )

            resumed_outputs = list(interrupted_outputs)
            for _ in range(_NOISE_CHECKPOINT_STEP, _NOISE_STEPS):
                output, resumed_state = resumed_fn(
                    case.make_input(),
                    resumed_state,
                )
                resumed_outputs.append(output)

            for actual, expected in zip(
                resumed_outputs,
                uninterrupted_outputs,
                strict=True,
            ):
                _assert_nested_equal(actual, expected)
            _assert_nested_equal(resumed_state, uninterrupted_state)

            first_output = uninterrupted_outputs[0]
            resumed_output = resumed_outputs[_NOISE_CHECKPOINT_STEP]
            for resumed_stream, first_stream in zip(
                _noise_streams(resumed_output),
                _noise_streams(first_output),
                strict=True,
            ):
                for resumed_leaf, first_leaf in zip(
                    _nested_tensors(resumed_stream.pytree),
                    _nested_tensors(first_stream.pytree),
                    strict=True,
                ):
                    assert not torch.equal(resumed_leaf, first_leaf)


class TestRuntimeCheckpointDriftMetadata:
    """Per-field ``drift`` disposition on ``RuntimeCheckpoint`` is intact.

    The trainer's ``_warn_on_arg_drift`` reads these metadata keys to
    dispatch warn / raise / silent actions; if a field loses its
    ``drift`` tag, drift handling silently regresses to the default
    (``dp_relevant``), which is the safer side but hides the design
    intent.
    """

    def _field(self, name):
        for f in fields(ckpt.RuntimeCheckpoint):
            if f.name == name:
                return f
        raise AssertionError(f"no field {name!r} on RuntimeCheckpoint")

    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("sample_rate", "dp_relevant"),
            ("target_delta", "dp_relevant"),
            ("noise_multiplier", "dp_relevant"),
            ("expected_steps_per_epoch", "dp_relevant"),
            ("expected_batch_size", "dp_relevant"),
            ("mechanism_kind", "dp_relevant"),
            ("is_horizon_process", "dp_relevant"),
            ("horizon_process_state", "dp_relevant"),
            ("mf_n_steps", "dp_relevant"),
            ("mf_min_sep", "dp_relevant"),
            ("mf_max_participations", "dp_relevant"),
            ("lr_scheduler", "shape"),
            ("learning_rate", "shape"),
            ("warmup_steps", "shape"),
            ("lr_scheduler_kwargs", "shape"),
        ],
    )
    def test_string_dispositions(self, field_name, expected):
        meta = self._field(field_name).metadata
        assert meta.get("compare_on_resume") is True, field_name
        assert meta.get("drift") == expected, field_name

    def test_total_steps_per_mechanism_override(self):
        """``total_steps`` is silent for DP-SGD (intentional extend), forbidden
        for DP-FTRL (MF strategy is shape-locked)."""
        meta = self._field("total_steps").metadata
        assert meta.get("compare_on_resume") is True
        drift = meta.get("drift")
        assert isinstance(drift, dict)
        assert drift.get("gaussian") == "intentional_extend"
        assert drift.get("default") == "dp_relevant"


class TestDriftDispositionResolution:
    """``_resolve_drift_disposition`` picks the right rule per mechanism."""

    def test_string_disposition_passthrough(self):
        from opaque.api.transformers.trainer._dp_trainer import (
            _resolve_drift_disposition,
        )

        meta = {"drift": "shape"}
        assert _resolve_drift_disposition(meta, "gaussian") == "shape"
        assert _resolve_drift_disposition(meta, "mf_band") == "shape"

    def test_dict_disposition_per_mechanism(self):
        from opaque.api.transformers.trainer._dp_trainer import (
            _resolve_drift_disposition,
        )

        meta = {
            "drift": {
                "gaussian": "intentional_extend",
                "default": "dp_relevant",
            }
        }
        assert _resolve_drift_disposition(meta, "gaussian") == "intentional_extend"
        assert _resolve_drift_disposition(meta, "mf_band") == "dp_relevant"
        assert _resolve_drift_disposition(meta, "mf_blt") == "dp_relevant"

    def test_default_disposition_when_missing(self):
        from opaque.api.transformers.trainer._dp_trainer import (
            _resolve_drift_disposition,
        )

        # Missing ``drift`` key falls back to ``dp_relevant`` (safest).
        assert _resolve_drift_disposition({}, "gaussian") == "dp_relevant"
