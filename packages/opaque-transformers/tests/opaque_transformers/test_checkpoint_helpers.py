"""Unit tests for ``trainer/_checkpoint.py`` helpers."""

from __future__ import annotations

import os

import pytest
import torch

from opaque.api.engine.clipping.types import FixedClipState
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.scheduling import (
    constant_schedule,
    cosine_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.serialization import from_state_dict as opaque_from_state_dict
from opaque.serialization import state_dict as opaque_state_dict
import opaque.api.transformers.trainer._checkpoint as ckpt


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
        assert [os.path.basename(p) for p in listing] == [
            "checkpoint-10",
            "checkpoint-50",
            "checkpoint-100",
        ]
        assert (
            os.path.basename(ckpt.get_last_checkpoint(str(tmp_path)))
            == "checkpoint-100"
        )

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
        names = [os.path.basename(p) for p in ckpt.list_checkpoints(str(tmp_path))]
        assert names == ["checkpoint-4", "checkpoint-5"]

    def test_protects_best_when_outside_window(self, tmp_path):
        for s in (1, 2, 3, 4, 5):
            self._make(tmp_path, s)
        best = str(tmp_path / "checkpoint-1")
        ckpt.rotate_checkpoints(
            str(tmp_path), save_total_limit=2, best_model_checkpoint=best
        )
        names = sorted(
            os.path.basename(p) for p in ckpt.list_checkpoints(str(tmp_path))
        )
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
        names = sorted(
            os.path.basename(p) for p in ckpt.list_checkpoints(str(tmp_path))
        )
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
        ckpt.save_dp_runtime_state(
            path,
            clip_state=clip,
            noise_state=noise,
            sampler_state={
                "key_seed": 5,
                "key_impl": "opaque_threefry_like",
                "consumed": 2,
                "num_samples": 100,
                "sample_rate": 0.1,
                "n_steps": 10,
                "truncated_batch_size": None,
            },
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.1,
            expected_steps_per_epoch=10,
            expected_batch_size=32,
            total_steps=30,
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

    def test_unsupported_clip_state_type_raises(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        _, noise = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(TypeError, match="clip_state must be a ClipState"):
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
        with pytest.raises(TypeError, match="noise_state must be a NoiseState"):
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
        with pytest.raises(ValueError, match="unsupported dp_state"):
            ckpt.load_dp_runtime_state(path)


class TestLrSchedulerRoundTrip:
    """LR schedule (callable dataclass) round-trips through state_dict."""

    def test_lr_scheduler_name_constant_exposed(self):
        assert ckpt.LR_SCHEDULER_NAME == "lr_scheduler.pt"

    def test_linear_round_trip(self, tmp_path):
        s = linear_schedule(
            init_value=1e-4,
            end_value=0.0,
            transition_steps=100,
            transition_begin=10,
        )
        path = str(tmp_path / ckpt.LR_SCHEDULER_NAME)
        torch.save(opaque_state_dict(s), path)

        template = linear_schedule(
            init_value=0.0,
            end_value=0.0,
            transition_steps=1,
            transition_begin=0,
        )
        restored = opaque_from_state_dict(
            template,
            torch.load(path, map_location="cpu", weights_only=False),
        )
        for step in (0, 5, 10, 25, 50, 75, 100, 109):
            assert s(step) == restored(step), f"mismatch at step {step}"

    def test_constant_round_trip(self, tmp_path):
        s = constant_schedule(5e-5)
        path = str(tmp_path / ckpt.LR_SCHEDULER_NAME)
        torch.save(opaque_state_dict(s), path)
        template = constant_schedule(0.0)
        restored = opaque_from_state_dict(
            template,
            torch.load(path, map_location="cpu", weights_only=False),
        )
        for step in (0, 1, 100, 10_000):
            assert s(step) == restored(step) == pytest.approx(5e-5)

    def test_warmup_linear_round_trip_resume_scenario(self, tmp_path):
        """Phase-1 (50 steps) state overlaid on a fresh phase-2 (100 steps)
        schedule must reproduce phase-1's LR trajectory."""
        decay_p1 = linear_schedule(
            init_value=1e-4,
            end_value=0.0,
            transition_steps=50,
            transition_begin=10,
        )
        phase1 = with_warmup(
            decay_p1, transition_steps=10, ramp="linear", init_value=0.0
        )

        path = str(tmp_path / ckpt.LR_SCHEDULER_NAME)
        torch.save(opaque_state_dict(phase1), path)

        # Phase-2 freshly builds the schedule with the NEW max_steps=100,
        # then ``_apply_runtime_state`` overlays the saved phase-1 state.
        decay_p2_fresh = linear_schedule(
            init_value=1e-4,
            end_value=0.0,
            transition_steps=100,
            transition_begin=10,
        )
        phase2_fresh = with_warmup(
            decay_p2_fresh, transition_steps=10, ramp="linear", init_value=0.0
        )
        phase2_restored = opaque_from_state_dict(
            phase2_fresh,
            torch.load(path, map_location="cpu", weights_only=False),
        )

        # The inner schedule's transition_steps must reflect phase-1's 50,
        # not phase-2's freshly-built 100.
        assert phase2_restored.schedule.transition_steps == 50
        for step in (0, 5, 10, 25, 40, 50, 60):
            assert phase1(step) == phase2_restored(step), f"mismatch at step {step}"

    def test_cosine_round_trip(self, tmp_path):
        s = cosine_schedule(
            init_value=1e-4,
            end_value=0.0,
            transition_steps=100,
        )
        path = str(tmp_path / ckpt.LR_SCHEDULER_NAME)
        torch.save(opaque_state_dict(s), path)
        template = cosine_schedule(
            init_value=0.0,
            end_value=0.0,
            transition_steps=1,
        )
        restored = opaque_from_state_dict(
            template,
            torch.load(path, map_location="cpu", weights_only=False),
        )
        for step in (0, 25, 50, 75, 100, 150):
            assert s(step) == pytest.approx(restored(step))
