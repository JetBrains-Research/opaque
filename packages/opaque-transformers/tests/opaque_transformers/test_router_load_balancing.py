# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DPTrainer integration of the MoE router-load release (T10 to T22).

A tiny random-init Mellum (E=8, k=2, L=2) trains a few DP-SGD steps on CPU
with ``router_load_release`` on; the tests pin the probe hygiene, the
checkpoint sidecar, microbatch invariance, the logged-metric hygiene, the
configuration errors, the DPO pooling and the monitor decision rule.
"""

from __future__ import annotations

import contextlib
import functools
import math
import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import MellumConfig, MellumForCausalLM
from transformers.trainer_callback import TrainerCallback

from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.api.patches.transformers.components.moe_stats import (
    router_load_and_probs,
)
from opaque.api.transformers import moe_load
from opaque.api.transformers.moe_load import PROBE_NAME, RouterLoadState
from opaque.api.transformers.trainer._router_load import (
    ROUTER_LOAD_STATE_NAME,
    RouterLoadCallback,
)
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.patches import packed_sequences
from opaque.serialization import from_state_dict
from opaque.transformers.trainer import DPTrainer, TrainingArguments
from opaque.transformers.trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from opaque.types import PerGroup

NUM_EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 2
T_MAX = 16
VOCAB = 128
TINY_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rope_theta": 10000.0,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "moe_intermediate_size": 32,
    "router_aux_loss_coef": 0.01,
}
_TRAINABLE = ("q_proj", "k_proj", "v_proj", "o_proj", ".mlp.gate.")
GUARD = 1e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_mellum(seed: int = 0) -> MellumForCausalLM:
    torch.manual_seed(seed)
    model = MellumForCausalLM(MellumConfig(**TINY_CONFIG))
    for name, p in model.named_parameters():
        p.requires_grad_(any(s in name for s in _TRAINABLE))
    return model.train()


@pytest.fixture(autouse=True)
def _restore_class_forwards():
    """Undo class-level forward patches so every test starts unpatched."""
    probe_model = MellumForCausalLM(MellumConfig(**TINY_CONFIG))
    saved = {}
    for module in probe_model.modules():
        cls = type(module)
        if cls not in saved:
            saved[cls] = cls.__dict__.get("forward")
    yield
    for cls, forward in saved.items():
        if forward is None:
            if "forward" in cls.__dict__:
                delattr(cls, "forward")
        else:
            cls.forward = forward


@pytest.fixture(autouse=True)
def _restore_packed_sequences_policy():
    """The trainer sets the process-level masking policy; put it back."""
    from opaque.patches import set_packed_sequences

    saved = packed_sequences()
    yield
    set_packed_sequences(saved)


class _RaggedDataset(TorchDataset):
    """Right-padded ragged rows with an attention mask and ``-100`` labels."""

    def __init__(self, n: int = 32, seed: int = 1) -> None:
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(3, VOCAB, (n, T_MAX), generator=g)
        lengths = torch.randint(T_MAX // 2, T_MAX + 1, (n,), generator=g)
        mask = (torch.arange(T_MAX)[None] < lengths[:, None]).long()
        self.input_ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
        self.attention_mask = mask
        self.labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 4,
        "max_steps": 3,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": [],
        "disable_tqdm": True,
        "use_cpu": True,
        "seed": 0,
        "privacy_noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "learning_rate": 1e-3,
        "optim": "adamw",
        "dataloader_num_workers": 0,
        "router_load_max_tokens": T_MAX,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _trainer(tmp_path, dataset=None, **overrides) -> DPTrainer:
    return DPTrainer(
        model=_tiny_mellum(),
        args=_args(tmp_path, **overrides),
        train_dataset=dataset or _RaggedDataset(),
        data_collator=_collate,
    )


@contextlib.contextmanager
def _active_context(trainer: DPTrainer, **setup_kwargs):
    """Run ``_setup_training`` and expose the live context without the loop."""
    ctx = trainer._setup_training(**setup_kwargs)
    trainer._ctx = ctx
    try:
        yield ctx
    finally:
        trainer._ctx = None
        trainer._router_load = None
        trainer._detach_router_load_probe()


def _batch(trainer: DPTrainer, rows: range = range(4)) -> dict[str, torch.Tensor]:
    ds = trainer.train_dataset
    return trainer._prepare_input(_collate([ds[i] for i in rows]))


def _clipped(ctx, batch):
    """Clipped per-example gradient sum and aux of one batch (no noise)."""
    trainer_args = tuple(batch[k] for k in ctx.batch_keys)
    (grads, aux), _ = ctx.grad_fn(
        ctx.trainable_params, *trainer_args, state=ctx.clip_state
    )
    return grads, aux


class _Recorder(TrainerCallback):
    """Snapshot the probe leaf, the state and ``f_tilde`` at every step."""

    def __init__(self, trainer: DPTrainer) -> None:
        self.trainer = trainer
        self.probes: list[torch.Tensor] = []
        self.f_tildes: list[torch.Tensor] = []
        self.states: list[RouterLoadState] = []
        self.opt_probe_leaves: list[list[torch.Tensor]] = []

    def on_optimizer_step(self, args, state, control, trainable_params=None, **kw):
        self.probes.append(trainable_params[PROBE_NAME].detach().clone())
        rt = self.trainer._router_load
        self.f_tildes.append(rt.state.f_tilde.clone())
        self.states.append(rt.state)
        # The optimizer's moment tensors of the probe (same shape as the
        # probe; the schedule's per-parameter step counter is a scalar).
        self.opt_probe_leaves.append(
            [
                leaf.detach().clone()
                for path, leaf in _walk(self.trainer._ctx.opt_state)
                if leaf.shape == (NUM_LAYERS, NUM_EXPERTS) and PROBE_NAME in path
            ]
        )
        return control


def _walk(obj, path: str = ""):
    """Yield ``(path, tensor)`` for every tensor inside an optimizer state."""
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        for k in obj._fields:
            yield from _walk(getattr(obj, k), f"{path}/{k}")
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}/{i}")
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            yield from _walk(v, f"{path}/{k}")


# ---------------------------------------------------------------------------
# T10: probe hygiene, loss value, off path
# ---------------------------------------------------------------------------


def test_off_keeps_the_plain_path(tmp_path):
    trainer = _trainer(tmp_path, router_load_release="off")
    assert trainer._router_load is None
    assert "opaque_router_logits" not in str(
        __import__("inspect").signature(MellumForCausalLM.forward)
    )
    assert packed_sequences() is None
    with _active_context(trainer) as ctx:
        assert PROBE_NAME not in ctx.trainable_params
        assert PROBE_NAME not in dict(trainer.model.named_parameters())
        assert isinstance(ctx.clip_norm, float)
        assert ctx.router_load is None
        assert trainer._router_load_forward_kwargs() == {}
        assert not any(
            isinstance(cb, RouterLoadCallback)
            for cb in trainer.callback_handler.callbacks
        )
        metrics = trainer.training_step(trainer.model, _batch(trainer))
    assert not any(k.startswith("router_load/") for k in metrics)


def test_probe_is_zero_loss_is_ce_and_optimizer_never_moves_it(tmp_path):
    trainer = _trainer(tmp_path, router_load_release="surrogate", max_steps=3)
    recorder = _Recorder(trainer)
    trainer.add_callback(recorder)
    out = trainer.train()
    assert out.global_step == 3
    assert len(recorder.probes) == 3
    for probe, leaves in zip(recorder.probes, recorder.opt_probe_leaves, strict=True):
        assert probe.shape == (NUM_LAYERS, NUM_EXPERTS)
        assert not probe.any()
        assert leaves, "the optimizer state carries the probe's moments"
        assert all(not leaf.any() for leaf in leaves)
    # ``f_tilde`` is a distribution over experts summing to k, in [0, 1].
    for f in recorder.f_tildes:
        assert f.min() >= 0
        assert f.max() <= 1
        assert torch.allclose(f.sum(), torch.tensor(float(TOP_K)), atol=1e-5)
    # The model handed back carries no probe; the release runtime is cleared.
    assert PROBE_NAME not in dict(trainer.model.named_parameters())
    assert trainer._router_load is None


def test_loss_value_is_ce_and_gradient_is_ce_plus_alpha_surrogate(tmp_path):
    alpha = 0.3
    trainer = _trainer(
        tmp_path,
        router_load_release="surrogate",
        router_aux_loss_coef=alpha,
        privacy_noise_multiplier=0.0,
        clipping_norm=1e6,
    )
    with _active_context(trainer) as ctx:
        batch = _batch(trainer)
        trainer._augment_inputs(batch)
        rt = trainer._router_load
        # A non-uniform public estimate: at balance the surrogate vanishes.
        target = torch.linspace(0.1, 0.4, NUM_EXPERTS)
        rt.target.copy_(target * TOP_K / target.sum())
        merged = {**ctx.frozen_params, **ctx.trainable_params}
        example = {k: v[0] for k, v in batch.items()}

        def dp_loss(trainable, ex):
            return trainer.compute_per_example_loss(
                ctx.fmodel, {**ctx.frozen_params, **trainable}, ex
            )

        def ce_loss(trainable, ex):
            return ctx.fmodel(
                {**ctx.frozen_params, **trainable}, **ex, opaque_fused_loss_only=True
            )["loss"]

        def surrogate(trainable, ex):
            out = ctx.fmodel(
                {**ctx.frozen_params, **trainable}, **ex, opaque_router_logits=True
            )
            _h, probs, count = router_load_and_probs(
                out.router_logits,
                ex["attention_mask"],
                top_k=TOP_K,
                num_layers=NUM_LAYERS,
            )
            w = count / rt.mean_tokens
            return NUM_EXPERTS * w * ((rt.target - TOP_K / NUM_EXPERTS) * probs).sum()

        # Value: the per-example loss is the CE bit for bit.
        assert torch.equal(
            dp_loss(ctx.trainable_params, example),
            ce_loss(ctx.trainable_params, example),
        )
        # Gradient of the model leaves: grad CE + alpha grad S.
        g_dp = torch.func.grad(dp_loss)(ctx.trainable_params, example)
        g_ce = torch.func.grad(ce_loss)(ctx.trainable_params, example)
        g_s = torch.func.grad(surrogate)(ctx.trainable_params, example)
        for name in ctx.trainable_params:
            if name == PROBE_NAME:
                continue
            torch.testing.assert_close(
                g_dp[name], g_ce[name] + alpha * g_s[name], atol=1e-6, rtol=1e-5
            )
        # The probe leaf carries lam * w * (h - k/E) of the example.
        out = ctx.fmodel(merged, **example, opaque_router_logits=True)
        h_layers, _p, count = router_load_and_probs(
            out.router_logits,
            example["attention_mask"],
            top_k=TOP_K,
            num_layers=NUM_LAYERS,
        )
        expected = rt.lam * (count / rt.mean_tokens) * (h_layers - TOP_K / NUM_EXPERTS)
        torch.testing.assert_close(g_dp[PROBE_NAME], expected, atol=1e-6, rtol=1e-5)
        # The surrogate direction reaches the router through the captured logits.
        router_grads = [g_s[n] for n in g_s if ".mlp.gate." in n]
        assert router_grads
        assert any(g.abs().sum() > 0 for g in router_grads)


def test_monitor_differs_from_off_only_by_the_noise_inflation(tmp_path):
    """Same forward on both sides; only the per-group sigma differs."""
    ratio = 0.1
    off = _trainer(
        tmp_path / "off",
        router_load_release="off",
        performance_kernels_config={"fused_linear_cross_entropy": True},
    )
    on = _trainer(
        tmp_path / "on", router_load_release="monitor", router_load_ratio=ratio
    )
    with _active_context(off) as ctx_off, _active_context(on) as ctx_on:
        batch_off = _batch(off)
        batch_on = _batch(on)
        on._augment_inputs(batch_on)
        grads_off, aux_off = _clipped(ctx_off, batch_off)
        grads_on, aux_on = _clipped(ctx_on, batch_on)
        # Same clipped gradient sum on the model leaves (the per-group clipper
        # scales each example's gradient group by the same factor).
        for name, leaf in grads_off.pytree.items():
            torch.testing.assert_close(
                grads_on.pytree[name], leaf, atol=1e-6, rtol=1e-5
            )
        torch.testing.assert_close(aux_on.loss_values, aux_off.loss_values)
        noisy_off, _ = ctx_off.noise_fn(grads_off, ctx_off.noise_state)
        noisy_on, _ = ctx_on.noise_fn(grads_on, ctx_on.noise_state)
        sigma_off = float(noisy_off.noise_stddev)
        sigma_on = float(noisy_on.noise_stddev.values["fallback"])
        assert sigma_on == pytest.approx(
            sigma_off * math.sqrt(1 + ratio * (1 + GUARD)), rel=1e-9
        )
        # Which is exactly the MSE-optimal per-group allocation of the engine.
        expected = per_group_noise_stddev(grads_on.max_norm, ctx_on.noise_multiplier)
        assert sigma_on == pytest.approx(expected.values["fallback"], rel=1e-12)


def test_monitor_at_zero_noise_updates_the_model_exactly_like_off(tmp_path):
    common = {"privacy_noise_multiplier": 0.0, "clipping_norm": 1e6, "optim": "sgd"}
    off = _trainer(
        tmp_path / "off",
        router_load_release="off",
        performance_kernels_config={"fused_linear_cross_entropy": True},
        **common,
    )
    on = _trainer(tmp_path / "on", router_load_release="monitor", **common)
    with _active_context(off) as ctx_off, _active_context(on) as ctx_on:
        off.training_step(off.model, _batch(off))
        on.training_step(on.model, _batch(on))
        for name, leaf in ctx_off.trainable_params.items():
            torch.testing.assert_close(
                ctx_on.trainable_params[name], leaf, atol=1e-7, rtol=1e-6
            )
        assert not ctx_on.trainable_params[PROBE_NAME].any()


# ---------------------------------------------------------------------------
# T11: checkpoint sidecar and resume
# ---------------------------------------------------------------------------


def _run_with_recorder(tmp_path, out_dir, **overrides):
    trainer = _trainer(
        tmp_path / out_dir,
        router_load_release="monitor",
        max_steps=4,
        **overrides,
    )
    # The trainer's weight loader reads the saved file with a strict
    # ``load_state_dict`` and does not apply Hugging Face's checkpoint weight
    # conversion; write the stacked-experts tensors as they live in memory so
    # the resume below can load them (independent of the release).
    model = trainer.model
    trainer.model.save_pretrained = functools.partial(
        model.save_pretrained, save_original_format=False
    )
    recorder = _Recorder(trainer)
    trainer.add_callback(recorder)
    trainer.train()
    return trainer, recorder


def test_checkpoint_sidecar_round_trips_and_resume_continues_the_filter(tmp_path):
    continuous_trainer, continuous = _run_with_recorder(
        tmp_path, "continuous", save_strategy="steps", save_steps=2
    )
    ckpt_dir = tmp_path / "continuous" / "checkpoint-2"
    sidecar = ckpt_dir / ROUTER_LOAD_STATE_NAME
    assert sidecar.exists()
    payload = torch.load(sidecar, weights_only=False)
    saved_state = continuous.states[1]
    restored = from_state_dict(saved_state, payload["state"])
    for field in RouterLoadState.__dataclass_fields__:
        a, b = getattr(restored, field), getattr(saved_state, field)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field
        else:
            assert a == b, field
    # The saved model weights carry no probe.
    from safetensors.torch import load_file

    weights = load_file(str(ckpt_dir / "model.safetensors"))
    assert PROBE_NAME not in weights
    del continuous_trainer

    resumed_trainer = _trainer(
        tmp_path / "resumed",
        router_load_release="monitor",
        max_steps=4,
        save_strategy="steps",
        save_steps=2,
        resume_from_checkpoint=str(ckpt_dir),
    )
    resumed = _Recorder(resumed_trainer)
    resumed_trainer.add_callback(resumed)
    resumed_trainer.train()
    assert len(resumed.f_tildes) == 2
    assert torch.equal(resumed.f_tildes[0], continuous.f_tildes[2])
    assert torch.equal(resumed.f_tildes[1], continuous.f_tildes[3])
    assert torch.equal(resumed.states[0].m, continuous.states[2].m)
    assert resumed.states[0].step == 3


def test_resume_rejects_configuration_drift_and_a_missing_sidecar(tmp_path):
    _run_with_recorder(tmp_path, "run", save_strategy="steps", save_steps=2)
    ckpt_dir = tmp_path / "run" / "checkpoint-2"
    drifted = _trainer(
        tmp_path / "drifted",
        router_load_release="monitor",
        router_load_ratio=0.05,
        max_steps=4,
        resume_from_checkpoint=str(ckpt_dir),
    )
    with pytest.raises(CheckpointError, match="ratio"):
        drifted.train()
    (ckpt_dir / ROUTER_LOAD_STATE_NAME).unlink()
    missing = _trainer(
        tmp_path / "missing",
        router_load_release="monitor",
        max_steps=4,
        resume_from_checkpoint=str(ckpt_dir),
    )
    with pytest.raises(CheckpointError, match=ROUTER_LOAD_STATE_NAME):
        missing.train()


# ---------------------------------------------------------------------------
# T13: microbatch invariance
# ---------------------------------------------------------------------------


def test_microbatch_chunks_match_a_single_chunk(tmp_path):
    single = _trainer(tmp_path / "single", router_load_release="surrogate")
    chunked = _trainer(tmp_path / "chunked", router_load_release="surrogate")
    with (
        _active_context(single) as ctx_single,
        _active_context(chunked, microbatch_size_override=2) as ctx_chunked,
    ):
        batch_single = _batch(single)
        batch_chunked = _batch(chunked)
        single._augment_inputs(batch_single)
        chunked._augment_inputs(batch_chunked)
        grads_single, aux_single = _clipped(ctx_single, batch_single)
        grads_chunked, aux_chunked = _clipped(ctx_chunked, batch_chunked)
        for name, leaf in grads_single.pytree.items():
            torch.testing.assert_close(
                grads_chunked.pytree[name], leaf, atol=1e-6, rtol=1e-5
            )
        torch.testing.assert_close(aux_chunked.loss_values, aux_single.loss_values)
        # Same release in, same estimate out.
        state = single._router_load.state
        next_single = moe_load.update(state, grads_single.pytree[PROBE_NAME])
        next_chunked = moe_load.update(state, grads_chunked.pytree[PROBE_NAME])
        torch.testing.assert_close(
            next_chunked.f_tilde, next_single.f_tilde, atol=1e-6, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# T17: telemetry hygiene
# ---------------------------------------------------------------------------


def test_logged_norms_exclude_the_probe_and_monitor_curves_are_the_summary(tmp_path):
    trainer = _trainer(tmp_path, router_load_release="surrogate")
    with _active_context(trainer) as ctx:
        batch = _batch(trainer)
        trainer._augment_inputs(batch)
        _grads, aux = _clipped(ctx, batch)
        metrics = trainer.training_step(trainer.model, batch)
        others = torch.stack([v for k, v in aux.group_norms.items() if k != PROBE_NAME])
        expected_grad_norm = torch.sqrt((others**2).sum(0)).mean().item()
        assert metrics["grad_norm"] == pytest.approx(expected_grad_norm, rel=1e-5)
        clipped = torch.sqrt(
            aux.clipped_grad_norms**2 - aux.group_norms[PROBE_NAME] ** 2
        )
        assert metrics["clipped_grad_norm"] == pytest.approx(
            clipped.mean().item(), rel=1e-5
        )
        assert PROBE_NAME not in metrics["group_metrics"]
        assert set(metrics["group_metrics"]) == {"fallback"}
        assert metrics["loss"] == pytest.approx(aux.loss_values.mean().item())
        summary = moe_load.summary(trainer._router_load.state)
        assert {
            k: v for k, v in metrics.items() if k.startswith("router_load/")
        } == summary
        assert summary["router_load/tripped"] == 0.0
        assert not any("probe" in k for k in metrics if k != "group_metrics")
    # ``loss_aux`` carries nothing new compared with the plain path.
    plain = _trainer(tmp_path / "plain", router_load_release="off")
    with _active_context(plain):
        plain_metrics = plain.training_step(plain.model, _batch(plain))
    assert set(metrics.get("loss_aux", {})) == set(plain_metrics.get("loss_aux", {}))
    assert set(metrics) - set(plain_metrics) == {
        "group_metrics",
        "clip_rate_max",
        *summary,
    }


def test_logged_norms_are_invariant_to_the_probe_scale(tmp_path):
    small = _trainer(
        tmp_path / "small", router_load_release="surrogate", router_load_ratio=0.02
    )
    large = _trainer(
        tmp_path / "large", router_load_release="surrogate", router_load_ratio=0.5
    )
    with _active_context(small), _active_context(large):
        batch_small = _batch(small)
        batch_large = _batch(large)
        m_small = small.training_step(small.model, batch_small)
        m_large = large.training_step(large.model, batch_large)
    # lam differs by 25x, the probe group norm with it; the logged norms do not.
    assert small._router_load is None
    assert m_small["grad_norm"] == pytest.approx(m_large["grad_norm"], rel=1e-6)
    assert m_small["clipped_grad_norm"] == pytest.approx(
        m_large["clipped_grad_norm"], rel=1e-6
    )
    for name in ("grad_norm", "clip_rate", "clipping_norm"):
        assert (
            m_small["group_metrics"]["fallback"][name]
            == m_large["group_metrics"]["fallback"][name]
        )
    # Only the gradient-noise inflation depends on the probe share.
    assert (
        m_large["group_metrics"]["fallback"]["noise_std"]
        > m_small["group_metrics"]["fallback"]["noise_std"]
    )


def test_evaluation_during_training_and_best_model_reload(tmp_path):
    trainer = DPTrainer(
        model=_tiny_mellum(),
        args=_args(
            tmp_path,
            router_load_release="surrogate",
            max_steps=2,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        train_dataset=_RaggedDataset(),
        eval_dataset=_RaggedDataset(8, seed=5),
        data_collator=_collate,
    )
    trainer.model.save_pretrained = functools.partial(
        trainer.model.save_pretrained, save_original_format=False
    )
    trainer.train()
    evals = [r for r in trainer.state.log_history if "eval_loss" in r]
    assert len(evals) == 2
    assert all(math.isfinite(r["eval_loss"]) for r in evals)
    assert trainer.state.best_model_checkpoint is not None
    assert PROBE_NAME not in dict(trainer.model.named_parameters())
    # Evaluation after training runs on the plain path.
    assert math.isfinite(trainer.evaluate()["eval_loss"])


def test_log_rows_carry_the_monitor_curves(tmp_path):
    trainer = _trainer(tmp_path, router_load_release="monitor", max_steps=2)
    trainer.train()
    rows = [r for r in trainer.state.log_history if "router_load/D" in r]
    stepped = [r for r in trainer.state.log_history if r.get("batch_size", 0) > 0]
    assert rows
    assert len(rows) == len(stepped)
    assert {"router_load/f_min", "router_load/noise_std", "router_load/tripped"} <= set(
        rows[0]
    )


# ---------------------------------------------------------------------------
# T19: configuration errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"clipping_mode": "auto"},
        {"clipping_mode": "adaptive"},
        {"router_aux": "per_layer"},
        {"performance_kernels_config": {"fused_linear_cross_entropy": False}},
        {"clipping_norm": math.inf, "privacy_noise_multiplier": 0.0},
        {"router_load_release": "sometimes"},
        {"router_load_ratio": 0.0},
        {"router_load_filter_kind": "median"},
    ],
)
def test_invalid_arguments_raise(tmp_path, overrides):
    kwargs = {"router_load_release": "surrogate", **overrides}
    with pytest.raises(ConfigurationError):
        _args(tmp_path, **kwargs)


def test_family_without_a_router_raises(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    )
    trainer = DPTrainer(
        model=model,
        args=_args(tmp_path, router_load_release="monitor"),
        train_dataset=_RaggedDataset(),
        data_collator=_collate,
    )
    with pytest.raises(ConfigurationError, match="mixture-of-experts"):
        trainer._setup_training()


def test_missing_chunked_forward_raises(tmp_path, monkeypatch):
    import opaque.patches

    monkeypatch.setattr(opaque.patches, "apply_model_patches", lambda *a, **k: None)
    with pytest.raises(ConfigurationError, match="opaque_router_logits"):
        _trainer(tmp_path, router_load_release="monitor")


def test_packed_sequences_policy_follows_the_public_flag(tmp_path):
    _trainer(tmp_path / "a", router_load_release="monitor")
    assert packed_sequences() is False
    _trainer(tmp_path / "b", router_load_release="monitor", packed_sequences=True)
    assert packed_sequences() is True
    _trainer(tmp_path / "c", router_load_release="off", packed_sequences=False)
    assert packed_sequences() is False
    from opaque.patches import set_packed_sequences

    set_packed_sequences(None)


# ---------------------------------------------------------------------------
# T22: decision rule
# ---------------------------------------------------------------------------


def _synthetic_callback(
    mode: str, *, num_experts=64, top_k=8, num_layers=28, ratio=0.02, shrink=True
):
    batch = 256.0
    trainable = {"w": torch.zeros(4), PROBE_NAME: torch.zeros(num_layers, num_experts)}
    bounds, lam = moe_load.probe_bounds(
        0.9,
        trainable,
        ratio=ratio,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        mean_tokens=1024.0,
        max_tokens=1024.0,
    )
    max_norm = bounds / batch
    nm = 0.5622
    phi = moe_load.filter_factors(
        None, n_steps=15625, kind="ema", beta=0.99, window=256, num_experts=num_experts
    )
    state = moe_load.initial_state(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        ratio=ratio,
        lam=lam,
        max_norm=max_norm,
        noise_multiplier=nm,
        alpha=0.0,
        shrink=shrink,
        mean_tokens=1024.0,
        max_tokens=1024.0,
        phi=phi,
    )
    sigma = per_group_noise_stddev(max_norm, nm).values[PROBE_NAME]
    callback = RouterLoadCallback(state, mode=mode, trip=0.5, alpha=1e-4)
    return callback, lam, sigma


def _feed(callback, leaf, step):
    grads = types.SimpleNamespace(pytree={PROBE_NAME: leaf})
    trainer_state = types.SimpleNamespace(global_step=step, logging_steps=1)
    callback.on_pre_optimizer_step(None, trainer_state, None, grads=grads)
    assert not leaf.any()


def test_no_false_alarm_under_balance():
    callback, _lam, sigma = _synthetic_callback("monitor_then_surrogate")
    g = torch.Generator().manual_seed(0)
    passes = 0
    steps = 1000
    raw_over_trip = 0
    for step in range(steps):
        leaf = sigma * torch.randn(callback.state.m_layers.shape, generator=g)
        _feed(callback, leaf, step)
        s = moe_load.summary(callback.state)
        raw_over_trip += s["router_load/D"] > 0.5
        passes += s["router_load/shrink"] > 0
    # The raw monitor exceeds the threshold only while the bias-corrected
    # estimate is still noise-dominated (a handful of early steps); the rule
    # reads the dead-zoned estimate, so it never trips.
    assert raw_over_trip < 10
    assert not callback.state.tripped
    assert callback.state.alpha_active == 0.0
    assert passes == 0
    assert torch.allclose(callback.state.f_tilde, torch.full((64,), 8 / 64))


def test_deviation_is_detected_within_one_window_and_only_alpha_switches():
    tripped_cb, lam, sigma = _synthetic_callback("monitor_then_surrogate")
    monitor_cb, _lam, _sigma = _synthetic_callback("monitor")
    share = 8 / 64
    d_true = torch.full((64,), -share / 63)
    d_true[0] = share  # expert 0 carries twice its share: deviation 1.0
    g = torch.Generator().manual_seed(1)
    tripped_at = None
    for step in range(100):
        noise = sigma * torch.randn(28, 64, generator=g)
        leaf = lam * d_true[None, :] + noise
        _feed(tripped_cb, leaf.clone(), step)
        _feed(monitor_cb, leaf.clone(), step)
        if tripped_cb.state.tripped and tripped_at is None:
            tripped_at = step
    assert tripped_at is not None
    assert tripped_at <= 50
    assert tripped_cb.state.alpha_active == 1e-4
    assert monitor_cb.state.tripped
    assert monitor_cb.state.alpha_active == 0.0
    # The switch touches nothing but the two consumer-owned fields.
    for field in RouterLoadState.__dataclass_fields__:
        if field in ("tripped", "alpha_active"):
            continue
        a, b = getattr(tripped_cb.state, field), getattr(monitor_cb.state, field)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, field
    assert moe_load.summary(tripped_cb.state)["router_load/D"] > 0.5
    assert tripped_cb.state.f_tilde[0] > share


def test_trip_needs_two_consecutive_logged_evaluations():
    callback, lam, _sigma = _synthetic_callback("monitor_then_surrogate", shrink=False)
    d_true = torch.full((64,), -(8 / 64) / 63)
    d_true[0] = 8 / 64
    leaf = lam * d_true[None, :].expand(28, 64)
    grads = types.SimpleNamespace(pytree={PROBE_NAME: leaf.clone()})
    # logging every 10 steps: step 9 is the first logged evaluation.
    callback.on_pre_optimizer_step(
        None, types.SimpleNamespace(global_step=9, logging_steps=10), None, grads=grads
    )
    assert callback.consecutive_over_trip == 1
    assert not callback.state.tripped
    for step in range(10, 19):  # unlogged steps do not count
        grads = types.SimpleNamespace(pytree={PROBE_NAME: leaf.clone()})
        callback.on_pre_optimizer_step(
            None,
            types.SimpleNamespace(global_step=step, logging_steps=10),
            None,
            grads=grads,
        )
    assert callback.consecutive_over_trip == 1
    assert not callback.state.tripped
    grads = types.SimpleNamespace(pytree={PROBE_NAME: leaf.clone()})
    callback.on_pre_optimizer_step(
        None, types.SimpleNamespace(global_step=19, logging_steps=10), None, grads=grads
    )
    assert callback.state.tripped
    assert callback.state.alpha_active == 1e-4


def test_switch_changes_neither_max_norm_nor_epsilon(tmp_path):
    monitor = _trainer(
        tmp_path / "m",
        router_load_release="monitor",
        max_steps=2,
        router_load_shrink=False,
    )
    switching = _trainer(
        tmp_path / "s",
        router_load_release="monitor_then_surrogate",
        router_load_trip=1e-9,
        router_load_shrink=False,
        max_steps=2,
    )
    rec_m, rec_s = _Recorder(monitor), _Recorder(switching)
    monitor.add_callback(rec_m)
    switching.add_callback(rec_s)
    monitor.train()
    switching.train()
    assert rec_s.states[-1].tripped
    assert rec_s.states[-1].alpha_active == 0.01
    assert rec_m.states[-1].tripped
    assert rec_m.states[-1].alpha_active == 0.0
    assert torch.equal(rec_s.f_tildes[-1], rec_m.f_tildes[-1])
    assert monitor._accountant.epsilon_at(1e-5) == switching._accountant.epsilon_at(
        1e-5
    )
    assert rec_s.states[-1].lam == rec_m.states[-1].lam


# ---------------------------------------------------------------------------
# SFT trainer
# ---------------------------------------------------------------------------


def _stub_tokenizer():
    return types.SimpleNamespace(
        pad_token_id=0,
        pad_token="<pad>",
        eos_token="</s>",
        save_pretrained=lambda *a, **k: None,
    )


def test_sft_chunked_nll_release_trains(tmp_path):
    dataset = Dataset.from_list(
        [{"input_ids": list(range(3, 3 + n))} for n in (8, 12, 16, 10, 8, 16, 9, 11)]
    )
    trainer = SFTTrainer(
        model=_tiny_mellum(),
        args=SFTConfig(
            output_dir=str(tmp_path),
            privacy_noise_multiplier=1.0,
            clipping_norm=1.0,
            per_device_train_batch_size=4,
            max_steps=2,
            max_length=T_MAX,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            seed=0,
            loss_type="chunked_nll",
            router_load_release="surrogate",
        ),
        train_dataset=dataset,
        processing_class=_stub_tokenizer(),
    )
    recorder = _Recorder(trainer)
    trainer.add_callback(recorder)
    out = trainer.train()
    assert out.global_step == 2
    assert all(not p.any() for p in recorder.probes)
    assert trainer.args.router_load_max_tokens is None
    rows = [r for r in trainer.state.log_history if "router_load/D" in r]
    assert len(rows) == 2


def test_sft_eager_nll_path_rejects_the_release(tmp_path):
    with pytest.raises(ConfigurationError, match="fused loss path"):
        SFTTrainer(
            model=_tiny_mellum(),
            args=SFTConfig(
                output_dir=str(tmp_path),
                privacy_noise_multiplier=1.0,
                max_length=T_MAX,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                loss_type="nll",
                log_completion_metrics=True,
                router_load_release="monitor",
            ),
            train_dataset=Dataset.from_list([{"input_ids": [3, 4, 5, 6]}] * 4),
            processing_class=_stub_tokenizer(),
        )


# ---------------------------------------------------------------------------
# T21: DPO pooling
# ---------------------------------------------------------------------------


def _pref_dataset() -> Dataset:
    rows = []
    for i in range(8):
        chosen = [1, 2, 3, 7, 8, 9][: 4 + i % 3]
        rejected = [1, 2, 3, 9, 10, 11, 12][: 4 + (i + 1) % 4]
        rows.append(
            {
                "chosen_input_ids": chosen,
                "rejected_input_ids": rejected,
                "chosen_completion_mask": [0, 0, 0] + [1] * (len(chosen) - 3),
                "rejected_completion_mask": [0, 0, 0] + [1] * (len(rejected) - 3),
            }
        )
    return Dataset.from_list(rows)


def _dpo_trainer(tmp_path, loss_type, ref_model=None, **overrides):
    return DPOTrainer(
        model=_tiny_mellum(),
        ref_model=ref_model,
        args=DPOConfig(
            output_dir=str(tmp_path),
            privacy_noise_multiplier=1.0,
            clipping_norm=1.0,
            per_device_train_batch_size=4,
            max_steps=2,
            max_length=8,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            seed=0,
            loss_type=loss_type,
            router_load_release="monitor",
            **overrides,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )


@pytest.mark.parametrize(
    ("loss_type", "fused", "kwargs"),
    [
        ("simpo", True, {"log_completion_metrics": False}),
        ("sigmoid", False, {"log_completion_metrics": True}),
    ],
)
def test_dpo_pools_chosen_and_rejected_and_never_touches_the_reference(
    tmp_path, loss_type, fused, kwargs
):
    ref_model = _tiny_mellum(seed=1) if loss_type == "sigmoid" else None
    seen_ref: list[dict] = []
    if ref_model is not None:
        ref_model.model.register_forward_pre_hook(
            lambda m, a, kw: seen_ref.append(dict(kw)), with_kwargs=True
        )
    trainer = _dpo_trainer(tmp_path, loss_type, ref_model=ref_model, **kwargs)
    assert (trainer._use_fused_logp and not trainer._log_completion_metrics) is fused
    if ref_model is not None:
        # The reference forward (precomputed outside the gradient transform,
        # possibly served from the dataset cache) never records router logits.
        ref_batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        trainer.compute_ref_log_probs(ref_batch, ref_model.eval(), null_ref=False)
        assert seen_ref, "the reference forward ran"
        assert not any(kw.get("output_router_logits") for kw in seen_ref)
    seen_policy: list[dict] = []
    trainer.model.model.register_forward_pre_hook(
        lambda m, a, kw: seen_policy.append(dict(kw)), with_kwargs=True
    )
    captured: dict = {}
    original = trainer._apply_router_load_terms

    def spy(loss, router_logits, mask, params, **kw):
        captured["router_logits"] = router_logits
        captured["mask"] = mask
        return original(loss, router_logits, mask, params, **kw)

    trainer._apply_router_load_terms = spy
    with _active_context(trainer) as ctx:
        rt = trainer._router_load
        assert (rt.mean_tokens, rt.max_tokens, rt.row_max_tokens) == (16.0, 16.0, 8)
        batch = trainer._prepare_input(
            trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
        )
        batch = trainer._augment_inputs(batch)
        merged = {**ctx.frozen_params, **ctx.trainable_params}
        example = {k: v[0] for k, v in batch.items()}
        loss, aux = trainer.compute_per_example_loss_and_metrics(
            ctx.fmodel, merged, example
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert all(kw.get("output_router_logits") for kw in seen_policy)
        # The pooled statistics equal the token-weighted pooling of the sides.
        pooled_h, pooled_p, pooled_t = router_load_and_probs(
            captured["router_logits"],
            captured["mask"],
            top_k=TOP_K,
            num_layers=NUM_LAYERS,
        )
        sides = []
        for side in ("chosen", "rejected"):
            _hidden, logits = trainer._last_hidden_state(
                merged,
                example[f"{side}_input_ids"],
                example[f"{side}_attention_mask"],
                output_router_logits=True,
            )
            sides.append(
                router_load_and_probs(
                    logits,
                    example[f"{side}_attention_mask"],
                    top_k=TOP_K,
                    num_layers=NUM_LAYERS,
                )
            )
        (h_c, p_c, t_c), (h_r, p_r, t_r) = sides
        assert pooled_t == t_c + t_r
        torch.testing.assert_close(
            pooled_h, (t_c * h_c + t_r * h_r) / (t_c + t_r), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            pooled_p, (t_c * p_c + t_r * p_r) / (t_c + t_r), atol=1e-6, rtol=1e-5
        )
        # The probe leaf of the batch is (lam / B) sum_x w_x (h_x - k/E) with
        # w_x = (T_c + T_r) / T_bar_pair, and never clips.
        grads, aux = _clipped(ctx, batch)
        expected = torch.zeros(NUM_LAYERS, NUM_EXPERTS)
        for i in range(4):
            ex = {k: v[i] for k, v in batch.items()}
            parts = []
            for side in ("chosen", "rejected"):
                _hidden, logits = trainer._last_hidden_state(
                    merged,
                    ex[f"{side}_input_ids"],
                    ex[f"{side}_attention_mask"],
                    output_router_logits=True,
                )
                parts.append(
                    router_load_and_probs(
                        logits,
                        ex[f"{side}_attention_mask"],
                        top_k=TOP_K,
                        num_layers=NUM_LAYERS,
                    )
                )
            (h_c, _, t_c), (h_r, _, t_r) = parts
            h = (t_c * h_c + t_r * h_r) / (t_c + t_r)
            expected += (
                rt.lam * ((t_c + t_r) / rt.mean_tokens) * (h - TOP_K / NUM_EXPERTS)
            )
        expected /= trainer.args.per_device_train_batch_size
        torch.testing.assert_close(
            grads.pytree[PROBE_NAME], expected, atol=1e-6, rtol=1e-5
        )
        bound = (
            grads.max_norm.values[PROBE_NAME] * trainer.args.per_device_train_batch_size
        )
        assert (aux.group_norms[PROBE_NAME] <= bound).all()
        metrics = trainer.training_step(trainer.model, batch)
        assert PROBE_NAME not in metrics["group_metrics"]
        assert "router_load/D" in metrics


def test_dpo_release_trains_end_to_end(tmp_path):
    trainer = _dpo_trainer(tmp_path, "simpo", log_completion_metrics=False)
    recorder = _Recorder(trainer)
    trainer.add_callback(recorder)
    out = trainer.train()
    assert out.global_step == 2
    assert all(not p.any() for p in recorder.probes)


# ---------------------------------------------------------------------------
# DP-FTRL: the two-group latch under band-MF
# ---------------------------------------------------------------------------


def test_band_mf_release_uses_the_strategy_filter(tmp_path):
    trainer = _trainer(
        tmp_path,
        router_load_release="monitor",
        privacy_noise_mechanism="mf_band",
        privacy_noise_mechanism_kwargs={"bands": 2},
        max_steps=8,
        optim="sgd",
    )
    recorder = _Recorder(trainer)
    trainer.add_callback(recorder)
    with _active_context(trainer) as ctx:
        state = trainer._router_load.state
        reference = moe_load.filter_factors(
            ctx.mf.strategy,
            n_steps=ctx.total_steps,
            kind="ema",
            beta=0.99,
            window=256,
            num_experts=NUM_EXPERTS,
        )
        assert torch.equal(state.phi, reference)
        iid = moe_load.filter_factors(
            None,
            n_steps=ctx.total_steps,
            kind="ema",
            beta=0.99,
            window=256,
            num_experts=NUM_EXPERTS,
        )
        assert not torch.equal(state.phi, iid)
        assert isinstance(ctx.clip_norm, PerGroup)
        assert PROBE_NAME in ctx.clip_norm.values
    out = trainer.train()
    assert out.global_step == 8
    assert all(not p.any() for p in recorder.probes)


@pytest.mark.slow
def test_torch_compile_step_matches_eager(tmp_path):
    eager = _trainer(
        tmp_path / "eager",
        router_load_release="surrogate",
        privacy_noise_multiplier=0.0,
    )
    compiled = _trainer(
        tmp_path / "compiled",
        router_load_release="surrogate",
        privacy_noise_multiplier=0.0,
        torch_compile=True,
    )
    with _active_context(eager) as ctx_e, _active_context(compiled) as ctx_c:
        batch_e, batch_c = _batch(eager), _batch(compiled)
        eager._augment_inputs(batch_e)
        compiled._augment_inputs(batch_c)
        grads_e, _ = _clipped(ctx_e, batch_e)
        grads_c, _ = _clipped(ctx_c, batch_c)
        for name, leaf in grads_e.pytree.items():
            torch.testing.assert_close(grads_c.pytree[name], leaf, atol=1e-4, rtol=1e-3)
