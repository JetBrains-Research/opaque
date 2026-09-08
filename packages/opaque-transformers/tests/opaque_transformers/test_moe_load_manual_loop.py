# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""T26: the helper round-trips through a tiny manual DP-FTRL loop.

The loop is shaped like ``examples/train_dpftrl.py`` (b-min-sep sampler,
band-MF correlated noise, functional optimizer) and calls the helper at its
four seams: probe attachment before ``make_functional``, ``probe_bounds`` for
the clipping bound, the post-processing between ``noise_fn`` and the
optimizer update, and checkpointing of the post-processing state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchopt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchopt
from _moe_load_shared import (
    NUM_EXPERTS,
    NUM_LAYERS,
    T_MAX,
    TOP_K,
    build_tiny_mellum,
    make_data,
    make_loss_fn,
)

from opaque.api.transformers.moe_load import (
    PROBE_NAME,
    filter_factors,
    initial_state,
    probe_bounds,
    summary,
    telemetry_without_probe,
    update,
)
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.dpftrl.sampling import BMinSepSampler
from opaque.dpsgd.clipping import clipped_grad
from opaque.optimizers import sgd
from opaque.random import key as rng_key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup

torch.set_num_threads(2)

N_TRAIN = 64
B_BAR = 8.0
N_STEPS = 4
BANDS = 2
C_G = 0.9
RATIO = 0.05
NM = 0.5
BETA = 0.9
ALPHA = 1e-3


def _run_loop(*, n_loop_steps: int, resume_at: int | None = None):
    """Run the loop; optionally save/restore the helper state at ``resume_at``."""
    # Seam 1: the probe is attached inside build_tiny_mellum before
    # make_functional(partition_trainable=True).
    _, _, fmodel, trainable, frozen = build_tiny_mellum(seed=0, router_scale=3.0)
    params = {k: v.clone() for k, v in trainable.items()}
    ids, mask, labels = make_data(N_TRAIN, seed=11)

    # Seam 2: the two-group clipping bound replaces the scalar clip norm.
    max_norm, lam = probe_bounds(
        C_G,
        params,
        ratio=RATIO,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        mean_tokens=float(T_MAX),
        max_tokens=float(T_MAX),
    )
    ctx = {
        "f_tilde": torch.full((NUM_EXPERTS,), TOP_K / NUM_EXPERTS),
        "alpha": ALPHA,
        "lam": lam,
        "mean_tokens": float(T_MAX),
    }
    loss_fn = make_loss_fn(fmodel, frozen, ctx)
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        has_aux=True,
        clipping_norm=max_norm,
        normalize_by=B_BAR,
        batch_argnums=(1, 2, 3),
        return_aux=True,
    )
    strategy = band_mf_strategy(bands=BANDS, momentum=0.95)
    noise_fn, noise_state = mf_gaussian_noise(
        params, strategy, n_steps=N_STEPS, noise_multiplier=NM, key=rng_key(5)
    )
    sampler = BMinSepSampler(
        ids, bands=BANDS, sampling_prob=B_BAR / N_TRAIN, n_steps=N_STEPS, key=rng_key(9)
    )
    per_step = PerGroup(
        max_norm.groups, {g: v / B_BAR for g, v in max_norm.values.items()}
    )
    phi = filter_factors(
        strategy,
        n_steps=N_STEPS,
        kind="ema",
        beta=BETA,
        window=1,
        num_experts=NUM_EXPERTS,
    )
    state = initial_state(
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ratio=RATIO,
        lam=lam,
        max_norm=per_step,
        noise_multiplier=NM,
        kind="ema",
        beta=BETA,
        alpha=ALPHA,
        mean_tokens=float(T_MAX),
        max_tokens=float(T_MAX),
        phi=phi,
    )
    template = state
    optimizer = sgd(lr=0.01)
    opt_state = optimizer.init(params)
    row_l2 = (
        strategy.streaming_matrix(n_steps=N_STEPS).row_norms_squared(N_STEPS).sqrt()
    )
    ce_only = make_loss_fn(fmodel, frozen, {**ctx, "alpha": 0.0, "lam": 0.0})

    trace = []
    for step, batch in enumerate(sampler):
        if step >= n_loop_steps:
            break
        idx = torch.as_tensor(list(batch), dtype=torch.long)
        b_ids, b_mask, b_labels = ids[idx], mask[idx], labels[idx]
        # Public constant f_tilde_t for this step; the probe must be zero.
        ctx["f_tilde"] = state.f_tilde
        assert torch.count_nonzero(params[PROBE_NAME]) == 0
        (grads, aux), clip_state = grad_fn(
            params, b_ids, b_mask, b_labels, state=clip_state
        )
        noisy, noise_state = noise_fn(grads, noise_state)

        # Seam 3: read the noised probe leaf, post-process, zero the leaf.
        probe_leaf = noisy.pytree[PROBE_NAME].clone()
        state = update(state, probe_leaf)
        noisy.pytree[PROBE_NAME].zero_()
        if resume_at is not None and step == resume_at:
            # Seam 4: checkpoint round-trip of the post-processing state.
            state = from_state_dict(template, state_dict(state))

        # The CE-only reference goes through the same vmap(grad_and_value)
        # transform the clipper uses, so kernel selection matches bit for bit.
        # It runs before the (in-place) optimizer update on the same params.
        _, (ce_values, _) = torch.func.vmap(
            torch.func.grad_and_value(ce_only, has_aux=True), in_dims=(None, 0, 0, 0)
        )(params, b_ids, b_mask, b_labels)

        updates, opt_state = optimizer.update(noisy, opt_state, params=params)
        params = torchopt.apply_updates(params, updates)

        grad_norm, clipped_norm = telemetry_without_probe(aux)
        trace.append(
            {
                "batch_size": int(aux.batch_size),
                "probe_norm_max": float(aux.group_norms[PROBE_NAME].max()),
                "probe_clip_events": int(
                    (aux.group_norms[PROBE_NAME] > max_norm.values[PROBE_NAME]).sum()
                ),
                "loss_equals_ce": torch.equal(aux.loss_values, ce_values),
                "probe_param_nonzero": int(torch.count_nonzero(params[PROBE_NAME])),
                "sigma_probe": float(noisy.noise_stddev.values[PROBE_NAME]),
                "row_l2": float(row_l2[step]),
                "f_tilde": state.f_tilde.clone(),
                "summary": summary(state),
                "grad_norm_mean": float(grad_norm.mean()),
                "clipped_norm_mean": float(clipped_norm.mean()),
            }
        )
    return trace, params, max_norm, per_step


def test_manual_dpftrl_loop_round_trips_the_helper():
    trace, params, max_norm, per_step = _run_loop(n_loop_steps=3)
    assert len(trace) == 3
    from opaque.api.engine.noise_allocation import per_group_noise_stddev

    base = per_group_noise_stddev(per_step, NM).values[PROBE_NAME]
    for row in trace:
        assert row["batch_size"] > 0
        # The probe group is a structural bound: never clipped.
        assert row["probe_norm_max"] <= max_norm.values[PROBE_NAME]
        assert row["probe_clip_events"] == 0
        # Value-neutral augmentation: the logged loss is exactly the CE.
        assert row["loss_equals_ce"]
        # The probe parameter never moves.
        assert row["probe_param_nonzero"] == 0
        # Realised MF sigma is base * ||row_t(C^-1)|| on the probe group.
        assert row["sigma_probe"] == pytest.approx(base * row["row_l2"], rel=1e-6)
        f = row["f_tilde"]
        assert bool(((f >= 0) & (f <= 1)).all())
        assert float(f.sum()) == pytest.approx(TOP_K, abs=1e-4)
        s = row["summary"]
        assert s["router_load/noise_std"] > 0
        assert s["router_load/tripped"] == 0.0
        assert row["grad_norm_mean"] > 0
        assert row["clipped_norm_mean"] <= row["grad_norm_mean"] + 1e-6
    assert torch.count_nonzero(params[PROBE_NAME]) == 0


def test_checkpoint_round_trip_reproduces_f_tilde():
    plain, _, _, _ = _run_loop(n_loop_steps=3)
    resumed, _, _, _ = _run_loop(n_loop_steps=3, resume_at=1)
    for a, b in zip(plain, resumed, strict=True):
        assert torch.equal(a["f_tilde"], b["f_tilde"])
        assert a["summary"] == b["summary"]
