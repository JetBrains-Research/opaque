# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Behaviour tests for the trainer-independent router-load helper.

Covers the surrogate identity against HF's load-balancing loss (T3), the
two-group clipping bound (T7), the pre-noise probe leaf through the real
clipper (T8), the post-processing closed forms (T9), the MF noise path and
filter factors (T14) and accountant invariance (T15).
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _moe_load_shared import (
    NUM_EXPERTS,
    NUM_LAYERS,
    T_MAX,
    TOP_K,
    build_tiny_mellum,
    ensure_moe_stats,
    make_data,
    make_loss_fn,
    token_weighted_load,
)

import opaque.dpftrl.accounting as dpftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.api.transformers.moe_load import (
    PROBE_NAME,
    RouterLoadState,
    attach_probe,
    filter_factors,
    initial_state,
    load_bound,
    probe_bounds,
    router_load_terms,
    summary,
    telemetry_without_probe,
    update,
)
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.dpsgd.noise import gaussian_noise
from opaque.exceptions import ConfigurationError
from opaque.random import key as rng_key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup, clipped

torch.set_num_threads(2)

SHARE = TOP_K / NUM_EXPERTS
DELTA_L = math.sqrt(TOP_K * NUM_LAYERS * (1 - TOP_K / NUM_EXPERTS))
B_BAR = 8.0
C_G = 0.9
RATIO = 0.05
GUARD = 1e-3


def _bounds(trainable, *, ratio=RATIO, mean_tokens=T_MAX, clipping_norm=C_G):
    return probe_bounds(
        clipping_norm,
        trainable,
        ratio=ratio,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        mean_tokens=mean_tokens,
        max_tokens=T_MAX,
        guard=GUARD,
    )


def _per_step(max_norm: PerGroup, divisor: float = B_BAR) -> PerGroup:
    return PerGroup(
        max_norm.groups, {g: v / divisor for g, v in max_norm.values.items()}
    )


def _flat(grads: dict, skip=(PROBE_NAME,)) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for k, v in sorted(grads.items()) if k not in skip])


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_mellum(seed=0, router_scale=3.0)


@pytest.fixture(scope="module")
def tiny64():
    return build_tiny_mellum(seed=0, router_scale=3.0, dtype=torch.float64)


# ===========================================================================
# T7: setup
# ===========================================================================


class TestProbeBounds:
    def test_probe_in_trainable_and_attach_idempotent(self, tiny):
        wrapper, _, _, trainable, _ = tiny
        assert tuple(trainable[PROBE_NAME].shape) == (NUM_LAYERS, NUM_EXPERTS)
        assert trainable[PROBE_NAME].dtype == torch.float32
        assert torch.count_nonzero(trainable[PROBE_NAME]) == 0
        name = attach_probe(wrapper, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS)
        assert name == PROBE_NAME
        assert sum(1 for n, _ in wrapper.named_parameters() if n == PROBE_NAME) == 1
        with pytest.raises(ConfigurationError):
            attach_probe(wrapper, num_layers=NUM_LAYERS + 1, num_experts=NUM_EXPERTS)

    def test_attach_on_hf_model_lands_in_trainable(self):
        from opaque.functional import make_functional

        wrapper, _, _, _, _ = build_tiny_mellum(seed=1)
        model = wrapper.lm
        attach_probe(model, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS)
        _, trainable, _ = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )
        assert PROBE_NAME in trainable
        assert tuple(trainable[PROBE_NAME].shape) == (NUM_LAYERS, NUM_EXPERTS)

    def test_scalar_clipping_norm(self, tiny):
        _, _, _, trainable, _ = tiny
        max_norm, lam = _bounds(trainable)
        c_h = RATIO * C_G * (1 + GUARD)
        assert set(max_norm.values) == {"fallback", PROBE_NAME}
        assert max_norm.values["fallback"] == C_G
        assert max_norm.values[PROBE_NAME] == pytest.approx(c_h, rel=1e-12)
        assert lam == pytest.approx(RATIO * C_G / DELTA_L, rel=1e-12)
        assert max_norm.values[PROBE_NAME] == pytest.approx(
            lam * DELTA_L * (1 + GUARD), rel=1e-12
        )
        assert max_norm.groups[(PROBE_NAME,)] == PROBE_NAME
        for k in trainable:
            if k != PROBE_NAME:
                assert max_norm.groups[(k,)] == "fallback"
        assert set(max_norm.groups) == {(k,) for k in trainable}

    def test_mean_tokens_scales_the_bound(self, tiny):
        _, _, _, trainable, _ = tiny
        _, lam_full = _bounds(trainable, mean_tokens=T_MAX)
        _, lam_half = _bounds(trainable, mean_tokens=T_MAX / 2)
        assert lam_half == pytest.approx(lam_full / 2, rel=1e-12)
        assert load_bound(
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            mean_tokens=T_MAX / 2,
            max_tokens=T_MAX,
        ) == pytest.approx(2 * DELTA_L)

    def test_dict_clipping_norm_uses_fallback_as_c_g(self, tiny):
        _, _, _, trainable, _ = tiny
        max_norm, lam = _bounds(
            trainable, clipping_norm={"q_proj": 1.5, "fallback": 0.5}
        )
        assert max_norm.values["q_proj"] == 1.5
        assert max_norm.values["fallback"] == 0.5
        assert max_norm.values[PROBE_NAME] == pytest.approx(RATIO * 0.5 * (1 + GUARD))
        assert lam == pytest.approx(RATIO * 0.5 / DELTA_L)

    def test_dict_without_fallback_uses_largest_group(self, tiny):
        _, _, _, trainable, _ = tiny
        max_norm, lam = _bounds(
            trainable, clipping_norm={"self_attn": 0.4, "mlp.gate": 1.2}
        )
        assert max_norm.values[PROBE_NAME] == pytest.approx(RATIO * 1.2 * (1 + GUARD))
        assert lam == pytest.approx(RATIO * 1.2 / DELTA_L)

    def test_user_router_pattern_coexists_with_probe(self):
        trainable = {
            "layers.0.mlp.router.weight": torch.zeros(4, 3),
            "layers.0.self_attn.q_proj.weight": torch.zeros(3, 3),
            PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS),
        }
        # Substring matching would put the probe into the user's "router" group.
        collided = per_group(trainable, router=0.3, fallback=1.0)
        assert collided.groups[(PROBE_NAME,)] == "router"
        # Direct construction keeps them apart.
        max_norm, _ = _bounds(trainable, clipping_norm={"router": 0.3, "fallback": 1.0})
        assert max_norm.groups[("layers.0.mlp.router.weight",)] == "router"
        assert max_norm.groups[(PROBE_NAME,)] == PROBE_NAME
        assert max_norm.values["router"] == 0.3
        assert max_norm.values[PROBE_NAME] == pytest.approx(RATIO * 1.0 * (1 + GUARD))

    def test_pergroup_clipping_norm(self, tiny):
        _, _, _, trainable, _ = tiny
        non_probe = {k: v for k, v in trainable.items() if k != PROBE_NAME}
        user = per_group(non_probe, q_proj=0.7, fallback=C_G)
        max_norm, _ = _bounds(trainable, clipping_norm=user)
        assert max_norm.values["q_proj"] == 0.7
        assert max_norm.groups[(PROBE_NAME,)] == PROBE_NAME
        with pytest.raises(ConfigurationError):
            _bounds(trainable, clipping_norm=per_group(trainable, fallback=C_G))
        partial = per_group(dict(list(non_probe.items())[:2]), fallback=C_G)
        with pytest.raises(ConfigurationError):
            _bounds(trainable, clipping_norm=partial)

    def test_invalid_configuration_raises(self, tiny):
        _, _, _, trainable, _ = tiny
        with pytest.raises(ConfigurationError):
            _bounds(trainable, ratio=0.0)
        with pytest.raises(ConfigurationError):
            _bounds({k: v for k, v in trainable.items() if k != PROBE_NAME})
        with pytest.raises(ConfigurationError):
            probe_bounds(
                C_G,
                trainable,
                ratio=RATIO,
                num_layers=NUM_LAYERS,
                num_experts=TOP_K,
                top_k=TOP_K,
                mean_tokens=T_MAX,
                max_tokens=T_MAX,
            )
        with pytest.raises(ConfigurationError):
            _bounds({PROBE_NAME: trainable[PROBE_NAME]})

    @pytest.mark.parametrize("ratio", [0.5, 0.1, 0.02])
    def test_sigmas_and_mahalanobis_identity(self, tiny, ratio):
        _, _, _, trainable, _ = tiny
        max_norm, _ = _bounds(trainable, ratio=ratio)
        nm = 0.7
        stored = _per_step(max_norm)
        sigmas = per_group_noise_stddev(stored, nm)
        c_g, c_h = max_norm.values["fallback"], max_norm.values[PROBE_NAME]
        s = c_g + c_h
        assert sigmas.values["fallback"] == pytest.approx(
            nm * math.sqrt(c_g * s) / B_BAR, rel=1e-12
        )
        assert sigmas.values[PROBE_NAME] == pytest.approx(
            nm * math.sqrt(c_h * s) / B_BAR, rel=1e-12
        )
        maha = (c_g / B_BAR) ** 2 / sigmas.values["fallback"] ** 2 + (
            c_h / B_BAR
        ) ** 2 / sigmas.values[PROBE_NAME] ** 2
        assert maha * nm**2 == pytest.approx(1.0, rel=1e-12)
        inflation = sigmas.values["fallback"] / (nm * c_g / B_BAR)
        assert inflation == pytest.approx(math.sqrt(1 + c_h / c_g), rel=1e-12)
        # The real noise engine publishes exactly these stddevs.
        zeros = {k: torch.zeros_like(v) for k, v in trainable.items()}
        noise_fn, state = gaussian_noise(noise_multiplier=nm, key=rng_key(1))
        noised, _ = noise_fn(clipped(zeros, max_norm=stored), state)
        assert noised.noise_stddev.values == sigmas.values


# ===========================================================================
# T3: surrogate identity
# ===========================================================================


class TestSurrogateIdentity:
    def _batch_stats(self, tiny64, params, ids, mask, labels):
        _, _, fmodel, _, frozen = tiny64
        stats = ensure_moe_stats()
        with torch.no_grad():
            _, router_logits = fmodel({**frozen, **params}, ids, mask, labels)
            n = ids.shape[0]
            per_layer = [z.reshape(n, -1, NUM_EXPERTS) for z in router_logits]
            h, counts = [], []
            for i in range(n):
                h_i, _, c_i = stats.router_load_and_probs(
                    [z[i] for z in per_layer],
                    mask[i],
                    top_k=TOP_K,
                    num_layers=NUM_LAYERS,
                )
                h.append(h_i)
                counts.append(c_i)
        return torch.stack(h), torch.stack(counts)

    @pytest.mark.parametrize("ragged", [True, False])
    def test_batch_mean_surrogate_is_collinear_with_hf_aux(self, tiny64, ragged):
        _, mod, fmodel, trainable, frozen = tiny64
        n = 6
        ids, mask, labels = make_data(n, seed=3, min_len=None if ragged else T_MAX)
        h, counts = self._batch_stats(tiny64, trainable, ids, mask, labels)
        f_batch = token_weighted_load(h, counts).double()
        t_tot = float(counts.sum())
        ctx = {
            "f_tilde": f_batch,
            "alpha": 1.0,
            "lam": 1.0,
            "mean_tokens": float(T_MAX),
        }
        surrogate_only = make_loss_fn(fmodel, frozen, ctx, include_ce=False)
        per_example = torch.func.vmap(
            torch.func.grad(surrogate_only, has_aux=True), in_dims=(None, 0, 0, 0)
        )
        grads, _ = per_example(trainable, ids, mask, labels)
        g_mean = _flat({k: v.mean(0) for k, v in grads.items()})
        non_probe = {k: v for k, v in trainable.items() if k != PROBE_NAME}

        def hf_aux(p):
            params = {**frozen, **p, PROBE_NAME: trainable[PROBE_NAME]}
            _, router_logits = fmodel(params, ids, mask, labels)
            return mod.load_balancing_loss_func(
                tuple(router_logits), NUM_EXPERTS, TOP_K, mask
            )

        g_hf = _flat(torch.func.grad(hf_aux)(non_probe))
        cosine = float(g_mean @ g_hf / (g_mean.norm() * g_hf.norm()))
        assert cosine >= 1 - 1e-12
        expected_scale = t_tot / (n * T_MAX)
        # HF's reference accumulates in float32 (``routing_weights.float()``),
        # so the scale is reproduced to float32 precision; the direction is
        # exact.
        assert float(g_mean.norm() / g_hf.norm()) == pytest.approx(
            expected_scale, rel=1e-6
        )
        if not ragged:
            assert expected_scale == 1.0
            assert torch.allclose(g_mean, g_hf, rtol=1e-5, atol=1e-8)

    def test_centred_and_uncentred_surrogates_match(self, tiny64):
        _, _, fmodel, trainable, frozen = tiny64
        ids, mask, labels = make_data(4, seed=5)
        f = torch.full(
            (NUM_EXPERTS,), SHARE, dtype=torch.float64
        ) + 0.05 * torch.arange(NUM_EXPERTS, dtype=torch.float64)
        grads = []
        for f_tilde in (f, f + 0.37):
            ctx = {
                "f_tilde": f_tilde,
                "alpha": 1.0,
                "lam": 1.0,
                "mean_tokens": float(T_MAX),
            }
            fn = make_loss_fn(fmodel, frozen, ctx, include_ce=False)
            g, _ = torch.func.vmap(
                torch.func.grad(fn, has_aux=True), in_dims=(None, 0, 0, 0)
            )(trainable, ids, mask, labels)
            grads.append(_flat({k: v.sum(0) for k, v in g.items()}))
        # The statistics run their softmax in fp32 (as the router does), so
        # the identity holds to fp32 precision on the float64 model.
        rel = float((grads[0] - grads[1]).norm() / grads[0].norm())
        assert rel < 1e-6

    def test_value_neutral_form_keeps_ce_value_and_adds_gradients(self, tiny64):
        _, _, fmodel, trainable, frozen = tiny64
        ids, mask, labels = make_data(4, seed=6)
        f = torch.full((NUM_EXPERTS,), SHARE, dtype=torch.float64)
        f[0] += 0.2
        f[1] -= 0.2
        alpha = 0.3
        ctx = {"f_tilde": f, "alpha": alpha, "lam": 0.5, "mean_tokens": float(T_MAX)}
        full = make_loss_fn(fmodel, frozen, ctx)
        ce_only = make_loss_fn(fmodel, frozen, {**ctx, "alpha": 0.0, "lam": 0.0})
        surr_only = make_loss_fn(
            fmodel, frozen, {**ctx, "alpha": 1.0}, include_ce=False
        )
        vg = torch.func.vmap(
            torch.func.grad_and_value(full, has_aux=True), in_dims=(None, 0, 0, 0)
        )
        g_full, (v_full, _) = vg(trainable, ids, mask, labels)
        with torch.no_grad():
            ce, _ = fmodel({**frozen, **trainable}, ids, mask, labels)
        assert torch.equal(v_full, ce)
        g_ce, _ = torch.func.vmap(
            torch.func.grad(ce_only, has_aux=True), in_dims=(None, 0, 0, 0)
        )(trainable, ids, mask, labels)
        g_s, _ = torch.func.vmap(
            torch.func.grad(surr_only, has_aux=True), in_dims=(None, 0, 0, 0)
        )(trainable, ids, mask, labels)
        for k in trainable:
            if k == PROBE_NAME:
                continue
            assert torch.allclose(
                g_full[k], g_ce[k] + alpha * g_s[k], rtol=1e-5, atol=1e-8
            ), k
        # Only the probe term differs between lam values: the CE-only run has
        # no probe gradient, the full run carries lam * w * d.
        assert torch.count_nonzero(g_ce[PROBE_NAME]) == 0
        assert torch.count_nonzero(g_full[PROBE_NAME]) > 0


# ===========================================================================
# T8: probe leaf through the real clipper
# ===========================================================================


class TestProbeLeaf:
    def test_pre_noise_leaf_matches_token_weighted_centred_load(self, tiny):
        _, _, fmodel, trainable, frozen = tiny
        ids, mask, labels = make_data(6, seed=8)
        max_norm, lam = _bounds(trainable)
        f = torch.full((NUM_EXPERTS,), SHARE)
        f[0] += 0.1
        f[2] -= 0.1
        ctx = {"f_tilde": f, "alpha": 1.0, "lam": lam, "mean_tokens": float(T_MAX)}
        gf, st = clipped_grad(
            make_loss_fn(fmodel, frozen, ctx),
            has_aux=True,
            clipping_norm=max_norm,
            normalize_by=B_BAR,
            batch_argnums=(1, 2, 3),
            return_aux=True,
        )
        (clipped_out, aux), _ = gf(trainable, ids, mask, labels, state=st)
        h, _, counts = aux.loss_aux
        w = counts / T_MAX
        expected = (lam / B_BAR) * (w[:, None, None] * (h - SHARE)).sum(0)
        leaf = clipped_out.pytree[PROBE_NAME]
        assert torch.allclose(leaf, expected, rtol=1e-6, atol=1e-8)
        assert abs(float(leaf.sum())) < 1e-6
        norms = aux.group_norms[PROBE_NAME]
        expected_norms = lam * w * (h - SHARE).reshape(len(w), -1).norm(dim=1)
        assert torch.allclose(norms, expected_norms, rtol=1e-5, atol=1e-7)
        assert bool((norms <= max_norm.values[PROBE_NAME]).all())
        assert clipped_out.max_norm.values[PROBE_NAME] == pytest.approx(
            max_norm.values[PROBE_NAME] / B_BAR
        )
        # Telemetry over the non-probe groups is the fallback group alone here.
        grad_norm, clipped_norm = telemetry_without_probe(aux)
        assert torch.allclose(grad_norm, aux.group_norms["fallback"], rtol=1e-6)
        assert torch.allclose(
            clipped_norm, torch.minimum(grad_norm, torch.tensor(C_G)), rtol=1e-5
        )


def _adversarial_leaf(probe_bound: str):
    """Run the real clipper on routing that sends every token to experts 0..k-1.

    Returns ``(clipped_output, aux, lam)``.  ``probe_bound="unbounded"`` lifts
    the probe bound to ``1e9`` so the two runs can be compared bit for bit.
    """
    n, t_len = 4, T_MAX
    mask = torch.ones(n, t_len, dtype=torch.long)
    mask[3] = 0  # a fully masked row contributes nothing
    logits = torch.zeros(n, NUM_LAYERS, t_len, NUM_EXPERTS)
    logits[..., :TOP_K] = 10.0
    logits = logits + 0.01 * torch.randn(
        logits.shape, generator=torch.Generator().manual_seed(0)
    )
    params = {"w": torch.ones(3), PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS)}
    max_norm, lam = _bounds(params)
    if probe_bound == "unbounded":
        max_norm = PerGroup(max_norm.groups, {**max_norm.values, PROBE_NAME: 1e9})

    def loss_fn(p, z, m):
        base = 1e-3 * (p["w"] * torch.tensor([1.0, 2.0, 3.0])).sum()
        return router_load_terms(
            base,
            tuple(z[i] for i in range(NUM_LAYERS)),
            m,
            p,
            f_tilde=torch.full((NUM_EXPERTS,), SHARE),
            alpha=0.0,
            lam=lam,
            top_k=TOP_K,
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            mean_tokens=float(T_MAX),
        )

    gf, st = clipped_grad(
        loss_fn,
        clipping_norm=max_norm,
        normalize_by=B_BAR,
        batch_argnums=(1, 2),
        return_aux=True,
    )
    (out, aux), _ = gf(params, logits, mask, state=st)
    return out, aux, lam


class TestAdversarialRouting:
    def test_adversarial_routing_never_clips(self):
        """Every token to the same k experts at T_max attains lam * Delta_L < C_h."""
        out, aux, lam = _adversarial_leaf("design")
        norms = aux.group_norms[PROBE_NAME]
        c_h = RATIO * C_G * (1 + GUARD)
        assert torch.allclose(norms[:3], torch.full((3,), lam * DELTA_L), rtol=1e-6)
        assert float(norms[3]) == 0.0
        assert bool((norms <= c_h).all())
        assert float((norms > c_h).float().mean()) == 0.0
        d_adv = torch.zeros(NUM_LAYERS, NUM_EXPERTS) - SHARE
        d_adv[:, :TOP_K] += 1.0
        expected = (lam / B_BAR) * 3 * d_adv
        assert torch.allclose(out.pytree[PROBE_NAME], expected, rtol=1e-6, atol=1e-8)

    def test_adversarial_scale_is_exactly_one(self):
        """The design bound and an unbounded probe give the bit-identical leaf."""
        design, _, _ = _adversarial_leaf("design")
        unbounded, _, _ = _adversarial_leaf("unbounded")
        assert torch.equal(design.pytree[PROBE_NAME], unbounded.pytree[PROBE_NAME])


# ===========================================================================
# T9: post-processing closed forms
# ===========================================================================


def _state(
    *, kind="ema", beta=0.9, window=3, nm=1.0, ratio=RATIO, dead_zone=2.0, shrink=True
):
    trainable = {"w": torch.zeros(5), PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS)}
    max_norm, lam = _bounds(trainable, ratio=ratio)
    phi = filter_factors(
        None, n_steps=64, kind=kind, beta=beta, window=window, num_experts=NUM_EXPERTS
    )
    state = initial_state(
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ratio=ratio,
        lam=lam,
        max_norm=_per_step(max_norm),
        noise_multiplier=nm,
        kind=kind,
        beta=beta,
        window=window,
        dead_zone=dead_zone,
        shrink=shrink,
        alpha=1e-3,
        mean_tokens=float(T_MAX),
        max_tokens=float(T_MAX),
        phi=phi,
    )
    sigma_h = per_group_noise_stddev(_per_step(max_norm), nm).values[PROBE_NAME]
    return state, lam, sigma_h


def _project(d: torch.Tensor) -> torch.Tensor:
    return d - d.mean(-1, keepdim=True)


class TestPostProcessing:
    def test_initial_state(self):
        state, lam, sigma_h = _state()
        assert torch.equal(state.f_tilde, torch.full((NUM_EXPERTS,), SHARE))
        assert state.step == 0
        assert state.noise_std == 0.0
        assert state.base_noise_std == pytest.approx(
            sigma_h / (lam * math.sqrt(NUM_LAYERS))
        )
        s = summary(state)
        assert s["router_load/D"] == 0.0
        assert s["router_load/dead_zone"] == 1.0
        assert s["router_load/f_min"] == pytest.approx(SHARE)
        assert s["router_load/entropy"] == pytest.approx(math.log(NUM_EXPERTS))

    def test_first_release_is_the_projected_shrunk_release(self):
        state, lam, _ = _state(beta=0.9, nm=0.05)
        g = torch.Generator().manual_seed(1)
        d_signal = 0.1 * torch.randn(NUM_LAYERS, NUM_EXPERTS, generator=g)
        y = lam * d_signal
        new = update(state, y)
        pooled = _project(d_signal.mean(0))
        assert new.step == 1
        # Bias correction: d_tilde_1 = m_1 / (1 - beta) = pooled release itself.
        assert torch.allclose(new.m / (1 - 0.9), pooled, atol=1e-6)
        base = state.base_noise_std
        s1 = base * (1 - 0.9) * math.sqrt((NUM_EXPERTS - 1) / NUM_EXPERTS) / (1 - 0.9)
        assert new.noise_std == pytest.approx(s1, rel=1e-6)
        n2 = float(pooled.pow(2).sum())
        assert n2 >= 2.0 * NUM_EXPERTS * s1**2
        factor = 1 - NUM_EXPERTS * s1**2 / n2
        expected = torch.clamp(SHARE + pooled * factor, 0.0, 1.0)
        assert torch.allclose(new.f_tilde, expected, atol=1e-6)
        assert float(new.f_tilde.sum()) == pytest.approx(TOP_K, abs=1e-5)
        info = summary(new)
        assert info["router_load/shrink"] == pytest.approx(factor, rel=1e-6)
        assert info["router_load/dead_zone"] == 0.0
        assert info["router_load/D"] == pytest.approx(
            float(pooled.abs().max() / SHARE), rel=1e-5
        )
        layers = _project(d_signal)
        assert info["router_load/D_layer_max"] == pytest.approx(
            float(layers.abs().max() / SHARE), rel=1e-5
        )

    def test_ema_matches_closed_form_over_steps(self):
        beta = 0.9
        state, lam, _ = _state(beta=beta)
        g = torch.Generator().manual_seed(2)
        m_ref = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
        phi2 = 0.0
        for t in range(1, 8):
            d = 0.2 * torch.randn(NUM_LAYERS, NUM_EXPERTS, generator=g)
            state = update(state, lam * d)
            pooled = _project(d.mean(0)).double()
            m_ref = beta * m_ref + (1 - beta) * pooled
            phi2 = beta**2 * phi2 + (1 - beta) ** 2 * (NUM_EXPERTS - 1) / NUM_EXPERTS
            corr = 1 - beta**t
            assert torch.allclose(state.m.double(), m_ref, atol=1e-6)
            assert state.noise_std == pytest.approx(
                state.base_noise_std * math.sqrt(phi2) / corr, rel=1e-6
            )
            d_tilde = m_ref / corr
            n2 = float(d_tilde.pow(2).sum())
            s2 = state.noise_std**2
            if n2 < 2.0 * NUM_EXPERTS * s2:
                expected = torch.full((NUM_EXPERTS,), SHARE, dtype=torch.float64)
            else:
                expected = SHARE + d_tilde * (1 - NUM_EXPERTS * s2 / n2)
            assert torch.allclose(
                state.f_tilde.double(), expected.clamp(0, 1), atol=1e-6
            )

    def test_dead_zone_zeroes_and_boundary_factor_is_half(self):
        state, lam, _ = _state(beta=0.5, nm=0.05)
        # After one release d_tilde equals the projected pooled release; the
        # noise std is base * sqrt((E-1)/E).
        s1 = state.base_noise_std * math.sqrt((NUM_EXPERTS - 1) / NUM_EXPERTS)
        direction = torch.tensor([1.0, -1.0] + [0.0] * (NUM_EXPERTS - 2))
        direction = direction / direction.norm()
        for scale, inside in ((0.5, True), (1.5, False)):
            norm = scale * math.sqrt(2.0 * NUM_EXPERTS) * s1
            d = (norm * direction)[None].expand(NUM_LAYERS, -1)
            new = update(state, lam * d)
            info = summary(new)
            if inside:
                assert torch.equal(new.f_tilde, torch.full((NUM_EXPERTS,), SHARE))
                assert info["router_load/dead_zone"] == 1.0
                assert info["router_load/shrink"] == 0.0
            else:
                # ||d||^2 = scale^2 * 2 E s^2  ->  factor = 1 - 1 / (2 scale^2)
                factor = 1 - 1 / (2 * scale**2)
                assert factor >= 0.5
                assert info["router_load/shrink"] == pytest.approx(factor, rel=1e-5)
                assert torch.allclose(
                    new.f_tilde, SHARE + norm * direction * factor, atol=1e-6
                )
        # Exactly at the boundary the factor is 1 - 1/c = 1/2.
        norm = math.sqrt(2.0 * NUM_EXPERTS) * s1 * (1 + 1e-6)
        new = update(state, lam * (norm * direction)[None].expand(NUM_LAYERS, -1))
        assert summary(new)["router_load/shrink"] == pytest.approx(0.5, abs=1e-4)

    def test_clamp_and_shrink_off(self):
        state, lam, _ = _state(beta=0.5, shrink=False)
        d = torch.zeros(NUM_LAYERS, NUM_EXPERTS)
        d[:, 0] = 5.0
        d[:, 1] = -5.0
        new = update(state, lam * d)
        assert float(new.f_tilde[0]) == 1.0
        assert float(new.f_tilde[1]) == 0.0
        assert bool(((new.f_tilde >= 0) & (new.f_tilde <= 1)).all())
        # Plain estimate: no dead zone, no shrinkage.
        tiny_d = torch.full((NUM_LAYERS, NUM_EXPERTS), 0.0)
        tiny_d[:, 0] = 1e-4
        tiny_d[:, 1] = -1e-4
        new = update(state, lam * tiny_d)
        assert float(new.f_tilde[0]) == pytest.approx(SHARE + 1e-4, abs=1e-7)
        assert summary(new)["router_load/shrink"] == 1.0

    def test_window_mean(self):
        window = 3
        state, lam, _ = _state(kind="window", window=window, shrink=False)
        g = torch.Generator().manual_seed(4)
        releases = []
        for t in range(1, 6):
            d = 0.05 * torch.randn(NUM_LAYERS, NUM_EXPERTS, generator=g)
            releases.append(_project(d.mean(0)))
            state = update(state, lam * d)
            recent = torch.stack(releases[-window:])
            assert torch.allclose(state.m / min(t, window), recent.mean(0), atol=1e-6)
            assert torch.allclose(state.f_tilde, SHARE + recent.mean(0), atol=1e-6)
            expected_std = state.base_noise_std * math.sqrt(
                (NUM_EXPERTS - 1) / (NUM_EXPERTS * min(t, window))
            )
            assert state.noise_std == pytest.approx(expected_std, rel=1e-6)

    def test_masked_release_keeps_balance(self):
        state, _, _ = _state()
        new = update(state, torch.zeros(NUM_LAYERS, NUM_EXPERTS))
        assert torch.equal(new.f_tilde, torch.full((NUM_EXPERTS,), SHARE))
        with pytest.raises(ConfigurationError):
            update(state, torch.zeros(NUM_LAYERS + 1, NUM_EXPERTS))

    @pytest.mark.parametrize("kind", ["ema", "window"])
    def test_state_round_trips_through_serialization(self, kind):
        state, lam, _ = _state(kind=kind, beta=0.8, window=2)
        g = torch.Generator().manual_seed(5)
        for _ in range(3):
            state = update(
                state, lam * 0.2 * torch.randn(NUM_LAYERS, NUM_EXPERTS, generator=g)
            )
        template, _, _ = _state(kind=kind, beta=0.8, window=2)
        restored = from_state_dict(template, state_dict(state))
        assert isinstance(restored, RouterLoadState)
        for field in RouterLoadState.__dataclass_fields__:
            a, b = getattr(state, field), getattr(restored, field)
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), field
            else:
                assert a == b, field
        d = 0.1 * torch.randn(NUM_LAYERS, NUM_EXPERTS, generator=g)
        assert torch.equal(
            update(state, lam * d).f_tilde, update(restored, lam * d).f_tilde
        )


# ===========================================================================
# T14: MF noise and filter factors
# ===========================================================================


def _dense_phi(strategy, n, *, kind, beta, window):
    coef = strategy.coefficients(n_steps=n).double().numpy()
    import numpy as np

    c = np.zeros(n)
    c[: min(n, len(coef))] = coef[:n]
    C = np.zeros((n, n))
    F = np.zeros((n, n))
    if kind == "ema":
        f = (1 - beta) * beta ** np.arange(n)
    else:
        f = np.zeros(n)
        f[:window] = 1.0
    for i in range(n):
        C[i:, i] = c[: n - i]
        F[i:, i] = f[: n - i]
    FC = F @ np.linalg.inv(C)
    return torch.from_numpy(
        np.linalg.norm(FC, axis=1) * math.sqrt((NUM_EXPERTS - 1) / NUM_EXPERTS)
    )


class TestMatrixFactorization:
    @pytest.mark.parametrize(
        ("kind", "beta", "window"),
        [("ema", 0.99, 256), ("ema", 0.9, 256), ("window", 0.99, 8)],
    )
    def test_filter_factors_match_dense_reference(self, kind, beta, window):
        strategy = band_mf_strategy(bands=4, momentum=0.95)
        for n in (16, 64):
            phi = filter_factors(
                strategy,
                n_steps=n,
                kind=kind,
                beta=beta,
                window=window,
                num_experts=NUM_EXPERTS,
            )
            assert phi.shape == (n,)
            assert torch.allclose(
                phi,
                _dense_phi(strategy, n, kind=kind, beta=beta, window=window),
                rtol=1e-8,
                atol=1e-12,
            )

    def test_dpsgd_factors_are_the_closed_form_recursion(self):
        beta = 0.99
        with pytest.warns(UserWarning, match="not stabilised"):
            phi = filter_factors(
                None,
                n_steps=300,
                kind="ema",
                beta=beta,
                window=1,
                num_experts=NUM_EXPERTS,
                n_phi=128,
            )
        assert phi.shape == (128,)
        phi2 = 0.0
        for t in range(128):
            phi2 = beta**2 * phi2 + (1 - beta) ** 2 * (NUM_EXPERTS - 1) / NUM_EXPERTS
            assert float(phi[t]) == pytest.approx(math.sqrt(phi2), rel=1e-12)
        stationary = math.sqrt((1 - beta) / (1 + beta)) * math.sqrt(
            (NUM_EXPERTS - 1) / NUM_EXPERTS
        )
        phi_long = filter_factors(
            None, n_steps=5000, kind="ema", beta=beta, window=1, num_experts=NUM_EXPERTS
        )
        assert float(phi_long[-1]) == pytest.approx(stationary, rel=1e-6)
        window = filter_factors(
            None,
            n_steps=10,
            kind="window",
            beta=beta,
            window=4,
            num_experts=NUM_EXPERTS,
        )
        expected = torch.tensor(
            [
                math.sqrt(min(t, 4) * (NUM_EXPERTS - 1) / NUM_EXPERTS)
                for t in range(1, 11)
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(window, expected)
        # Band-MF anti-correlation beats iid noise for the same filter.
        mf = filter_factors(
            band_mf_strategy(bands=4, momentum=0.95),
            n_steps=300,
            kind="ema",
            beta=beta,
            window=1,
            num_experts=NUM_EXPERTS,
        )
        assert float(mf[-1]) < float(phi_long[-1])

    def test_filter_factors_rejects_non_toeplitz_strategy(self):
        from opaque.dpftrl.noise import lambda_cgd_strategy

        with pytest.raises(ConfigurationError):
            filter_factors(
                lambda_cgd_strategy(lambda_=0.5),
                n_steps=8,
                kind="ema",
                beta=0.9,
                window=1,
                num_experts=NUM_EXPERTS,
            )
        with pytest.raises(ConfigurationError):
            filter_factors(
                None,
                n_steps=8,
                kind="mean",
                beta=0.9,
                window=1,
                num_experts=NUM_EXPERTS,
            )

    def test_mf_noise_accepts_two_group_bound_and_publishes_row_scaled_sigma(self):
        n_steps = 8
        strategy = band_mf_strategy(bands=4, momentum=0.95)
        trainable = {
            "w": torch.zeros(3),
            PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS),
        }
        max_norm, _ = _bounds(trainable)
        stored = _per_step(max_norm)
        nm = 0.8
        base = per_group_noise_stddev(stored, nm)
        row_l2 = (
            strategy.streaming_matrix(n_steps=n_steps).row_norms_squared(n_steps).sqrt()
        )
        noise_fn, state = mf_gaussian_noise(
            trainable, strategy, n_steps=n_steps, noise_multiplier=nm, key=rng_key(7)
        )
        for t in range(n_steps):
            noised, state = noise_fn(clipped(trainable, max_norm=stored), state)
            for group in ("fallback", PROBE_NAME):
                assert noised.noise_stddev.values[group] == pytest.approx(
                    base.values[group] * float(row_l2[t]), rel=1e-6
                )
            assert torch.count_nonzero(noised.pytree[PROBE_NAME]) > 0
        assert state._first_max_norm == stored

    def test_monte_carlo_matches_filtered_noise_factor(self):
        n_steps, beta, n_keys = 6, 0.7, 128
        strategy = band_mf_strategy(bands=3, momentum=0.95)
        trainable = {
            "w": torch.zeros(1),
            PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS),
        }
        max_norm, lam = _bounds(trainable)
        stored = _per_step(max_norm)
        phi = filter_factors(
            strategy,
            n_steps=n_steps,
            kind="ema",
            beta=beta,
            window=1,
            num_experts=NUM_EXPERTS,
        )
        template = initial_state(
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            ratio=RATIO,
            lam=lam,
            max_norm=stored,
            noise_multiplier=1.0,
            kind="ema",
            beta=beta,
            alpha=0.0,
            mean_tokens=float(T_MAX),
            max_tokens=float(T_MAX),
            phi=phi,
            shrink=False,
        )
        samples = []
        for k in range(n_keys):
            noise_fn, nstate = mf_gaussian_noise(
                trainable,
                strategy,
                n_steps=n_steps,
                noise_multiplier=1.0,
                key=rng_key(100 + k),
            )
            state = template
            for _ in range(n_steps):
                noised, nstate = noise_fn(clipped(trainable, max_norm=stored), nstate)
                state = update(state, noised.pytree[PROBE_NAME])
            samples.append(state.m / (1 - beta**n_steps))
        empirical = torch.stack(samples).pow(2).mean().sqrt()
        assert float(empirical) == pytest.approx(state.noise_std, rel=0.10)


# ===========================================================================
# T15: accountant invariance
# ===========================================================================


class TestAccountingInvariance:
    def test_factories_take_no_per_group_information(self):
        for factory in (dpsgd_acc.gaussian, dpsgd_acc.poisson, dpftrl_acc.mf_gaussian):
            names = inspect.signature(factory).parameters
            assert not any(
                n in names for n in ("max_norm", "clipping_norm", "num_groups")
            )

    def test_dpsgd_epsilon_unchanged_by_the_probe(self):
        trainable = {
            "w": torch.zeros(3),
            PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS),
        }
        nm, q, steps = 1.1, 0.05, 20
        without = per_group({"w": trainable["w"]}, fallback=C_G)
        with_probe, _ = _bounds(trainable)
        eps = {}
        for label, max_norm in (("without", without), ("with", with_probe)):
            stored = _per_step(max_norm)
            noise_fn, state = gaussian_noise(noise_multiplier=nm, key=rng_key(0))
            zeros = {
                k: torch.zeros_like(v)
                for k, v in trainable.items()
                if (k,) in max_norm.groups
            }
            noised, _ = noise_fn(clipped(zeros, max_norm=stored), state)
            maha = sum(
                (stored.values[g] / noised.noise_stddev.values[g]) ** 2
                for g in stored.values
            )
            assert maha * nm**2 == pytest.approx(1.0, rel=1e-12)
            eps[label] = (
                dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q) * steps
            ).epsilon_at(1e-5)
        assert eps["with"] == eps["without"]

    def test_mf_epsilon_unchanged_by_the_probe(self):
        trainable = {
            "w": torch.zeros(3),
            PROBE_NAME: torch.zeros(NUM_LAYERS, NUM_EXPERTS),
        }
        nm, n_steps = 1.0, 8
        strategy = band_mf_strategy(bands=4, momentum=0.95)
        with_probe, _ = _bounds(trainable)
        stored = _per_step(with_probe)
        row_l2 = (
            strategy.streaming_matrix(n_steps=n_steps).row_norms_squared(n_steps).sqrt()
        )
        noise_fn, state = mf_gaussian_noise(
            trainable, strategy, n_steps=n_steps, noise_multiplier=nm, key=rng_key(3)
        )
        for t in range(n_steps):
            noised, state = noise_fn(clipped(trainable, max_norm=stored), state)
            base = {
                g: v / float(row_l2[t]) for g, v in noised.noise_stddev.values.items()
            }
            maha = sum((stored.values[g] / base[g]) ** 2 for g in stored.values)
            assert maha * nm**2 == pytest.approx(1.0, rel=1e-9)
        eps_with = dpftrl_acc.mf_gaussian(nm, strategy, n_steps=n_steps).epsilon_at(
            1e-5
        )
        eps_without = dpftrl_acc.mf_gaussian(
            nm, band_mf_strategy(bands=4, momentum=0.95), n_steps=n_steps
        ).epsilon_at(1e-5)
        assert eps_with == eps_without
