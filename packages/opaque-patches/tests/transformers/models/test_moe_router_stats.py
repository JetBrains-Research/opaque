# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""MoE router statistics, fp32 router, router-logit forward, grouped default.

Tiny random-init Mellum 2.0 models (E = 8, k = 2, L = 2) exercise the
per-example load statistics under ``vmap(grad)``, the opt-in fp32 router, the
chunked-CE forward's ``opaque_router_logits`` contract, gradient checkpointing
and the grouped-vs-dense experts route.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import math
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from opaque.api.engine.clipping import clipped_grad
from opaque.api.patches.kernels.moe import _grouped_route_available
from opaque.api.patches.transformers.components.moe import _make_moe_experts_forward
from opaque.api.patches.transformers.components.moe_stats import (
    centred_load,
    load_balancing_surrogate,
    router_load_and_probs,
    router_z_loss,
)
from opaque.api.patches.transformers.components.router import (
    has_fp32_router,
    install_fp32_router,
    remove_fp32_router,
)
from opaque.exceptions import ConfigurationError
from opaque.functional import make_functional
from opaque.patches import apply_model_patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import build_moe_model

E, K, L = 8, 2, 2
VOCAB = 128


def _tiny_mellum(device, **overrides):
    torch.manual_seed(0)
    config = {
        "num_experts": E,
        "num_experts_per_tok": K,
        "moe_intermediate_size": 32,
        "num_hidden_layers": L,
    }
    config.update(overrides)
    model, mod = build_moe_model("mellum", device, **config)
    model.train()
    return model, mod


@contextlib.contextmanager
def _restored_class_forwards(model):
    """Undo class-level ``forward`` replacements made while the block runs.

    ``apply_model_patches`` installs class-level forwards guarded by a marker,
    so the first install in a process is captured by every later instance.
    Tests that request the chunked-CE forward restore the classes afterwards
    to leave the other Mellum tests on the default forward.
    """
    saved = {}
    for module in model.modules():
        cls = type(module)
        if cls not in saved:
            saved[cls] = cls.__dict__.get("forward")
    try:
        yield
    finally:
        for cls, forward in saved.items():
            if forward is None:
                if "forward" in cls.__dict__:
                    delattr(cls, "forward")
            else:
                cls.forward = forward


def _ragged_batch(device, lengths, seq_len):
    torch.manual_seed(1)
    batch = len(lengths)
    input_ids = torch.randint(3, VOCAB, (batch, seq_len), device=device)
    positions = torch.arange(seq_len, device=device)[None, :]
    mask = (positions < torch.as_tensor(lengths, device=device)[:, None]).long()
    input_ids = torch.where(mask.bool(), input_ids, torch.zeros_like(input_ids))
    labels = torch.where(mask.bool(), input_ids, torch.full_like(input_ids, -100))
    return input_ids, mask, labels


def _router_modules(model, mod):
    return [m for m in model.modules() if isinstance(m, mod.MellumTopKRouter)]


def _random_router_logits(seed, num_layers=L, seq_len=12, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(seq_len, E, generator=generator).to(device)
        for _ in range(num_layers)
    )


def _eager_stats(router_logits, mask):
    """Token-loop reference for ``router_load_and_probs``."""
    m = [1.0 if v != 0 else 0.0 for v in mask.reshape(-1).tolist()]
    t_x = sum(m)
    h = torch.zeros(len(router_logits), E)
    probs = torch.zeros(E)
    for layer, z in enumerate(router_logits):
        z = z.reshape(-1, E)
        for t in range(z.shape[0]):
            if m[t] == 0.0:
                continue
            p = torch.softmax(z[t].float(), dim=-1)
            for e in torch.topk(p, K).indices.tolist():
                h[layer, e] += 1.0
            probs += p
    if t_x == 0:
        return h, probs, torch.tensor(0.0)
    return h / t_x, probs / (len(router_logits) * t_x), torch.tensor(t_x)


# ---------------------------------------------------------------------------
# T2: router_load_and_probs
# ---------------------------------------------------------------------------


class TestRouterLoadAndProbs:
    def test_structure_and_bounds(self):
        logits = _random_router_logits(0)
        mask = torch.ones(12, dtype=torch.long)
        mask[-3:] = 0
        h, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)

        assert h.shape == (L, E)
        assert probs.shape == (E,)
        assert t_x.shape == ()
        assert h.dtype == probs.dtype == t_x.dtype == torch.float32
        assert t_x.item() == 9.0
        torch.testing.assert_close(h.sum(-1), torch.full((L,), float(K)))
        assert (h >= 0).all()
        assert (h <= 1).all()
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        d = centred_load(h, top_k=K)
        torch.testing.assert_close(d.sum(-1), torch.zeros(L))
        assert torch.equal(centred_load(h, top_k=K, valid_tokens=t_x), d)
        assert d.norm().item() <= math.sqrt(K * L * (1 - K / E)) + 1e-6

    def test_matches_eager_loop_and_excludes_padding(self):
        logits = _random_router_logits(1)
        mask = torch.ones(12, dtype=torch.long)
        mask[7:] = 0
        h, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        h_ref, p_ref, t_ref = _eager_stats(logits, mask)
        torch.testing.assert_close(h, h_ref)
        torch.testing.assert_close(probs, p_ref)
        torch.testing.assert_close(t_x, t_ref)

        # Padded positions do not influence the result at all.
        perturbed = tuple(
            torch.cat([z[:7], torch.randn_like(z[7:]) * 50]) for z in logits
        )
        h2, p2, _ = router_load_and_probs(perturbed, mask, top_k=K, num_layers=L)
        torch.testing.assert_close(h2, h)
        torch.testing.assert_close(p2, probs)

    def test_mask_none_counts_every_position(self):
        logits = _random_router_logits(2)
        h_none, p_none, t_none = router_load_and_probs(
            logits, None, top_k=K, num_layers=L
        )
        h_ones, p_ones, t_ones = router_load_and_probs(
            logits, torch.ones(12), top_k=K, num_layers=L
        )
        torch.testing.assert_close(h_none, h_ones)
        torch.testing.assert_close(p_none, p_ones)
        assert t_none.item() == t_ones.item() == 12.0

    @pytest.mark.parametrize(
        "variant",
        ["additive", "mixed_sign", "two_dim"],
    )
    def test_non_binary_masks_are_binarised(self, variant):
        """Any mask is read as ``mask != 0``; entries never act as weights.

        Multiplying a float mask (as HF's aux loss does) would score padding
        under an additive mask and break the sensitivity bound under a
        mixed-sign one; binarising keeps ``h`` a load fraction in both cases.
        """
        logits = _random_router_logits(3)
        binary = torch.ones(12, dtype=torch.long)
        binary[8:] = 0
        if variant == "additive":
            mask = torch.where(binary.bool(), 0.0, -1e9)
        elif variant == "mixed_sign":
            mask = torch.where(binary.bool(), -1.0, 0.0)
            mask[0] = 2.5
        else:
            mask = binary[None, :]
        binarised = (mask != 0).long().reshape(-1)
        expected = router_load_and_probs(logits, binarised, top_k=K, num_layers=L)
        actual = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
        h, probs, t_x = actual
        assert t_x.item() == binarised.sum().item()
        assert (h >= 0).all()
        assert (h <= 1).all()
        torch.testing.assert_close(h.sum(-1), torch.full((L,), float(K)))
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        assert (
            centred_load(h, top_k=K).norm().item()
            <= math.sqrt(K * L * (1 - K / E)) + 1e-6
        )

    def test_fully_masked_row_gives_zero_statistics(self):
        logits = _random_router_logits(4)
        mask = torch.zeros(12, dtype=torch.long)
        h, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        assert t_x.item() == 0.0
        assert torch.equal(h, torch.zeros(L, E))
        assert torch.equal(probs, torch.zeros(E))
        assert torch.isfinite(h).all()
        assert torch.isfinite(probs).all()
        assert torch.equal(
            centred_load(h, top_k=K, valid_tokens=t_x), torch.zeros(L, E)
        )
        z_loss = router_z_loss(logits, mask)
        assert z_loss.item() == 0.0

    def test_duplicated_capture_raises(self):
        logits = _random_router_logits(5)
        with pytest.raises(ConfigurationError, match="layers"):
            router_load_and_probs(logits + logits, None, top_k=K, num_layers=L)
        with pytest.raises(ConfigurationError, match="layers"):
            router_load_and_probs(logits[:1], None, top_k=K, num_layers=L)
        # The category is a ``ValueError`` for callers that catch the builtin.
        assert issubclass(ConfigurationError, ValueError)

    def test_mask_length_mismatch_raises(self):
        logits = _random_router_logits(6)
        with pytest.raises(ConfigurationError, match="tokens"):
            router_load_and_probs(logits, torch.ones(11), top_k=K, num_layers=L)

    def test_vmap_matches_per_example_loop(self):
        batch = 4
        generator = torch.Generator().manual_seed(7)
        logits = tuple(torch.randn(batch, 12, E, generator=generator) for _ in range(L))
        mask = torch.ones(batch, 12, dtype=torch.long)
        mask[1, 9:] = 0
        mask[2, :] = 0
        mask[3, 5:] = 0

        def stats(z0, z1, m):
            return router_load_and_probs((z0, z1), m, top_k=K, num_layers=L)

        h_v, p_v, t_v = torch.vmap(stats)(logits[0], logits[1], mask)
        for b in range(batch):
            h_b, p_b, t_b = stats(logits[0][b], logits[1][b], mask[b])
            torch.testing.assert_close(h_v[b], h_b)
            torch.testing.assert_close(p_v[b], p_b)
            torch.testing.assert_close(t_v[b], t_b)

    def test_load_from_logits_equals_router_indices(self, device):
        """``h`` recovered from the captured logits equals the executed routes."""
        model, mod = _tiny_mellum(device)
        seq_len = 12
        input_ids, mask, _ = _ragged_batch(device, [seq_len, 8], seq_len)
        captured = []
        hooks = [
            router.register_forward_hook(lambda _m, _i, out: captured.append(out[2]))
            for router in _router_modules(model, mod)
        ]
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids, attention_mask=mask, output_router_logits=True
            )
        for hook in hooks:
            hook.remove()
        assert len(out.router_logits) == L
        assert len(captured) == L

        for b in range(input_ids.shape[0]):
            per_example = tuple(
                z.reshape(input_ids.shape[0], seq_len, E)[b] for z in out.router_logits
            )
            h, _, t_x = router_load_and_probs(
                per_example, mask[b], top_k=K, num_layers=L
            )
            for layer, indices in enumerate(captured):
                idx = indices.reshape(input_ids.shape[0], seq_len, K)[b]
                valid = idx[mask[b].bool()]
                counts = torch.bincount(valid.reshape(-1), minlength=E).float()
                torch.testing.assert_close(h[layer], counts / t_x)

    def test_surrogate_and_z_loss_closed_forms(self):
        logits = _random_router_logits(8)
        mask = torch.ones(12, dtype=torch.long)
        mask[10:] = 0
        _, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        f_tilde = torch.softmax(torch.randn(E), dim=-1) * K
        w = t_x / 12.0
        expected = E * w * ((f_tilde - K / E) * probs).sum()
        actual = load_balancing_surrogate(probs, f_tilde, w, num_experts=E, top_k=K)
        torch.testing.assert_close(actual, expected)
        # sum_e P = 1: the centring shifts the value by a constant only.
        uncentred = E * w * (f_tilde * probs).sum() - E * w * (K / E)
        torch.testing.assert_close(actual, uncentred)

        z_expected = torch.stack(
            [
                (torch.logsumexp(z.float(), dim=-1) ** 2 * mask.float()).sum()
                for z in logits
            ]
        ).sum() / (L * 10)
        torch.testing.assert_close(router_z_loss(logits, mask), z_expected)


# ---------------------------------------------------------------------------
# T1: fp32 router
# ---------------------------------------------------------------------------


class TestFp32Router:
    def test_dtypes_and_indices_match_unpatched_router(self, device):
        model, mod = _tiny_mellum(device)
        router = _router_modules(model, mod)[0]
        torch.manual_seed(2)
        hidden = torch.randn(9, model.config.hidden_size, device=device)
        with torch.no_grad():
            ref_logits, ref_scores, ref_indices = router(hidden)
        undo = install_fp32_router(model)
        assert has_fp32_router(model)
        with torch.no_grad():
            logits, scores, indices = router(hidden)
        assert logits.dtype == torch.float32
        assert scores.dtype == hidden.dtype
        assert torch.equal(indices, ref_indices)
        torch.testing.assert_close(logits, ref_logits)
        torch.testing.assert_close(scores, ref_scores)

        undo()
        assert not has_fp32_router(model)
        with torch.no_grad():
            back_logits, _, back_indices = router(hidden)
        assert torch.equal(back_logits, ref_logits)
        assert torch.equal(back_indices, ref_indices)
        # The class-level forward was never touched.
        assert "forward" not in router.__dict__
        assert mod.MellumTopKRouter.forward is type(router).forward

    def test_bf16_hidden_states_get_fp32_logits(self, device):
        model, mod = _tiny_mellum(device)
        model.to(torch.bfloat16)
        router = _router_modules(model, mod)[0]
        hidden = torch.randn(
            9, model.config.hidden_size, device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            stock_logits, stock_scores, _ = router(hidden)
        assert stock_logits.dtype == torch.bfloat16
        install_fp32_router(model)
        with torch.no_grad():
            logits, scores, indices = router(hidden)
        assert logits.dtype == torch.float32
        assert scores.dtype == torch.bfloat16 == stock_scores.dtype
        assert indices.shape == (9, K)
        torch.testing.assert_close(
            scores.float().sum(-1), torch.ones(9, device=device), atol=2e-2, rtol=0
        )
        remove_fp32_router(model)

    def test_installed_through_apply_model_patches_and_removable(self, device):
        model, mod = _tiny_mellum(device)
        apply_model_patches(model, router_fp32=True)
        assert has_fp32_router(model)
        assert all("forward" in r.__dict__ for r in _router_modules(model, mod))
        # Idempotent: a second install keeps a single removable binding.
        apply_model_patches(model, router_fp32=True)
        apply_model_patches(model, router_fp32=False)
        assert not has_fp32_router(model)
        assert all("forward" not in r.__dict__ for r in _router_modules(model, mod))
        # Default: the router is left alone.
        other, _ = _tiny_mellum(device)
        assert not has_fp32_router(other)

    def test_model_without_a_matching_router_is_rejected(self, device):
        """A requested fp32 router never installs nothing silently."""
        from opaque.exceptions import ConfigurationError

        dense = torch.nn.Sequential(torch.nn.Linear(4, 4)).to(device)
        with pytest.raises(ConfigurationError, match="nothing to install"):
            install_fp32_router(dense)
        # A class object that matches no module (remote-code / re-imported
        # modeling modules) is rejected the same way.
        model, _ = _tiny_mellum(device)

        class NotTheRouter(torch.nn.Module):
            pass

        with pytest.raises(ConfigurationError, match="NotTheRouter"):
            install_fp32_router(model, router_cls=NotTheRouter)
        assert not has_fp32_router(model)

    def test_vmap_and_per_example_gradients(self, device):
        model, _ = _tiny_mellum(device)
        install_fp32_router(model)
        input_ids, mask, labels = _ragged_batch(device, [12, 12, 9, 12], 12)

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(tr, fr, ids, m, lbl):
            return fmodel({**fr, **tr}, ids, attention_mask=m, labels=lbl).loss

        with torch.no_grad():
            losses = torch.vmap(per_example_loss, in_dims=(None, None, 0, 0, 0))(
                trainable, frozen, input_ids, mask, labels
            )
        assert losses.shape == (4,)
        assert torch.isfinite(losses).all()

        grad_fn, state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, mask, labels, state=state)
        router_grads = [g for n, g in grads.pytree.items() if ".mlp.gate." in n]
        assert router_grads
        assert all(torch.isfinite(g).all() for g in router_grads)
        assert any(g.abs().sum() > 0 for g in router_grads)


# ---------------------------------------------------------------------------
# T4: chunked-CE forward with opaque_router_logits
# ---------------------------------------------------------------------------


def _per_example_forward(model):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def forward(tr, ids, m, lbl, **kw):
        return fmodel(
            {**frozen, **tr}, input_ids=ids, attention_mask=m, labels=lbl, **kw
        )

    return forward, trainable


class TestRouterLogitsForward:
    def test_named_parameter_only_with_fused_linear_cross_entropy(self, device):
        model, _ = _tiny_mellum(device)
        params = inspect.signature(model.forward).parameters
        assert "opaque_router_logits" not in params
        with _restored_class_forwards(model):
            apply_model_patches(model, fused_linear_cross_entropy=True)
            params = inspect.signature(model.forward).parameters
            assert "opaque_router_logits" in params
            assert (
                params["opaque_router_logits"].kind is not inspect.Parameter.VAR_KEYWORD
            )

    def test_router_logits_under_vmap_keep_chunked_loss(self, device):
        model, _ = _tiny_mellum(device)
        seq_len = 12
        input_ids, mask, labels = _ragged_batch(device, [seq_len, 8, seq_len], seq_len)
        with _restored_class_forwards(model):
            apply_model_patches(model, fused_linear_cross_entropy=True)
            forward, trainable = _per_example_forward(model)

            def with_router(tr, ids, m, lbl):
                out = forward(tr, ids, m, lbl, opaque_router_logits=True)
                assert out.logits is None
                assert out.aux_loss is None
                assert len(out.router_logits) == L
                stacked = torch.stack([z.reshape(-1, E) for z in out.router_logits])
                return out.loss, stacked

            def loss_only(tr, ids, m, lbl):
                return forward(tr, ids, m, lbl, opaque_fused_loss_only=True).loss

            def default(tr, ids, m, lbl):
                out = forward(tr, ids, m, lbl)
                assert out.logits is not None
                return out.loss

            with torch.no_grad():
                loss_r, logits = torch.vmap(with_router, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
                loss_c = torch.vmap(loss_only, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
                loss_d = torch.vmap(default, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
            assert logits.shape == (3, L, seq_len, E)
            assert torch.equal(loss_r, loss_c)
            torch.testing.assert_close(loss_r, loss_d, atol=1e-5, rtol=1e-5)

    def test_output_router_logits_still_takes_hf_fallback(self, device):
        model, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(model):
            apply_model_patches(model, fused_linear_cross_entropy=True)
            with torch.no_grad():
                hf = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    output_router_logits=True,
                )
                ours = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    opaque_router_logits=True,
                )
            assert hf.aux_loss is not None
            assert hf.logits is not None
            assert ours.aux_loss is None
            assert ours.logits is None
            assert len(ours.router_logits) == L
            # HF adds ``coef * aux`` to the loss; ours is the plain CE.
            expected_hf = ours.loss + model.config.router_aux_loss_coef * hf.aux_loss
            torch.testing.assert_close(hf.loss, expected_hf, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# T5: gradient checkpointing
# ---------------------------------------------------------------------------


def _stats_loss(forward, f_tilde):
    def loss_fn(tr, ids, m, lbl):
        out = forward(tr, ids, m, lbl, opaque_router_logits=True)
        h, probs, t_x = router_load_and_probs(
            out.router_logits, m, top_k=K, num_layers=L
        )
        surrogate = load_balancing_surrogate(
            probs, f_tilde, t_x / 12.0, num_experts=E, top_k=K
        )
        return out.loss + surrogate, (h, probs)

    return loss_fn


def test_gradient_checkpointing_matches_plain_run(device):
    model, _ = _tiny_mellum(device)
    input_ids, mask, labels = _ragged_batch(device, [12, 7, 12, 10], 12)
    f_tilde = torch.tensor([0.5, 0.1, 0.4, 0.2, 0.3, 0.2, 0.1, 0.2], device=device)
    with _restored_class_forwards(model):
        apply_model_patches(model, fused_linear_cross_entropy=True)
        forward, trainable = _per_example_forward(model)
        grad_fn, state = clipped_grad(
            _stats_loss(forward, f_tilde),
            has_aux=True,
            return_aux=True,
            batch_argnums=(1, 2, 3),
            clipping_norm=1e6,
        )
        (plain, plain_aux), _ = grad_fn(trainable, input_ids, mask, labels, state=state)

        model.gradient_checkpointing_enable()
        assert any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
        (ckpt, ckpt_aux), _ = grad_fn(trainable, input_ids, mask, labels, state=state)
        model.gradient_checkpointing_disable()

    h_plain, p_plain = plain_aux.loss_aux
    h_ckpt, p_ckpt = ckpt_aux.loss_aux
    assert torch.equal(h_plain, h_ckpt)
    torch.testing.assert_close(p_plain, p_ckpt)
    torch.testing.assert_close(plain_aux.loss_values, ckpt_aux.loss_values)
    for name, g in plain.pytree.items():
        torch.testing.assert_close(ckpt.pytree[name], g, atol=1e-5, rtol=1e-4)
    # The surrogate gradient flows through the captured logits into the router.
    router_grads = [g for n, g in plain.pytree.items() if ".mlp.gate." in n]
    assert router_grads
    assert all(g.abs().sum() > 0 for g in router_grads)


# ---------------------------------------------------------------------------
# T6: grouped vs dense experts route, and the default resolution
# ---------------------------------------------------------------------------


def test_grouped_route_available_matches_host():
    from opaque.api.patches.kernels import moe as moe_kernels

    expected = (moe_kernels._TRITON_AVAILABLE and torch.cuda.is_available()) or (
        hasattr(torch, "_grouped_mm") or hasattr(torch.nn.functional, "grouped_mm")
    )
    assert _grouped_route_available() is bool(expected)


@pytest.mark.parametrize(
    ("kernels", "route_available", "explicit", "expected"),
    [
        (False, True, None, True),
        (False, False, None, False),
        (True, False, None, True),
        (True, True, False, False),
        (False, True, False, False),
        (False, False, True, True),
    ],
)
def test_grouped_moe_default_decoupled_from_kernels(
    monkeypatch, kernels, route_available, explicit, expected
):
    pytest.importorskip("transformers.models.mellum.modeling_mellum")
    from opaque.api.patches.transformers.models.mellum import apply_mellum_patches

    captured = []
    monkeypatch.setattr(
        "opaque.api.patches.transformers._factory._patch_forward",
        lambda cls, factory, model: captured.append((cls, factory)),
    )
    monkeypatch.setattr(
        "opaque.api.patches.kernels.moe._grouped_route_available",
        lambda: route_available,
    )
    kwargs = {} if explicit is None else {"grouped_moe": explicit}
    apply_mellum_patches(
        None,
        performance=False,
        compat=False,
        kernels=kernels,
        moe=True,
        **kwargs,
    )
    experts = [f for cls, f in captured if cls.__name__ == "MellumExperts"]
    assert len(experts) == 1
    assert experts[0].keywords["grouped"] is expected


def test_grouped_and_dense_routes_agree_on_statistics(device, monkeypatch):
    """Grouped-GEMM and dense experts give identical ``h`` and close gradients."""
    from opaque.api.patches.kernels._grouped_moe import (
        Opaque_GroupedMoE,
        grouped_mm_available,
    )

    if not grouped_mm_available():
        pytest.skip("torch._grouped_mm unavailable")
    if device.type == "cuda":
        pytest.skip("CUDA fp32 activations take the dense route by design")

    num_experts = 16  # the grouped route only engages from 16 experts
    model, mod = _tiny_mellum(
        device, num_experts=num_experts, num_experts_per_tok=K, moe_intermediate_size=16
    )
    calls = []
    original_apply = Opaque_GroupedMoE.apply

    def counting_apply(*args, **kwargs):
        calls.append(1)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(Opaque_GroupedMoE, "apply", counting_apply)

    input_ids, mask, labels = _ragged_batch(device, [12, 9, 12], 12)
    f_tilde = torch.full((num_experts,), K / num_experts, device=device)
    f_tilde[0] += 0.2
    f_tilde[1] -= 0.2

    def stats_loss(forward):
        def loss_fn(tr, ids, msk, lbl):
            out = forward(tr, ids, msk, lbl, opaque_router_logits=True)
            h, probs, t_x = router_load_and_probs(
                out.router_logits, msk, top_k=K, num_layers=L
            )
            surrogate = load_balancing_surrogate(
                probs, f_tilde, t_x / 12.0, num_experts=num_experts, top_k=K
            )
            return out.loss + surrogate, h

        return loss_fn

    results = {}
    with _restored_class_forwards(model):
        apply_model_patches(model, fused_linear_cross_entropy=True)
        # The experts route is captured class-wide, so the dense twin gets its
        # forward bound per instance instead of through a second patch call.
        dense = copy.deepcopy(model)
        for experts in dense.modules():
            if isinstance(experts, mod.MellumExperts):
                experts.forward = types.MethodType(
                    _make_moe_experts_forward(type(experts).forward, grouped=False),
                    experts,
                )
        for name, m in (("grouped", model), ("dense", dense)):
            forward, trainable = _per_example_forward(m)
            grad_fn, state = clipped_grad(
                stats_loss(forward),
                has_aux=True,
                return_aux=True,
                batch_argnums=(1, 2, 3),
                clipping_norm=1e6,
            )
            before = len(calls)
            (grads, aux), _ = grad_fn(trainable, input_ids, mask, labels, state=state)
            results[name] = (grads, aux, len(calls) - before)

    assert results["grouped"][2] > 0, "grouped route was not taken"
    assert results["dense"][2] == 0, "dense model must not use the grouped kernel"
    assert torch.equal(results["grouped"][1].loss_aux, results["dense"][1].loss_aux)
    torch.testing.assert_close(
        results["grouped"][1].loss_values,
        results["dense"][1].loss_values,
        atol=1e-4,
        rtol=1e-4,
    )
    for name, g in results["dense"][0].pytree.items():
        torch.testing.assert_close(
            results["grouped"][0].pytree[name], g, atol=1e-4, rtol=1e-3
        )
