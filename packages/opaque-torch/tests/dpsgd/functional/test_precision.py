"""Precision support over the functional DP step.

Three axes:

1. **Full-cast bf16** — ``model.to(bfloat16)`` flows cleanly through
   ``vmap(grad)`` → ``clipped_grad`` → ``gaussian_noise`` → ``adamw``.
   Reduction-stability promotion to fp32 already lives in
   ``opaque/clipping/clipped_fun.py:_sum_clipped_tensor``; we exercise it
   end-to-end and check dtype is preserved on the public boundary.

2. **TF32** — global toggle (``torch.backends.cuda.matmul.allow_tf32`` /
   ``torch.set_float32_matmul_precision``) is orthogonal to the
   functional DP step. CUDA-only.

3. **fp16 autocast + functional loss-scale** — ``torch.autocast(fp16)``
   inside the loss closure works with ``vmap(grad)``. The DP-critical
   invariant — *clipping must operate on unscaled grads* — is preserved
   by routing the unscale through ``clipped_grad``'s
   ``pre_clipping_transform`` parameter. No new ``OpaqueGradScaler``
   source module is needed: that hook IS the functional analog. CUDA-only
   (CPU autocast does not support fp16).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.api.engine.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import adamw, apply_updates
from opaque.random import key
from opaque.torch.functional import make_functional


def _build_model(seed: int = 0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )


def _make_batch(dtype: torch.dtype, device: torch.device, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(5, 8, generator=g).to(device=device, dtype=dtype)
    y = torch.randn(5, 4, generator=g).to(device=device, dtype=dtype)
    return x, y


def _full_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, noise_stddev: float
):
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=noise_stddev, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer_step, opt_state = adamw(params, lr=1e-2)
    updates, _ = optimizer_step(noised, opt_state, params=params)
    new_params = apply_updates(params, updates)
    return grads.pytree, noised.pytree, new_params


# ----------------------------------------------------------------------------
# Full-cast bf16
# ----------------------------------------------------------------------------


def test_bf16_grad_finite_cpu():
    """Full-cast bf16 produces finite per-example clipped gradients."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    grads, noised, _ = _full_step(model, x, y, noise_stddev=0.0)
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite bf16 gradient"
    for n in noised:
        assert torch.isfinite(n).all(), "non-finite noised gradient"


def test_bf16_grad_dtype_preserved_cpu():
    """Reduction promotes to fp32 internally, but caller sees bf16 back."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    grads, noised, _ = _full_step(model, x, y, noise_stddev=0.0)
    for g in grads:
        assert g.dtype == torch.bfloat16, f"grad dtype leaked to {g.dtype}"
    for n in noised:
        assert n.dtype == torch.bfloat16, f"noised dtype leaked to {n.dtype}"


def test_bf16_full_pipeline_proximity_to_fp32_cpu():
    """bf16 grads are close to fp32 reference within calibrated tolerance."""
    model_fp32 = _build_model()
    x_fp32, y_fp32 = _make_batch(torch.float32, torch.device("cpu"))
    grads_fp32, _, _ = _full_step(model_fp32, x_fp32, y_fp32, noise_stddev=0.0)

    model_bf16 = _build_model().to(dtype=torch.bfloat16)
    x_bf16 = x_fp32.to(dtype=torch.bfloat16)
    y_bf16 = y_fp32.to(dtype=torch.bfloat16)
    grads_bf16, _, _ = _full_step(model_bf16, x_bf16, y_bf16, noise_stddev=0.0)

    for g32, g16 in zip(grads_fp32, grads_bf16, strict=True):
        diff = (g32.float() - g16.float()).pow(2).mean().item()
        assert diff < 5e-2, f"bf16 grad too far from fp32 reference: mse={diff:.4f}"


def test_bf16_optimizer_step_finite_cpu():
    """adamw.update on bf16 grads produces finite updated params."""
    model = _build_model().to(dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cpu"))
    _, _, new_params = _full_step(model, x, y, noise_stddev=0.5)
    for p in new_params:
        assert torch.isfinite(p).all(), "adamw produced non-finite param update"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_bf16_full_pipeline_cuda():
    """Same end-to-end claim on CUDA (bf16 hardware path)."""
    model = _build_model().to(device="cuda", dtype=torch.bfloat16)
    x, y = _make_batch(torch.bfloat16, torch.device("cuda"))
    grads, noised, new_params = _full_step(model, x, y, noise_stddev=0.5)
    for g in grads:
        assert torch.isfinite(g).all()
        assert g.dtype == torch.bfloat16
    for n in noised:
        assert torch.isfinite(n).all()
        assert n.dtype == torch.bfloat16
    for p in new_params:
        assert torch.isfinite(p).all()
        assert p.dtype == torch.bfloat16


# ----------------------------------------------------------------------------
# TF32 (CUDA-only)
# ----------------------------------------------------------------------------


@pytest.fixture
def _saved_tf32():
    """Snapshot + restore TF32 flags so tests don't bleed into one another."""
    saved_matmul = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn = torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved_matmul
    torch.backends.cudnn.allow_tf32 = saved_cudnn


def _step_simple(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Trimmed DP step for TF32 parity — AdamW without DP bias correction."""
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=0.0, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer_step, opt_state = adamw(params, lr=1e-2)
    updates, _ = optimizer_step(noised, opt_state, params=params)
    new_params = apply_updates(params, updates)
    return grads, new_params


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is CUDA-only")
@pytest.mark.usefixtures("_saved_tf32")
def test_tf32_on_off_parity_within_tolerance():
    """TF32 on vs off: matmul precision differs but gradient flow stays close."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grads_off, params_off = _step_simple(model, x, y)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    grads_on, params_on = _step_simple(model, x, y)

    for g_off, g_on in zip(grads_off, grads_on, strict=True):
        torch.testing.assert_close(g_off, g_on, rtol=2e-3, atol=1e-4)
    for p_off, p_on in zip(params_off, params_on, strict=True):
        torch.testing.assert_close(p_off, p_on, rtol=2e-3, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is CUDA-only")
@pytest.mark.usefixtures("_saved_tf32")
def test_tf32_via_set_float32_matmul_precision():
    """``torch.set_float32_matmul_precision('high')`` is also non-disruptive."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4)).cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    torch.set_float32_matmul_precision("highest")
    grads_high, _ = _step_simple(model, x, y)

    torch.set_float32_matmul_precision("medium")
    grads_med, _ = _step_simple(model, x, y)

    for ghigh, gmed in zip(grads_high, grads_med, strict=True):
        torch.testing.assert_close(ghigh, gmed, rtol=5e-3, atol=5e-4)

    torch.set_float32_matmul_precision("highest")


# ----------------------------------------------------------------------------
# fp16 autocast + functional loss-scale (CUDA-only)
# ----------------------------------------------------------------------------


_AUTOCAST_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fp16 autocast requires CUDA"
)


def _step_eager_no_autocast(model, x, y):
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    return grads.pytree


def _step_with_autocast(model, x, y, *, loss_scale: float = 1.0):
    """Run the DP step with autocast(fp16). loss_scale=1.0 is the bare probe.

    When loss_scale > 1, the loss is multiplied inside grad and the gradient
    pytree is divided by the same factor BEFORE clipping — the functional
    analog of ``GradScaler.unscale_grads_``. Routed through
    ``pre_clipping_transform`` so clipping operates on unscaled grads.
    """
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            loss = ((pred - yi) ** 2).mean()
        return loss * loss_scale

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        pre_clipping_transform=lambda g: tuple(t / loss_scale for t in g),
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)
    return grads.pytree


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_bare_probe_runs():
    """Sanity: vmap(grad(autocast(loss))) executes without exception."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")
    grads = _step_with_autocast(model, x, y, loss_scale=1.0)
    assert grads is not None
    assert len(grads) > 0


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_bare_probe_parity_to_fp32():
    """Bare autocast grads should be near fp32 reference (no underflow yet)."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    g_fp32 = _step_eager_no_autocast(model, x, y)
    g_fp16 = _step_with_autocast(model, x, y, loss_scale=1.0)

    for a, b in zip(g_fp32, g_fp16, strict=True):
        diff = (a.float() - b.float()).pow(2).mean().item()
        assert diff < 1e-2, f"fp16 autocast mse vs fp32 too large: {diff:.5f}"


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_with_loss_scale_preserves_clipping_invariant():
    """Clipping must see *unscaled* grads — sensitivity calibration would break otherwise."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    g_no_scale = _step_with_autocast(model, x, y, loss_scale=1.0)
    g_scaled = _step_with_autocast(model, x, y, loss_scale=128.0)

    for a, b in zip(g_no_scale, g_scaled, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_with_loss_scale_no_underflow():
    """Loss-scale prevents gradient underflow with very small inputs."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.full((5, 8), 1e-4, device="cuda")
    y = torch.full((5, 4), 1e-4, device="cuda")

    g_scaled = _step_with_autocast(model, x, y, loss_scale=2**16)
    for g in g_scaled:
        assert torch.isfinite(g).all(), "scaled fp16 produced non-finite grad"


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_with_loss_scaler_primitive_matches_inline_lambda():
    """Functional ``loss_scaler`` matches the inline-lambda baseline.

    Pins down that promoting the loss-scale machinery from an inline lambda
    to the ``opaque.precision.loss_scaler`` primitive does not change the
    clipped gradient — i.e. the primitive really is just packaging the
    same ``pre_clipping_transform`` shape, with the DP-critical
    unscale-before-clip ordering preserved.
    """
    from opaque.precision import loss_scaler

    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    g_inline = _step_with_autocast(model, x, y, loss_scale=128.0)

    scaler, scaler_state = loss_scaler(init_scale=128.0)

    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            loss = ((pred - yi) ** 2).mean()
        return scaler.scale_loss(loss, scaler_state)

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        pre_clipping_transform=lambda g: scaler.unscale_grads(g, scaler_state),
    )
    g_primitive, _ = grad_fn(params, x, y, state=clip_state)

    for a, b in zip(g_inline, g_primitive.pytree, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_full_pipeline_with_optimizer():
    """End-to-end: autocast → scaled-grad → clipped → noised → adamw step."""
    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    fmodel, params = make_functional(model)
    loss_scale = 128.0

    def loss_fn(p, xi, yi):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
            loss = ((pred - yi) ** 2).mean()
        return loss * loss_scale

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        pre_clipping_transform=lambda g: tuple(t / loss_scale for t in g),
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=0.5, key=key(42))
    noised, _ = noise_fn(grads, ns)

    optimizer_step, opt_state = adamw(params, lr=1e-2)
    updates, _ = optimizer_step(noised, opt_state, params=params)
    new_params = apply_updates(params, updates)

    for p in new_params:
        assert torch.isfinite(p).all(), (
            "fp16 autocast pipeline produced non-finite params"
        )


@pytest.mark.cuda
@_AUTOCAST_REQUIRES_CUDA
def test_fp16_autocast_overflow_is_observed_and_still_produces_a_step():
    """The overflow branch, on the path that can actually overflow.

    ``return_stats`` reports finiteness of the *pre-clipping* per-example
    gradients, so a scale large enough to overflow fp16 must set
    ``all_finite=False`` while clipping still returns a bounded, finite
    contribution — that is what lets the surrounding loop compose the
    accountant on every attempted step instead of branching on the batch.
    """
    from opaque.precision import loss_scaler

    torch.manual_seed(0)
    model = _build_model().cuda()
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 4, device="cuda")

    fmodel, params = make_functional(model)

    def run(scale: float):
        scaler, scaler_state = loss_scaler(init_scale=scale)

        def loss_fn(p, xi, yi):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
                loss = ((pred - yi) ** 2).mean()
            return scaler.scale_loss(loss, scaler_state)

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            pre_clipping_transform=lambda g: scaler.unscale_grads(g, scaler_state),
            return_stats=True,
        )
        (grads, stats), _ = grad_fn(params, x, y, state=clip_state)
        return grads, stats

    grads, stats = run(128.0)
    assert stats.all_finite is True

    overflowed, overflow_stats = run(float(2**30))
    assert overflow_stats.all_finite is False, (
        "a scale past finfo(float16).max must be visible to the loss scaler"
    )
    for g in overflowed.pytree:
        assert torch.isfinite(g).all(), (
            "clipping must sanitize the overflow rather than propagate it"
        )
    assert overflowed.sensitivity == grads.sensitivity
