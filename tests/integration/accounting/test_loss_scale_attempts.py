"""Cross-wheel accounting for finite and overflowed loss-scale attempts."""

import torch

import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import Accountant
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.precision import loss_scaler
from opaque.random import key
from opaque.types import ClippedPytree


def test_loss_scale_overflow_still_consumes_the_attempted_dp_step():
    """Overflow backs off the scale but cannot suppress noise or accounting."""
    scaler, scaler_state = loss_scaler(
        init_scale=8.0,
        backoff_factor=0.5,
        growth_interval=2,
    )

    def loss(params, data):
        return scaler.scale_loss(torch.sqrt(data) * params, scaler_state)

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
        pre_clipping_transform=lambda grad: scaler.unscale_grads(grad, scaler_state),
        return_stats=True,
    )
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    step_process = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.1)
    accountant = Accountant()
    params = torch.tensor(1.0)

    all_finite = []
    for data in (torch.tensor([1.0]), torch.tensor([-1.0])):
        (grads, stats), clip_state = grad_fn(params, data, state=clip_state)
        assert isinstance(grads, ClippedPytree)
        assert torch.isfinite(grads.pytree).all()

        noised, noise_state = noise_fn(grads, noise_state)
        assert torch.isfinite(noised.pytree).all()
        params = params - 0.1 * noised.pytree
        accountant = accountant | step_process
        scaler_state = scaler.update(scaler_state, stats.all_finite)
        all_finite.append(stats.all_finite)

    assert all_finite == [True, False]
    assert noise_state._step_counter == 2
    assert accountant.process == step_process * 2
    assert scaler_state.scale == 4.0
    assert scaler_state.growth_tracker == 0
