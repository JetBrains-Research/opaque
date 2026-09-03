"""Backend-free state serialization for loss scaling."""

from opaque.precision import LossScalerState, loss_scaler
from opaque.serialization import from_state_dict, state_dict


def test_loss_scaler_state_round_trips_through_opaque_serialization() -> None:
    scaler, state = loss_scaler(init_scale=128.0, growth_interval=2)
    state = scaler.update(state, grads_were_finite=True)
    state = scaler.update(state, grads_were_finite=True)

    restored = from_state_dict(
        LossScalerState(scale=0.0, growth_tracker=0), state_dict(state)
    )

    assert restored == state
