"""Internal distributed synchronization registration for denoiser state."""

from __future__ import annotations

from opaque.distributed import (
    assert_pytree_equal,
    assert_scalar_equal,
    is_distributed,
    register_sync_type,
)

from opaque.denoising._kalman import DiskDenoiserState


def sync_disk_denoiser_state(state: DiskDenoiserState) -> DiskDenoiserState:
    """Validate DiSK denoiser state consistency across ranks."""
    if not is_distributed():
        return state

    assert_scalar_equal(
        int(state._step_counter), name="DiskDenoiserState._step_counter"
    )
    assert_pytree_equal(state._estimate, name="DiskDenoiserState._estimate")
    assert_pytree_equal(state._error_var, name="DiskDenoiserState._error_var")
    return state


register_sync_type(DiskDenoiserState, sync_disk_denoiser_state)
