"""Worker process for the MLX DP-FTRL distributed-state smoke test."""

from __future__ import annotations


def main() -> None:
    import mlx.core as mx

    from opaque.api.engine import runtime
    from opaque.distributed import sync, wait_for_everyone
    from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
    from opaque.mlx import mlx_backend
    from opaque.random import key
    from opaque.types import clipped

    mlx_backend()
    template = {"w": mx.zeros(16, dtype=mx.float32)}
    noise_fn, state = mf_gaussian_noise(
        template,
        band_mf_strategy(bands=2, momentum=0.5),
        n_steps=4,
        min_sep=4,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(7),
    )
    _, state = noise_fn(clipped(template, max_norm=1.0), state)
    synchronized = sync(state)
    wait_for_everyone()

    assert runtime.distributed_world_size() == 2
    assert synchronized._step_counter == 1
    assert synchronized._rng_key == state._rng_key


if __name__ == "__main__":
    main()
