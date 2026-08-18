"""Worker process for distributed JAX MF-state synchronization smoke coverage."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=2,
        process_id=args.rank,
        local_device_ids=[0],
    )
    try:
        from opaque.api.engine.backend import use_backend
        from opaque.distributed import sync, wait_for_everyone
        from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
        from opaque.jax import jax_backend
        from opaque.random import key
        from opaque.types import clipped

        template = {"weight": jnp.zeros(8, dtype=jnp.float32)}
        with use_backend(jax_backend()):
            noise_fn, state = mf_gaussian_noise(
                template,
                band_mf_strategy(bands=2, momentum=0.8),
                n_steps=3,
                min_sep=1,
                max_participations=1,
                noise_multiplier=1.0,
                key=key(31),
            )
            _, state = noise_fn(clipped(template, max_norm=1.0), state)
            synchronized = sync(state)
            wait_for_everyone()

        assert synchronized is state
        assert synchronized._step_counter == 1
        assert synchronized._first_max_norm == 1.0
        assert synchronized._first_max_norm_sync_fingerprint is not None
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
