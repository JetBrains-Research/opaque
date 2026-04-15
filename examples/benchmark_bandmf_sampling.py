#!/usr/bin/env python3
"""Compare BandMF privacy calibration: cyclic Poisson vs b-min-sep (validation).

Run from repo root (requires opaque-accounting built, opaque installed)::

    python examples/benchmark_bandmf_sampling.py

This prints calibrated noise multipliers and wall times for fixed hyperparameters
without training a model.
"""

from __future__ import annotations

import time

import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque.noise.mf import band_mf_strategy


def main() -> None:
    n_steps = 2000
    bands = 8
    dataset_size = 50_000
    batch_size = 256
    target_epsilon = 3.0
    target_delta = 1e-5

    p0 = batch_size / dataset_size
    sampling_prob_cp = batch_size * bands / dataset_size

    strategy = band_mf_strategy(n_steps=n_steps, bands=bands, momentum=0.95)
    coef = strategy.coefficients

    def calibrate_one(build_proc, label: str) -> tuple[float, float, float]:
        t0 = time.perf_counter()

        def acct(nm: float):
            return build_proc(nm)

        r = cal.calibrate(
            cal.epsilon_budget(target_epsilon, delta=target_delta),
            acct,
            param_min=0.2,
            param_max=15.0,
            tolerance=1e-2,
        )
        elapsed = time.perf_counter() - t0
        achieved = r.achieved
        return r.param, achieved, elapsed

    print("BandMF sampling benchmark (no training)")
    print(f"  n_steps={n_steps}, bands={bands}, |D|={dataset_size}, batch={batch_size}")
    print(f"  target ε={target_epsilon}, δ={target_delta:.0e}")
    print()

    nm_cp, eps_cp, t_cp = calibrate_one(
        lambda nm: acc.cyclic_poisson(
            acc.band_mf(
                nm,
                sensitivity=strategy.sensitivity,
                num_groups=strategy.num_groups,
            ),
            sample_rate=sampling_prob_cp,
        ),
        "cyclic_poisson",
    )
    print(f"cyclic_poisson: σ≈{nm_cp:.4f}, achieved ε≈{eps_cp:.3f}, time {t_cp:.2f}s")

    nm_bms, eps_bms, t_bms = calibrate_one(
        lambda nm: acc.b_min_sep(
            acc.band_mf(
                nm,
                sensitivity=strategy.sensitivity,
                num_groups=strategy.num_groups,
            ),
            strategy_coefficients=coef,
            n_steps=n_steps,
            participation_rate_p0=p0,
            num_mc_samples=50_000,
            mc_seed=42,
        ),
        "b_min_sep",
    )
    print(f"b_min_sep (MC 50k): σ≈{nm_bms:.4f}, achieved ε≈{eps_bms:.3f}, time {t_bms:.2f}s")

    if nm_cp > 0:
        print(f"\n  Ratio σ_bms / σ_cp ≈ {nm_bms / nm_cp:.3f} (<1 means less noise at same ε)")

    # Reference: unamplified MF (single-shot Gaussian), not comparable apples-to-apples
    nm_mf = calibrate_one(
        lambda nm: acc.band_mf(
            nm, sensitivity=strategy.sensitivity, num_groups=strategy.num_groups
        ),
        "mf_only",
    )[0]
    print(f"\n  (reference) BandMF without subsampling σ≈{nm_mf:.4f}")


if __name__ == "__main__":
    main()
