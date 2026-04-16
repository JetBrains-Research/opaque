#!/usr/bin/env python3
"""Wall-clock comparison of MF strategy construction (no training).

Run from repo root::

    python examples/benchmark_mf_strategy_init.py

Compares BandMF (L-BFGS) vs BSR (closed-form) for similar hyperparameters.
"""
from __future__ import annotations

import time

import torch

from opaque.noise.mf import band_mf_strategy, bsr_strategy


def main() -> None:
    n_steps = 2000
    bands = 8
    momentum = 0.95
    bsr_alpha = 1.0
    min_sep = 100
    max_participations = 5

    t0 = time.perf_counter()
    band = band_mf_strategy(
        n_steps=n_steps, bands=bands, momentum=momentum, lr_schedule=None
    )
    t_band = time.perf_counter() - t0

    t0 = time.perf_counter()
    bsr = bsr_strategy(
        bandwidth=bands,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        momentum=momentum,
        alpha=bsr_alpha,
    )
    t_bsr = time.perf_counter() - t0

    print("MF strategy construction benchmark (no training)")
    print(f"  n_steps={n_steps}, bands/bsr_bandwidth={bands}, β={momentum}")
    print(f"  BSR: min_sep={min_sep}, max_participations={max_participations}, α={bsr_alpha}")
    print()
    print(f"  band_mf_strategy: {t_band * 1000:.2f} ms")
    print(f"  bsr_strategy:      {t_bsr * 1000:.2f} ms")
    if t_band > 0:
        print(f"\n  Speedup (BandMF / BSR): {t_band / t_bsr:.1f}×")


if __name__ == "__main__":
    main()
