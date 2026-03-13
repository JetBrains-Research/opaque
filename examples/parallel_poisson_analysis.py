"""Compare privacy cost of sharded (Poisson) vs replicated (parallel Poisson) DDP.

Sharded: each rank owns a disjoint shard, standard Poisson accounting.
Parallel: each rank sees the full dataset, parallel_poisson accounting
with per-rank sample_rate = q / num_workers so total expected batch
matches the sharded case.

Usage::

    uv run python examples/parallel_poisson_analysis.py
"""

from __future__ import annotations

import opaque.accounting as acc
from opaque.accounting import calibration as cal


def calibrate_noise(
    target_eps: float,
    delta: float,
    total_steps: int,
    sample_rate: float,
    *,
    num_workers: int | None = None,
    param_min: float = 0.11,
    param_max: float = 1.19,
) -> cal.CalibrateResult:
    """Calibrate noise_multiplier for a given accounting mode.

    When num_workers is None, uses standard Poisson accounting (sharded DDP).
    When set, uses parallel_poisson with the given sample_rate on each worker.
    """
    if num_workers is None:
        process_fn = (
            lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate)
            * total_steps
        )
    else:
        process_fn = (
            lambda nm: acc.parallel_poisson(
                acc.gaussian(nm),
                sample_rate=sample_rate,
                num_workers=num_workers,
            )
            * total_steps
        )

    budget = cal.epsilon_budget(target_eps, delta=delta)
    return cal.calibrate(budget, process_fn, param_min=param_min, param_max=param_max)


def mellum_comparison() -> None:
    """Compare noise_multiplier for the mellum-kstack preset parameters."""
    n = 50_000
    batch_size = 128
    epochs = 3
    target_eps = 10.0

    sample_rate = batch_size / n
    total_steps = int(epochs / sample_rate)
    delta = 1.0 / (n**1.1)

    print("=" * 72)
    print("Mellum-kstack preset: sharded vs parallel Poisson")
    print("=" * 72)
    print(f"  n={n}, batch_size={batch_size}, epochs={epochs}")
    print(f"  sample_rate={sample_rate:.6f}, total_steps={total_steps}")
    print(f"  target ε={target_eps}, δ={delta:.2e}")
    print()

    header = f"{'mode':<22} {'m':>3} {'q_per_rank':>12} {'noise_mult':>12} {'achieved_ε':>12}"
    print(header)
    print("-" * len(header))

    baseline = calibrate_noise(target_eps, delta, total_steps, sample_rate)
    print(
        f"{'sharded (poisson)':<22} {'1':>3} {sample_rate:>12.6f} "
        f"{baseline.param:>12.6f} {baseline.achieved:>12.6f}"
    )

    for m in [2, 4, 8]:
        per_rank_rate = sample_rate / m
        result = calibrate_noise(
            target_eps, delta, total_steps, per_rank_rate, num_workers=m
        )
        penalty_pct = (result.param - baseline.param) / baseline.param * 100
        print(
            f"{'parallel_poisson':<22} {m:>3} {per_rank_rate:>12.6f} "
            f"{result.param:>12.6f} {result.achieved:>12.6f}  "
            f"(+{penalty_pct:.2f}% noise)"
        )


def sweep_sample_rates() -> None:
    """Sweep q to show where the parallel_poisson penalty becomes significant."""
    n = 50_000
    epochs = 3
    target_eps = 10.0
    delta = 1.0 / (n**1.1)
    m = 4

    print()
    print("=" * 72)
    print(f"Sweep: noise_multiplier penalty of parallel_poisson (m={m}) vs sharded")
    print("=" * 72)
    print(f"  n={n}, epochs={epochs}, target ε={target_eps}, δ={delta:.2e}")
    print()

    header = (
        f"{'q':>10} {'batch':>8} {'steps':>8} "
        f"{'σ_sharded':>12} {'σ_parallel':>12} {'penalty%':>10}"
    )
    print(header)
    print("-" * len(header))

    for batch_size in [32, 64, 128, 256, 512, 1024, 2048, 5000]:
        q = batch_size / n
        if q > 1.0:
            continue
        total_steps = int(epochs / q)

        try:
            sharded = calibrate_noise(target_eps, delta, total_steps, q)
            parallel = calibrate_noise(
                target_eps, delta, total_steps, q / m, num_workers=m
            )
            penalty_pct = (parallel.param - sharded.param) / sharded.param * 100
            print(
                f"{q:>10.6f} {batch_size:>8} {total_steps:>8} "
                f"{sharded.param:>12.6f} {parallel.param:>12.6f} "
                f"{penalty_pct:>+10.4f}%"
            )
        except ValueError as e:
            print(
                f"{q:>10.6f} {batch_size:>8} {total_steps:>8}  "
                f"{'skipped':>12}  {str(e)[:40]}"
            )


def epsilon_at_fixed_sigma() -> None:
    """For a fixed noise_multiplier, show ε gap between sharded and parallel."""
    n = 50_000
    batch_size = 128
    epochs = 3
    delta = 1.0 / (n**1.1)

    q = batch_size / n
    total_steps = int(epochs / q)
    sigma = 0.8

    print()
    print("=" * 72)
    print(f"Fixed σ={sigma}: epsilon comparison")
    print("=" * 72)
    print(f"  n={n}, batch_size={batch_size}, q={q:.6f}, steps={total_steps}")
    print()

    header = f"{'mode':<22} {'m':>3} {'ε':>12}"
    print(header)
    print("-" * len(header))

    eps_sharded = (
        acc.poisson(acc.gaussian(sigma), sample_rate=q) * total_steps
    ).epsilon_at(delta)
    print(f"{'sharded (poisson)':<22} {'1':>3} {eps_sharded:>12.6f}")

    for m in [2, 4, 8]:
        per_rank_rate = q / m
        eps_parallel = (
            acc.parallel_poisson(
                acc.gaussian(sigma), sample_rate=per_rank_rate, num_workers=m
            )
            * total_steps
        ).epsilon_at(delta)
        gap_pct = (eps_parallel - eps_sharded) / eps_sharded * 100
        print(
            f"{'parallel_poisson':<22} {m:>3} {eps_parallel:>12.6f}  "
            f"(+{gap_pct:.2f}%)"
        )


if __name__ == "__main__":
    mellum_comparison()
    epsilon_at_fixed_sigma()
    sweep_sample_rates()
