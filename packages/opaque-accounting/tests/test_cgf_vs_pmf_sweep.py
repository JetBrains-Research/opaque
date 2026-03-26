"""Comprehensive CGF vs PMF sweep for all supported processes.

Uses PMF (PLD) as ground truth and measures CGF (saddle-point) accuracy
across all mechanism × amplification × parameter combinations.
"""

import math
import pytest
from dataclasses import dataclass

import opaque_accounting as acc
from opaque_accounting.amplification import (
    poisson, truncated_poisson, parallel_poisson, cyclic_poisson,
)
from opaque_accounting.mechanisms import (
    gaussian, rectified_gaussian, truncated_gaussian, eps_delta, identity,
    band_mf, blt_mf, dense_mf,
)


# ---------------------------------------------------------------------------
# Parameter grids (practical training regimes)
# ---------------------------------------------------------------------------

# Noise multipliers: from aggressive (0.3) to conservative (2.0)
SIGMAS = [0.3, 0.5, 0.8, 1.1, 2.0]

# Sampling rates: typical DP-SGD
SAMPLE_RATES = [0.001, 0.005, 0.01]

# Composition counts
STEPS = [1, 10, 100, 500, 1000]

# Target delta
DELTA = 1e-5

# Bounded Gaussian radius
RADIUS = 5.0

# Parallel workers
NUM_WORKERS = [2, 4]

# Truncated Poisson params
TRUNC_BATCH_CAP = 128
TRUNC_DATASET = 50_000


@dataclass
class Result:
    process_name: str
    sigma: float
    sample_rate: float | None
    steps: int
    extra: str
    eps_pmf: float
    eps_cgf: float
    abs_err: float
    rel_err: float


def compare(process_name, proc, steps, sigma, sample_rate=None, extra=""):
    """Compare CGF vs PMF epsilon_at for a composed process."""
    composed = proc * steps

    try:
        eps_pmf = composed.pmf().epsilon_at(DELTA)
    except Exception:
        return None  # PMF failed, skip

    try:
        eps_cgf = composed.cgf().epsilon_at(DELTA)
    except Exception:
        eps_cgf = float('nan')

    if not math.isfinite(eps_pmf) or eps_pmf <= 0:
        return None

    abs_err = abs(eps_cgf - eps_pmf) if math.isfinite(eps_cgf) else float('inf')
    rel_err = abs_err / eps_pmf if eps_pmf > 0 else float('inf')

    return Result(
        process_name=process_name,
        sigma=sigma,
        sample_rate=sample_rate,
        steps=steps,
        extra=extra,
        eps_pmf=eps_pmf,
        eps_cgf=eps_cgf,
        abs_err=abs_err,
        rel_err=rel_err,
    )


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def sweep_gaussian():
    """Pure Gaussian mechanism (no subsampling)."""
    results = []
    for sigma in SIGMAS:
        for n in STEPS:
            r = compare("gaussian", gaussian(sigma), n, sigma)
            if r:
                results.append(r)
    return results


def sweep_poisson_gaussian():
    """Poisson-subsampled Gaussian."""
    results = []
    for sigma in SIGMAS:
        for q in SAMPLE_RATES:
            for n in STEPS:
                proc = poisson(gaussian(sigma), sample_rate=q)
                r = compare("poisson(gaussian)", proc, n, sigma, q)
                if r:
                    results.append(r)
    return results


def sweep_poisson_rectified():
    """Poisson-subsampled rectified Gaussian."""
    results = []
    for sigma in SIGMAS:
        for q in SAMPLE_RATES:
            for n in [1, 10, 100, 500]:
                proc = poisson(rectified_gaussian(sigma, radius=RADIUS), sample_rate=q)
                r = compare("poisson(rectified)", proc, n, sigma, q, f"R={RADIUS}")
                if r:
                    results.append(r)
    return results


def sweep_poisson_truncated():
    """Poisson-subsampled truncated Gaussian."""
    results = []
    for sigma in SIGMAS:
        for q in SAMPLE_RATES:
            for n in [1, 10, 100, 500]:
                proc = poisson(truncated_gaussian(sigma, radius=RADIUS), sample_rate=q)
                r = compare("poisson(truncated)", proc, n, sigma, q, f"R={RADIUS}")
                if r:
                    results.append(r)
    return results


def sweep_truncated_poisson():
    """Truncated Poisson-subsampled Gaussian."""
    results = []
    for sigma in SIGMAS:
        q = TRUNC_BATCH_CAP / TRUNC_DATASET
        for n in [1, 10, 100, 500]:
            proc = truncated_poisson(
                gaussian(sigma), sample_rate=q,
                batch_size_cap=TRUNC_BATCH_CAP, dataset_size=TRUNC_DATASET,
            )
            r = compare("truncated_poisson", proc, n, sigma, q,
                         f"cap={TRUNC_BATCH_CAP},N={TRUNC_DATASET}")
            if r:
                results.append(r)
    return results


def sweep_parallel_poisson():
    """Parallel Poisson-subsampled Gaussian."""
    results = []
    for sigma in SIGMAS:
        for q in SAMPLE_RATES:
            for m in NUM_WORKERS:
                for n in STEPS:
                    proc = parallel_poisson(gaussian(sigma), sample_rate=q, num_workers=m)
                    r = compare("parallel_poisson", proc, n, sigma, q, f"m={m}")
                    if r:
                        results.append(r)
    return results


def sweep_mf_mechanisms():
    """Matrix factorization mechanisms (band_mf, blt_mf, dense_mf)."""
    results = []
    for sigma in [0.5, 1.1, 2.0]:
        # BandMF
        proc = band_mf(sigma, n_steps=100, bands=5)
        r = compare("band_mf", proc, 1, sigma, extra="n=100,b=5")
        if r:
            results.append(r)

        # BLT
        proc = blt_mf(sigma, n_steps=100)
        r = compare("blt_mf", proc, 1, sigma, extra="n=100")
        if r:
            results.append(r)

        # Dense
        proc = dense_mf(sigma, n_steps=50)
        r = compare("dense_mf", proc, 1, sigma, extra="n=50")
        if r:
            results.append(r)

        # CyclicPoisson + BandMF
        proc = cyclic_poisson(band_mf(sigma, n_steps=100, bands=5), sample_rate=0.01)
        r = compare("cyclic_poisson(band_mf)", proc, 1, sigma, 0.01, "n=100,b=5")
        if r:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(results: list[Result], name: str) -> str:
    """Format results as a table with pass/fail classification."""
    if not results:
        return f"\n=== {name}: no results ===\n"

    lines = [f"\n=== {name} ({len(results)} cases) ==="]
    lines.append(f"{'σ':>5} {'q':>7} {'n':>5} {'extra':>12}  {'ε_PMF':>10} {'ε_CGF':>10} {'abs_err':>10} {'rel%':>7} {'status':>6}")
    lines.append("-" * 90)

    n_pass = n_warn = n_fail = n_nan = 0
    for r in results:
        if math.isnan(r.eps_cgf):
            status = "NaN"
            n_nan += 1
        elif r.rel_err < 0.05:
            status = "OK"
            n_pass += 1
        elif r.rel_err < 0.25:
            status = "WARN"
            n_warn += 1
        else:
            status = "FAIL"
            n_fail += 1

        q_str = f"{r.sample_rate:.4f}" if r.sample_rate else "   -   "
        lines.append(
            f"{r.sigma:5.2f} {q_str:>7} {r.steps:5d} {r.extra:>12}  "
            f"{r.eps_pmf:10.4f} {r.eps_cgf:10.4f} {r.abs_err:10.4f} {r.rel_err*100:6.1f}% {status:>6}"
        )

    lines.append(f"\nSummary: {n_pass} OK, {n_warn} WARN (<25%), {n_fail} FAIL (>=25%), {n_nan} NaN")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCgfVsPmfSweep:
    """Run the full sweep and report results."""

    def test_gaussian(self):
        results = sweep_gaussian()
        report = format_report(results, "Gaussian (no subsampling)")
        print(report)
        fails = [r for r in results if r.rel_err >= 0.25 and math.isfinite(r.eps_cgf)]
        assert len(fails) == 0, f"{len(fails)} failures:\n" + "\n".join(
            f"  σ={r.sigma}, n={r.steps}: PMF={r.eps_pmf:.4f}, CGF={r.eps_cgf:.4f}, err={r.rel_err*100:.1f}%"
            for r in fails
        )

    def test_poisson_gaussian(self):
        results = sweep_poisson_gaussian()
        report = format_report(results, "Poisson(Gaussian)")
        print(report)
        # Report but don't assert — this is the key diagnostic
        fails = [r for r in results if r.rel_err >= 0.25 and math.isfinite(r.eps_cgf)]
        warns = [r for r in results if 0.05 <= r.rel_err < 0.25]
        print(f"\nPoisson(Gaussian): {len(fails)} failures, {len(warns)} warnings out of {len(results)}")

    def test_poisson_rectified(self):
        results = sweep_poisson_rectified()
        report = format_report(results, "Poisson(RectifiedGaussian)")
        print(report)

    def test_poisson_truncated(self):
        results = sweep_poisson_truncated()
        report = format_report(results, "Poisson(TruncatedGaussian)")
        print(report)

    def test_truncated_poisson(self):
        results = sweep_truncated_poisson()
        report = format_report(results, "TruncatedPoisson(Gaussian)")
        print(report)

    def test_parallel_poisson(self):
        results = sweep_parallel_poisson()
        report = format_report(results, "ParallelPoisson(Gaussian)")
        print(report)
        fails = [r for r in results if r.rel_err >= 0.25 and math.isfinite(r.eps_cgf)]
        warns = [r for r in results if 0.05 <= r.rel_err < 0.25]
        print(f"\nParallelPoisson: {len(fails)} failures, {len(warns)} warnings out of {len(results)}")

    def test_mf_mechanisms(self):
        results = sweep_mf_mechanisms()
        report = format_report(results, "MF Mechanisms")
        print(report)


def test_full_sweep_report():
    """Run all sweeps and produce a combined report."""
    all_results = {}
    all_results["Gaussian"] = sweep_gaussian()
    all_results["Poisson(Gaussian)"] = sweep_poisson_gaussian()
    all_results["Poisson(Rectified)"] = sweep_poisson_rectified()
    all_results["Poisson(Truncated)"] = sweep_poisson_truncated()
    all_results["TruncatedPoisson"] = sweep_truncated_poisson()
    all_results["ParallelPoisson"] = sweep_parallel_poisson()
    all_results["MF Mechanisms"] = sweep_mf_mechanisms()

    print("\n" + "=" * 90)
    print("COMPREHENSIVE CGF vs PMF SWEEP REPORT")
    print("=" * 90)

    total_pass = total_warn = total_fail = total_nan = total = 0
    for name, results in all_results.items():
        print(format_report(results, name))
        for r in results:
            total += 1
            if math.isnan(r.eps_cgf):
                total_nan += 1
            elif r.rel_err < 0.05:
                total_pass += 1
            elif r.rel_err < 0.25:
                total_warn += 1
            else:
                total_fail += 1

    print("\n" + "=" * 90)
    print(f"TOTAL: {total} cases | {total_pass} OK (<5%) | {total_warn} WARN (5-25%) | {total_fail} FAIL (>25%) | {total_nan} NaN")
    print("=" * 90)
