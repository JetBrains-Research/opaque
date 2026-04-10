"""Compare calibrated noise multipliers for BandMF vs BLT vs DP-SGD.

Run this before choosing a mechanism to see which one requires the
least noise for your target (ε, δ).  Lower σ = less noise = better utility.

Usage:
    uv run python examples/compare_accounting.py
    uv run python examples/compare_accounting.py --num-train-samples 1000000 --num-epochs 5
"""

import argparse
import time

import opaque.accounting as acc
from opaque.accounting import calibration as cal


def main():
    parser = argparse.ArgumentParser(description="Compare DP-FTRL accounting mechanisms")
    parser.add_argument("--num-train-samples", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--target-epsilon", type=float, default=3.0)
    parser.add_argument("--target-delta", type=float, default=None)
    parser.add_argument("--cal-min", type=float, default=0.1)
    parser.add_argument("--cal-max", type=float, default=50.0)
    args = parser.parse_args()

    N = args.num_train_samples
    B = args.batch_size
    epochs = args.num_epochs
    steps_per_epoch = N // B
    T = steps_per_epoch * epochs
    target_eps = args.target_epsilon
    target_delta = args.target_delta or 1.0 / (N ** 1.1)

    q_poisson = B / N

    print(f"N={N:,}, B={B}, epochs={epochs}")
    print(f"steps/epoch={steps_per_epoch:,}, T={T:,}")
    print(f"target: ε={target_eps}, δ={target_delta:.2e}")
    print()

    results = {}

    # DP-SGD baseline
    print("=" * 60)
    print(f"DP-SGD (Poisson, q={q_poisson:.6f}, T={T})")
    t0 = time.time()
    try:
        r = cal.calibrate(
            cal.epsilon_budget(target_eps, delta=target_delta),
            lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=q_poisson) * T,
            param_min=args.cal_min, param_max=args.cal_max, tolerance=1e-3,
        )
        print(f"  σ = {r.param:.4f}, ε = {r.achieved:.4f}  ({time.time()-t0:.1f}s)")
        results["dpsgd"] = r.param
    except Exception as e:
        print(f"  FAILED: {e}")

    # BandMF at various band counts
    for bands in [8, 16, 32, 64, 128]:
        q_cyclic = B * bands / N
        if q_cyclic > 1.0:
            print(f"\nBandMF (b={bands}): SKIPPED (q_cyclic={q_cyclic:.3f} > 1)")
            continue
        n_groups = -(-T // bands)  # ceil division
        print()
        print(f"BandMF (b={bands}, q_cyclic={q_cyclic:.4f}, groups={n_groups})")
        t0 = time.time()
        try:
            r = cal.calibrate(
                cal.epsilon_budget(target_eps, delta=target_delta),
                lambda nm, _b=bands, _q=q_cyclic: acc.cyclic_poisson(
                    acc.band_mf(nm, n_steps=T, bands=_b),
                    sample_rate=_q,
                ),
                param_min=args.cal_min, param_max=args.cal_max, tolerance=1e-3,
            )
            ratio = r.param / results["dpsgd"] if "dpsgd" in results else float("nan")
            print(f"  σ = {r.param:.4f}, ε = {r.achieved:.4f}  (ratio: {ratio:.2f}×)  ({time.time()-t0:.1f}s)")
            results[f"bandmf_b{bands}"] = r.param
        except Exception as e:
            print(f"  FAILED: {e}")

    # BLT
    print()
    print(f"BLT (min_sep={steps_per_epoch}, max_part={epochs}, no subsampling)")
    t0 = time.time()
    try:
        r = cal.calibrate(
            cal.epsilon_budget(target_eps, delta=target_delta),
            lambda nm: acc.blt_mf(
                nm, n_steps=T,
                min_sep=steps_per_epoch,
                max_participations=epochs,
                max_buffers=10,
            ),
            param_min=args.cal_min, param_max=args.cal_max, tolerance=1e-3,
        )
        ratio = r.param / results["dpsgd"] if "dpsgd" in results else float("nan")
        print(f"  σ = {r.param:.4f}, ε = {r.achieved:.4f}  (ratio: {ratio:.2f}×)  ({time.time()-t0:.1f}s)")
        results["blt"] = r.param
    except Exception as e:
        print(f"  FAILED: {e}")

    # Summary
    if results:
        print()
        print("=" * 60)
        print("SUMMARY (lower σ = less noise = better)")
        print("=" * 60)
        best = min(results, key=results.get)
        for name, sigma in sorted(results.items(), key=lambda x: x[1]):
            marker = " ← best" if name == best else ""
            print(f"  {name:20s}  σ = {sigma:.4f}{marker}")


if __name__ == "__main__":
    main()
