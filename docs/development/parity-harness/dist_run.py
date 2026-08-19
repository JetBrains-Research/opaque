"""Distributional-equivalence harness: gaussian + MF noise, adafactor probe.

Runs on both sides (auto-detect); dumps samples for statistical comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import traceback

import torch

IS_BRANCH = importlib.util.find_spec("opaque.api.torch") is not None
RESULTS: dict = {"side": "branch" if IS_BRANCH else "base"}
ERRORS: dict = {}

if IS_BRANCH:
    from opaque.api.engine.backend import clear_backend, ensure_backend

    clear_backend()
    ensure_backend(torch.empty(0))

from opaque.random import fold_in, key  # noqa: E402
from opaque.types import clipped  # noqa: E402


def section(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
            print(f"[ok] {name}")
        except Exception:
            ERRORS[name] = traceback.format_exc()
            print(f"[FAIL] {name}\n{ERRORS[name]}")
        return fn

    return deco


def leaves_of(obj):
    out = []

    def rec(x):
        if torch.is_tensor(x):
            out.append(x.detach().clone().cpu())
        elif isinstance(x, dict):
            for k in sorted(x):
                rec(x[k])
        elif isinstance(x, (tuple, list)):
            for e in x:
                rec(e)
        elif hasattr(x, "pytree"):
            rec(x.pytree)

    rec(obj)
    return out


@section("gaussian_samples")
def _gauss():
    from opaque.dpsgd.noise import gaussian_noise

    zero = torch.zeros(1000)
    cp = clipped((zero,), max_norm=1.0)
    noise_fn, st = gaussian_noise(noise_multiplier=1.3, key=key(777))
    draws = []
    for _ in range(50):
        noised, st = noise_fn(cp, st)
        draws.append(leaves_of(noised)[0])
    return torch.stack(draws)  # 50 x 1000, expected N(0, (1.3*1.0)^2) iid


@section("mf_streams")
def _mf():
    from opaque.dpftrl.noise import mf_gaussian_noise

    mod = __import__("opaque.dpftrl", fromlist=["band_mf_strategy"])
    strat = mod.band_mf_strategy(bands=4)
    zero = torch.zeros(16)
    cp = clipped((zero,), max_norm=1.0)
    streams = []
    for r in range(300):
        noise_fn, st = mf_gaussian_noise(
            (zero,), strat, n_steps=8, noise_multiplier=1.0,
            key=fold_in(key(1000), r),
        )
        steps = []
        for _ in range(8):
            noised, st = noise_fn(cp, st)
            steps.append(leaves_of(noised)[0])
        streams.append(torch.stack(steps))
    return torch.stack(streams)  # 300 x 8 x 16


@section("adafactor_probe")
def _adafactor():
    import opaque.optimizers as O

    if not IS_BRANCH:
        import torchopt

    gp = torch.Generator().manual_seed(3)
    shapes = [(10, 5), (5,)]
    init_params = tuple(torch.randn(*s, generator=gp) for s in shapes)

    def run(wd):
        g = torch.Generator().manual_seed(17)
        params = tuple(p.clone() for p in init_params)
        traj = []
        opt = step_fn = None
        if IS_BRANCH:
            step_fn, st = O.adafactor(params, lr=0.01, weight_decay=wd)
        else:
            # noinspection PyArgumentList
            opt = O.adafactor(lr=0.01, weight_decay=wd)
            st = opt.init(params)
        for _ in range(10):
            grads = tuple(torch.randn(*s, generator=g) * 0.1 for s in shapes)
            if IS_BRANCH:
                updates, st = step_fn(grads, st, params=params)
                params = O.apply_updates(params, updates)
            else:
                updates, st = opt.update(grads, st, params=params)
                params = torchopt.apply_updates(params, updates)
            traj.append([p.detach().clone() for p in params])
        return traj

    return {"wd0": run(0.0), "wd01": run(0.01)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RESULTS["errors"] = ERRORS
    torch.save(RESULTS, args.out)
    print("sections:", [k for k in RESULTS if k not in ("side", "errors")], "errors:", list(ERRORS))


if __name__ == "__main__":
    main()
