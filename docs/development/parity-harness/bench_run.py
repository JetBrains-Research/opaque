"""Wall-clock microbenchmark of the DP-SGD step: base vs branch (torch CPU).

Measures per-step time of clip+noise+optimizer on a small MLP (dispatch
overhead dominated) and on a larger MLP (compute dominated).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time

import torch
import torch.nn as nn

IS_BRANCH = importlib.util.find_spec("opaque.api.torch") is not None

if IS_BRANCH:
    from opaque.api.engine.backend import clear_backend, ensure_backend

    clear_backend()
    ensure_backend(torch.empty(0))
    from opaque.torch.functional import make_functional
    import opaque.optimizers as O
else:
    from opaque.functional import make_functional
    import opaque.optimizers as O
    import torchopt

from opaque.dpsgd.clipping import clipped_grad  # noqa: E402
from opaque.dpsgd.noise import gaussian_noise  # noqa: E402
from opaque.random import key  # noqa: E402

torch.set_num_threads(2)


def bench(hidden, batch, steps, warmup=8):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(64, hidden), nn.Tanh(), nn.Linear(hidden, 8))
    fmodel, params = make_functional(model)
    g = torch.Generator().manual_seed(11)
    X = torch.randn(batch, 64, generator=g)
    y = torch.randint(0, 8, (batch,), generator=g)

    def loss_fn(p, ex):
        x, t = ex
        return torch.nn.functional.cross_entropy(fmodel(p, x.unsqueeze(0)), t.unsqueeze(0))

    grad_fn, cst = clipped_grad(loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=batch)
    noise_fn, nst = gaussian_noise(noise_multiplier=1.0, key=key(42))
    opt = step_fn = None
    if IS_BRANCH:
        step_fn, ost = O.adamw(params, lr=1e-3, weight_decay=0.01)
    else:
        # noinspection PyArgumentList
        opt = O.adamw(lr=1e-3, weight_decay=0.01)
        ost = opt.init(params)

    def one_step(params, cst, nst, ost):
        grads, cst = grad_fn(params, (X, y), state=cst)
        noised, nst = noise_fn(grads, nst)
        if IS_BRANCH:
            updates, ost = step_fn(noised, ost, params=params)
            params = O.apply_updates(params, updates)
        else:
            updates, ost = opt.update(noised, ost, params=params)
            params = torchopt.apply_updates(params, updates)
        return params, cst, nst, ost

    for _ in range(warmup):
        params, cst, nst, ost = one_step(params, cst, nst, ost)
    t0 = time.perf_counter()
    for _ in range(steps):
        params, cst, nst, ost = one_step(params, cst, nst, ost)
    dt = (time.perf_counter() - t0) / steps
    return dt * 1000  # ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {
        "side": "branch" if IS_BRANCH else "base",
        "small_ms": bench(hidden=32, batch=32, steps=60),
        "large_ms": bench(hidden=512, batch=64, steps=25),
    }
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(res)


if __name__ == "__main__":
    main()
