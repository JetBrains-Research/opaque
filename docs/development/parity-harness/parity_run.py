"""Parity harness: runs identical DP workloads on base (2aabb0d) and the
multiplatform branch, dumping tensors for offline comparison.

Usage: uv run python parity_run.py --out <file.pt>
Auto-detects which side it runs on via presence of opaque.api.torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback

import torch
import torch.nn as nn

IS_BRANCH = importlib.util.find_spec("opaque.api.torch") is not None

RESULTS: dict = {"side": "branch" if IS_BRANCH else "base"}
ERRORS: dict = {}


def section(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
            print(f"[ok] {name}")
        except Exception:
            ERRORS[name] = traceback.format_exc()
            print(f"[FAIL] {name}")
            print(ERRORS[name])
        return fn

    return deco


def flatten(obj, out=None):
    if out is None:
        out = []
    if torch.is_tensor(obj):
        out.append(obj.detach().clone().cpu())
    elif isinstance(obj, dict):
        for k in sorted(obj):
            flatten(obj[k], out)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            flatten(x, out)
    elif hasattr(obj, "pytree"):
        flatten(obj.pytree, out)
    return out


if IS_BRANCH:
    # Activate the torch backend explicitly (mirrors conftest fixture).
    from opaque.api.engine.backend import clear_backend, ensure_backend

    clear_backend()
    ensure_backend(torch.empty(0))

from opaque.random import fold_in, key, split  # noqa: E402

if IS_BRANCH:
    from opaque.torch.functional import make_functional
    from opaque.torch.random import generator_from_key
else:
    from opaque.functional import make_functional
    from opaque.random import generator_from_key


# ---------------------------------------------------------------- 1. keys
@section("key_derivation")
def _keys():
    vals = {"key": [key(s).seed for s in (0, 1, 42, 2**63 + 17, 2**64 - 1, -5)]}
    k = key(42)
    vals["fold_int"] = [fold_in(k, i).seed for i in range(6)]
    vals["fold_str"] = [fold_in(k, "noise", 7).seed, fold_in(k, "clip").seed]
    vals["fold_big"] = [fold_in(k, 2**100).seed]
    vals["split"] = [x.seed for x in split(key(7), 4)]
    return vals


# ------------------------------------------------- 2. generator streams
@section("generator_streams")
def _gens():
    out = {}
    for s in (1, 42, 123456789, 2**62 + 3, 2**63 - 2, 2**63 + 5, 2**64 - 7):
        g = generator_from_key(key(s))
        out[str(s)] = torch.randn(4, generator=g)
    # derived keys, as the noise path uses them
    for i in range(4):
        kk = fold_in(key(42), i)
        g = generator_from_key(kk)
        out[f"fold42_{i}"] = torch.randn(4, generator=g)
    return out


# ------------------------------------------------------------- 3. clipping
def build_model_and_batch():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
    fmodel, params = make_functional(model)
    g = torch.Generator().manual_seed(11)
    X = torch.randn(64, 16, generator=g)
    y = torch.randint(0, 4, (64,), generator=g)

    def loss_fn(p, example):
        x, t = example
        logits = fmodel(p, x.unsqueeze(0))
        return torch.nn.functional.cross_entropy(logits, t.unsqueeze(0))

    return fmodel, params, (X, y), loss_fn


GRADS = None
PARAMS = None


@section("clip_fixed")
def _clip():
    global GRADS, PARAMS
    from opaque.dpsgd.clipping import clipped_grad

    fmodel, params, batch, loss_fn = build_model_and_batch()
    PARAMS = params
    grad_fn, st = clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=64
    )
    grads, st = grad_fn(params, batch, state=st)
    GRADS = grads
    return {"leaves": flatten(grads), "sensitivity": float(grads.sensitivity)}


@section("clip_autos")
def _clip_auto():
    from opaque.dpsgd.clipping import auto_clipped_grad

    fmodel, params, batch, loss_fn = build_model_and_batch()
    grad_fn, st = auto_clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, R=1.0, gamma=0.01, normalize_by=64
    )
    grads, st = grad_fn(params, batch, state=st)
    return {"leaves": flatten(grads), "sensitivity": float(grads.sensitivity)}


# ------------------------------------------------- 4. gaussian noise stream
@section("gaussian_noise_stream")
def _noise():
    from opaque.dpsgd.noise import gaussian_noise

    assert GRADS is not None
    base_leaves = flatten(GRADS)
    out = {}
    for kseed in (42, 12345):
        noise_fn, st = gaussian_noise(noise_multiplier=1.0, key=key(kseed))
        steps = []
        for _ in range(8):
            noised, st = noise_fn(GRADS, st)
            leaves = flatten(noised)
            steps.append([n - b for n, b in zip(leaves, base_leaves)])
        out[str(kseed)] = steps
    return out


# ----------------------------------------------------------- 5. optimizers
@section("optimizers")
def _opts():
    import opaque.optimizers as O

    if not IS_BRANCH:
        import torchopt

    gp = torch.Generator().manual_seed(3)
    shapes = [(10, 5), (5,), (5, 3)]
    init_params = tuple(torch.randn(*s, generator=gp) for s in shapes)

    def grad_seq():
        g = torch.Generator().manual_seed(17)
        for _ in range(15):
            yield tuple(torch.randn(*s, generator=g) * 0.1 for s in shapes)

    names = [
        "sgd",
        "adam",
        "adamw",
        "lion",
        "rmsprop",
        "adagrad",
        "adafactor",
        "ademamix",
        "adadelta",
        "radam",
        "schedule_free",
    ]
    out = {}
    for name in names:
        fn = getattr(O, name, None)
        if fn is None:
            out[name] = "MISSING"
            continue
        try:
            kwargs = {"lr": 0.01}
            if name != "sgd":
                kwargs["weight_decay"] = 0.01
            params = tuple(p.clone() for p in init_params)
            traj = []
            if IS_BRANCH:
                step_fn, st = fn(params, **kwargs)
                for grads in grad_seq():
                    updates, st = step_fn(grads, st, params=params)
                    params = O.apply_updates(params, updates)
                    traj.append([p.detach().clone() for p in params])
            else:
                opt = fn(**kwargs)
                st = opt.init(params)
                for grads in grad_seq():
                    updates, st = opt.update(grads, st, params=params)
                    params = torchopt.apply_updates(params, updates)
                    traj.append([p.detach().clone() for p in params])
            out[name] = traj
        except Exception:
            out[name] = "ERROR: " + traceback.format_exc()
    return out


# ------------------------------------------------------------ 6. e2e dpsgd
@section("e2e_dpsgd")
def _e2e():
    from opaque.dpsgd.clipping import clipped_grad
    from opaque.dpsgd.noise import gaussian_noise
    import opaque.optimizers as O

    if not IS_BRANCH:
        import torchopt

    g = torch.Generator().manual_seed(5)
    X = torch.randn(256, 12, generator=g)
    w_true = torch.randn(12, generator=g)
    y = X @ w_true + 0.1 * torch.randn(256, generator=g)

    torch.manual_seed(9)
    model = nn.Linear(12, 1)
    fmodel, params = make_functional(model)

    def loss_fn(p, example):
        x, t = example
        pred = fmodel(p, x.unsqueeze(0)).squeeze()
        return (pred - t) ** 2

    grad_fn, cst = clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=32
    )
    noise_fn, nst = gaussian_noise(noise_multiplier=0.8, key=key(123))

    losses = []
    opt = step_fn = None
    if IS_BRANCH:
        step_fn, ost = O.sgd(params, lr=0.05)
    else:
        # noinspection PyArgumentList
        opt = O.sgd(lr=0.05)
        ost = opt.init(params)

    step = 0
    for epoch in range(5):
        for i in range(0, 256, 32):
            batch = (X[i : i + 32], y[i : i + 32])
            grads, cst = grad_fn(params, batch, state=cst)
            noised, nst = noise_fn(grads, nst)
            if IS_BRANCH:
                updates, ost = step_fn(noised, ost, params=params)
                params = O.apply_updates(params, updates)
            else:
                updates, ost = opt.update(noised, ost, params=params)
                params = torchopt.apply_updates(params, updates)
            with torch.no_grad():
                pred = fmodel(params, X).squeeze()
                losses.append(((pred - y) ** 2).mean().item())
            step += 1
    return {"losses": torch.tensor(losses), "final_params": [p.detach().clone() for p in params]}


# ---------------------------------------------------------- 7. accounting
@section("accounting")
def _acc():
    import opaque.accounting as acc
    import opaque.dpsgd.accounting as dpsgd_acc

    # noinspection PyTypeChecker
    result = acc.calibrate(
        budget=acc.epsilon_budget(3.0, delta=1e-5),
        process=lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.032) * 300,
        param_min=0.1,
        param_max=100.0,
    )
    nm = result.param
    proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.032) * 300
    from opaque.accounting import Accountant

    a = Accountant() | proc
    return {"noise_multiplier": nm, "eps_at_delta": a.epsilon_at(1e-5)}


# ------------------------------------------------------------- 8. dpftrl MF
@section("dpftrl_mf")
def _mf():
    from opaque.dpftrl.noise import mf_gaussian_noise

    strat = None
    errs = []
    for modpath, name, kwargs in [
        ("opaque.dpftrl", "band_mf_strategy", {"bands": 4}),
        ("opaque.dpftrl.strategy", "band_mf_strategy", {"bands": 4}),
        ("opaque.dpftrl.noise", "band_mf_strategy", {"bands": 4}),
    ]:
        try:
            mod = __import__(modpath, fromlist=[name])
            strat = getattr(mod, name)(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{modpath}.{name}: {e}")
    if strat is None:
        raise RuntimeError("no strategy importable: " + "; ".join(errs))

    assert GRADS is not None
    base_leaves = flatten(GRADS)
    noise_fn, st = mf_gaussian_noise(
        GRADS.pytree, strat, n_steps=8, noise_multiplier=1.0, key=key(0)
    )
    steps = []
    for _ in range(8):
        noised, st = noise_fn(GRADS, st)
        leaves = flatten(noised)
        steps.append([n - b for n, b in zip(leaves, base_leaves)])
    return {"steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RESULTS["errors"] = ERRORS
    RESULTS["torch_version"] = torch.__version__
    torch.save(RESULTS, args.out)
    print(json.dumps({"side": RESULTS["side"], "sections_ok": [k for k in RESULTS if k not in ("side", "errors", "torch_version")], "errors": list(ERRORS)}, indent=1))


if __name__ == "__main__":
    main()
