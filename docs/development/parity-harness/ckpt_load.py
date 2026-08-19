"""Attempt to load a base-version checkpoint on the branch (run on branch venv)."""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn as nn

from opaque.api.engine.backend import clear_backend, ensure_backend

clear_backend()
ensure_backend(torch.empty(0))

from opaque.dpsgd.clipping import clipped_grad  # noqa: E402
from opaque.dpsgd.noise import gaussian_noise  # noqa: E402
from opaque.dpftrl.noise import mf_gaussian_noise  # noqa: E402
from opaque.dpftrl import band_mf_strategy  # noqa: E402
from opaque.torch.functional import make_functional  # noqa: E402
from opaque.random import key  # noqa: E402
from opaque.serialization import from_state_dict, state_dict  # noqa: E402
import opaque.optimizers as O  # noqa: E402

sd = torch.load(sys.argv[1], weights_only=False)
print(f"loaded base checkpoint with {len(sd)} keys")

torch.manual_seed(9)
model = nn.Linear(12, 1)
fmodel, params = make_functional(model)


def loss_fn(p, ex):
    x, t = ex
    return (fmodel(p, x.unsqueeze(0)).squeeze() - t) ** 2


grad_fn, cst = clipped_grad(loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=32)
noise_fn, nst = gaussian_noise(noise_multiplier=0.8, key=key(123))
mf_fn, mfst = mf_gaussian_noise(params, band_mf_strategy(bands=4), n_steps=8, noise_multiplier=1.0, key=key(0))
step_fn, ost = O.adamw(params, lr=0.01, weight_decay=0.01)

template = {
    "params": params,
    "opt_state": ost,
    "clip_state": cst,
    "noise_state": nst,
    "mf_state": mfst,
}
print("branch template keys:", sorted(state_dict(template)))

results = {}
for name in ("params", "opt_state", "clip_state", "noise_state", "mf_state"):
    sub = {k[len(name) + 1 :]: v for k, v in sd.items() if k.startswith(name + ".") or k.startswith(name + "[")}
    # keep original composite key prefixing: rebuild via full-dict restore of one field
    try:
        restored = from_state_dict({name: template[name]}, {k: v for k, v in sd.items() if k.startswith(name)})
        # count how many checkpoint values actually landed vs template kept
        results[name] = "RESTORED"
    except Exception as e:  # noqa: BLE001
        results[name] = f"FAILED: {type(e).__name__}: {e}"

for k, v in results.items():
    print(f"  {k}: {v}")

# Full restore + numeric spot-check on params
try:
    restored = from_state_dict(template, sd)
    p0 = restored["params"][0]
    print("full restore OK; params[0][0,:3] =", p0[0, :3].tolist())
    # verify params actually took checkpoint values, not template values
    same_as_template = torch.equal(p0, params[0])
    print("params identical to fresh template (BAD if True):", same_as_template)
    # resume noise stream one step
    grads_dummy, cst2 = grad_fn(params, (torch.randn(32, 12), torch.randn(32)), state=restored["clip_state"])
    noised, nst2 = noise_fn(grads_dummy, restored["noise_state"])
    print("resumed gaussian noise step from restored state: OK, step_counter ->", getattr(nst2, "_step_counter", "?"))
    mfnoised, mfst2 = mf_fn(grads_dummy, restored["mf_state"])
    print("resumed MF noise step from restored state: OK, step_counter ->", getattr(mfst2, "_step_counter", "?"))
except Exception:
    print("FULL RESTORE/RESUME FAILED:")
    traceback.print_exc()
