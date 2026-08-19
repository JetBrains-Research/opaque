"""Write a base-version checkpoint of DP training state (run on base venv)."""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.dpftrl import band_mf_strategy
from opaque.functional import make_functional
from opaque.random import key
from opaque.serialization import state_dict
import opaque.optimizers as O
import torchopt

torch.manual_seed(9)
model = nn.Linear(12, 1)
fmodel, params = make_functional(model)

g = torch.Generator().manual_seed(5)
X = torch.randn(64, 12, generator=g)
y = X @ torch.randn(12, generator=g)


def loss_fn(p, ex):
    x, t = ex
    return (fmodel(p, x.unsqueeze(0)).squeeze() - t) ** 2


grad_fn, cst = clipped_grad(loss_fn, argnums=0, batch_argnums=1, clipping_norm=1.0, normalize_by=32)
noise_fn, nst = gaussian_noise(noise_multiplier=0.8, key=key(123))
mf_fn, mfst = mf_gaussian_noise(params, band_mf_strategy(bands=4), n_steps=8, noise_multiplier=1.0, key=key(0))

# noinspection PyArgumentList
opt = O.adamw(lr=0.01, weight_decay=0.01)  # pre-split torchopt-style API
ost = opt.init(params)

for i in range(3):
    batch = (X[i * 32 : (i + 1) * 32], y[i * 32 : (i + 1) * 32]) if i < 2 else (X[:32], y[:32])
    grads, cst = grad_fn(params, batch, state=cst)
    noised, nst = noise_fn(grads, nst)
    mfnoised, mfst = mf_fn(grads, mfst)
    updates, ost = opt.update(noised, ost, params=params)
    params = torchopt.apply_updates(params, updates)

ckpt = {
    "params": params,
    "opt_state": ost,
    "clip_state": cst,
    "noise_state": nst,
    "mf_state": mfst,
}
sd = state_dict(ckpt)
torch.save(sd, sys.argv[1])
print("saved keys:", sorted(sd) if isinstance(sd, dict) else type(sd))
