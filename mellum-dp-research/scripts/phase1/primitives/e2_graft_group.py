"""E2: graft a per-example NON-gradient statistic (router-load vector) into the clipped
pytree as its own clipping group, using only existing Opaque primitives.

Route A: extra differentiated argument z (argnums=(0,1)); grad wrt z == load vector.
Route B: probe parameter inside the trainable dict ('router_load_probe'); per_group() keyed by name.
Both: PerGroup clipping (grad: C, load: lam) -> gaussian_noise per-group sigma -> mf_gaussian_noise latch.
"""
import math, torch
torch.set_num_threads(2); torch.manual_seed(0)
from opaque.types import PerGroup, ClippedPytree, NoisedPytree
from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy, identity_strategy
from opaque.random import key
from opaque.pytree import tree_flatten_with_paths

E, D, n = 4, 8, 6           # experts, feature dim, batch
C, lam, nm = 1.0, 0.5, 1.0
x = torch.randn(n, D); y = torch.randn(n)
params = {"w": torch.randn(D), "router": torch.randn(E, D)}

def load_vector(params, xi):
    # per-example "router load": top-2 one-hot counts (argmax-derived, zero grad) + mean prob (differentiable)
    logits = xi @ params["router"].T                      # (E,)
    p = torch.softmax(logits.float(), -1)
    top = torch.topk(p, 2).indices
    f = (top[..., None] == torch.arange(E)).sum(0).float()  # counts per expert (vmap-safe one-hot)
    return f, p

# ---------- Route A: extra argnum ----------
def loss_A(params, z, xi, yi):
    pred = xi @ params["w"]
    f, p = load_vector(params, xi)
    return 0.5 * (pred - yi) ** 2 + (z * f.detach()).sum()   # d/dz = f

z = torch.zeros(E)
paths = [("0", "w"), ("0", "router"), ("1",)]
pgA = PerGroup(groups={(0, "w"): "grad", (0, "router"): "grad", (1,): "load"}, values={"grad": C, "load": lam})
gfA, stA = clipped_grad(loss_A, argnums=(0, 1), batch_argnums=(2, 3), clipping_norm=pgA, normalize_by=n, return_aux=True)
(outA, auxA), stA = gfA(params, z, x, y, state=stA)
print("Route A pytree structure:", [p for p, _, in zip(*tree_flatten_with_paths(outA.pytree)[:2])])
# reference: per-example loads clipped to lam then averaged
F = torch.stack([load_vector(params, x[i])[0] for i in range(n)])
Fc = F * torch.clamp(lam / F.norm(dim=1, keepdim=True), max=1.0)
print("  load leaf == mean of per-example clipped loads:", torch.allclose(outA.pytree[1], Fc.mean(0), atol=1e-2), outA.pytree[1], Fc.mean(0))
print("  max_norm:", outA.max_norm.values, " effective(sqrt sum sq):", outA.max_norm.effective, " group_norms keys:", list(auxA.group_norms))
nfA, nsA = gaussian_noise(noise_multiplier=nm, key=key(1))
noisyA, nsA = nfA(outA, nsA)
sd = noisyA.noise_stddev
print("  gaussian_noise PerGroup sigma:", dict(sd.values),
      " expected grad:", nm * math.sqrt((C / n) * ((C + lam) / n)), " load:", nm * math.sqrt((lam / n) * ((C + lam) / n)))
print("  isotropic alt:", outA.noise_stddev_for(noise_multiplier=nm, allocation="isotropic"))

# ---------- Route B: probe parameter in the dict ----------
paramsB = dict(params, router_load_probe=torch.zeros(E))
def loss_B(params, xi, yi):
    pred = xi @ params["w"]
    f, p = load_vector(params, xi)
    return 0.5 * (pred - yi) ** 2 + (params["router_load_probe"] * f.detach()).sum()
pgB = per_group(paramsB, router_load_probe=lam, fallback=C)
print("Route B per_group:", dict(pgB.groups), dict(pgB.values))
gfB, stB = clipped_grad(loss_B, argnums=0, batch_argnums=(1, 2), clipping_norm=pgB, normalize_by=n, return_aux=True)
(outB, auxB), stB = gfB(paramsB, x, y, state=stB)
print("  probe leaf == mean clipped loads:", torch.allclose(outB.pytree["router_load_probe"], Fc.mean(0), atol=1e-2))
print("  other-param grads unaffected by probe (z=0):", torch.allclose(outB.pytree["w"], outA.pytree[0]["w"]))

# ---------- DP-FTRL: mf_gaussian_noise accepts the PerGroup template and latches ----------
for strat in (identity_strategy(), band_mf_strategy(bands=4)):
    mfn, mfs = mf_gaussian_noise(outB.pytree, strat, n_steps=8, noise_multiplier=nm, key=key(2))
    o1, mfs = mfn(outB, mfs); o2, mfs = mfn(outB, mfs)
    print(f"  mf_gaussian_noise({type(strat).__name__}): latched max_norm={mfs._first_max_norm.values}, realized sigma step0={dict(o1.noise_stddev.values)}")
    try:
        bad = ClippedPytree(outB.pytree, max_norm=PerGroup(pgB.groups, {"fallback": C, "router_load_probe": 2 * lam}))
        mfn(bad, mfs)
    except Exception as e:
        print("    varying max_norm rejected:", type(e).__name__)

# ---------- Serialization of a custom EMA state dataclass with a tensor ----------
import dataclasses
from opaque.serialization import state_dict, from_state_dict
@dataclasses.dataclass(frozen=True)
class LoadEmaState:
    ema: torch.Tensor
    step: int
    beta: float
s = LoadEmaState(ema=noisyA.pytree[1].clone(), step=3, beta=0.9)
sd_ = state_dict(s); s2 = from_state_dict(LoadEmaState(ema=torch.zeros(E), step=0, beta=0.9), sd_)
print("serialization round trip:", sorted(sd_), torch.equal(s2.ema, s.ema), s2.step, s2.beta)
