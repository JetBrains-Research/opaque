import math, torch
from torch.func import grad, vmap
E,k,L,T=64,8,4,16
Dh=math.sqrt(k*(1-k/E)); lam=0.02*0.9/Dh
torch.manual_seed(0)
rnd = torch.stack([torch.stack([torch.randperm(E)[:k] for _ in range(T)]) for _ in range(L)])
def h_of(routes, mask):
    onehot=(routes[...,None]==torch.arange(E)).float().sum(-2); w=mask.float()[None,:,None]
    return (onehot*w).sum((0,1))/(L*mask.float().sum())
def loss(z, routes, mask):
    d=h_of(routes,mask)-k/E
    return (z*(lam*d).detach()).sum()
z=torch.zeros(E)
g0=grad(loss)(z, rnd, torch.zeros(T)); print("[A7 direct] T_x=0 eager grad wrt probe: finite?", bool(torch.isfinite(g0).all()), "nan count", int(torch.isnan(g0).sum()), "first entries", g0[:3].tolist())
gv=vmap(grad(loss), in_dims=(None,0,0))(z, torch.stack([rnd,rnd]), torch.stack([torch.zeros(T), torch.ones(T)]))
print("[A7 vmap] per-example probe grads finite?", torch.isfinite(gv).all(dim=1).tolist(), "nan count row0", int(torch.isnan(gv[0]).sum()))
# what the real clipper does with a NaN leaf
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.types import PerGroup
pg=PerGroup(groups={"router_load_probe":("z",),"fallback":("w",)}, values={"router_load_probe":lam*Dh*(1+1e-6),"fallback":0.9})
clipped, aux = clip_pytree({"z": gv[0], "w": torch.tensor([0.3,0.,0.])}, pg)
print("[A7 clip_pytree on NaN leaf] group norm probe =", aux.group_norms["router_load_probe"].item(), " clipped z finite?", bool(torch.isfinite(clipped["z"]).all()), " z[:3]=", clipped["z"][:3].tolist(), " w=", clipped["w"].tolist())
# --- mixed-sign mask breaks the bound
m=torch.zeros(T); m[0]=1.0; m[1]=-0.5
hm=h_of(rnd,m); print(f"[A6 mixed-sign mask] sum_e h={hm.sum().item():.3f} min h={hm.min().item():.3f} max h={hm.max().item():.3f} ||d||={(hm-k/E).norm().item():.3f} > Dh={Dh:.3f}: {(hm-k/E).norm().item()>Dh}")
# --- topk duplicates / exact ties with fp32 softmax of bf16 logits
logits=torch.zeros(1000,E, dtype=torch.bfloat16)            # all-tie rows
p=torch.softmax(logits, dtype=torch.float, dim=-1); _,idx=torch.topk(p,k,dim=-1)
dups=sum(len(set(r.tolist()))!=k for r in idx); print(f"[ties] all-equal logits: rows with duplicate top-k indices = {dups}/1000; distinct experts per row = {len(set(idx[0].tolist()))}")
