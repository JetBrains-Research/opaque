import math, torch
from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.api.engine.clipping._pytree import clip_pytree
E,k,L,T=64,8,4,16
Dh=math.sqrt(k*(1-k/E)); lam=0.02*0.9/Dh; Ch=lam*Dh*(1+1e-6)
torch.manual_seed(0)
rnd = torch.stack([torch.stack([torch.randperm(E)[:k] for _ in range(T)]) for _ in range(L)])
def h_of(routes, mask):
    onehot=(routes[...,None]==torch.arange(E)).float().sum(-2); w=mask.float()[None,:,None]
    return (onehot*w).sum((0,1))/(L*mask.float().sum())
def loss(params, batch):
    d=h_of(batch["routes"],batch["mask"])-k/E
    return 0.5*(params["w"]**2).sum()*batch["s"] + (params["router_load_probe"]*(lam*d).detach()).sum()
params={"w": torch.tensor([1.0,2.0,2.0]), "router_load_probe": torch.zeros(E)}
pg=per_group(params, router_load_probe=Ch, fallback=0.9)
# 1) real clipper on an explicit NaN probe leaf
nanleaf={"w": torch.tensor([0.3,0.,0.]), "router_load_probe": torch.full((E,), float("nan"))}
c,aux=clip_pytree(nanleaf, pg)
print("[clip_pytree NaN leaf] group norm probe =", aux.group_norms["router_load_probe"].item(), "| total norm =", aux.norm.item(), "| clipped probe finite?", bool(torch.isfinite(c["router_load_probe"]).all()), c["router_load_probe"][:2].tolist(), "| w ->", c["w"].tolist())
# 2) single all-masked example through clipped_grad
b={"routes": rnd[None], "mask": torch.zeros(1,T), "s": torch.tensor([0.1])}
gf,st=clipped_grad(loss, clipping_norm=pg, normalize_by=256.0, return_aux=True)
(out,a),_=gf(params,b,state=st)
print("[clipped_grad 1 all-masked example] group_norms =", {kk:v.tolist() for kk,v in a.group_norms.items()}, "| grad_norms =", a.grad_norms.tolist(), "| released probe finite?", bool(torch.isfinite(out.pytree["router_load_probe"]).all()), out.pytree["router_load_probe"][:2].tolist(), "| released w =", out.pytree["w"].tolist())
# 3) batch: NaN example + normal example -> does the NaN example's *w* gradient also vanish (whole example zeroed)?
b2={"routes": torch.stack([rnd,rnd]), "mask": torch.stack([torch.zeros(T), torch.ones(T)]), "s": torch.tensor([0.1,0.1])}
(out2,a2),_=gf(params,b2,state=st)
print("[2 examples] released w =", out2.pytree["w"].tolist(), "(expected 2 examples x 0.1*[1,2,2]/256 =", (2*0.1*torch.tensor([1.,2.,2.])/256).tolist(), "; if the NaN example is dropped entirely:", (0.1*torch.tensor([1.,2.,2.])/256).tolist(), ")")
