"""refute-sensitivity: concrete neighbouring-dataset attacks on the load-leaf bound, through the REAL clipped_grad + per_group.
CPU, tiny. E=64,k=8; L layers, T tokens synthetic routes (no model needed: the bound is a property of h(x), not of the model)."""
import math, torch
torch.manual_seed(0)
from opaque.dpsgd.clipping import clipped_grad, per_group
E,k=64,8
Cg=0.9; rho=0.02; Dh=math.sqrt(k*(1-k/E)); lam=rho*Cg/Dh; Ch=lam*Dh*(1+1e-6); Bbar=256.0
print(f"lam={lam:.6f} Ch={Ch:.6f} Dh={Dh:.6f}")

def h_of(routes, mask, n_layers=None, dtype=torch.float32):
    # routes (L,T,k) long ; mask (T,) any numeric ; returns pooled load fraction h in R^E
    L = routes.shape[0] if n_layers is None else n_layers
    onehot = (routes[..., None] == torch.arange(E)).to(dtype).sum(-2)   # (L,T,E) in {0,1}
    w = mask.to(dtype)[None, :, None]
    Tx = mask.to(dtype).sum()
    return (onehot * w).sum((0, 1)) / (L * Tx)

def make_loss(mask_key="mask", n_layers=None, dtype=torch.float32, guard=False):
    def loss(params, batch):
        h = h_of(batch["routes"], batch[mask_key], n_layers, dtype)
        if guard:
            Tx = batch[mask_key].to(dtype).sum().clamp(min=1.0)
            L = batch["routes"].shape[0]
            onehot = (batch["routes"][..., None] == torch.arange(E)).to(dtype).sum(-2)
            h = (onehot * batch[mask_key].to(dtype)[None,:,None]).sum((0,1)) / (L*Tx)
        d = (h - k/E).to(torch.float32)
        g_term = 0.5*(params["w"]**2).sum() * batch["s"]          # a dummy 'gradient' leaf, per-example scale s
        return g_term + (params["router_load_probe"] * (lam*d).detach()).sum()
    return loss

L,T=4,16
params={"w": torch.tensor([1.0,2.0,2.0]), "router_load_probe": torch.zeros(E)}
pg = per_group(params, router_load_probe=Ch, fallback=Cg)
print("PerGroup:", pg.values)

def run(batch, loss, label):
    gf, st = clipped_grad(loss, clipping_norm=pg, normalize_by=Bbar, return_aux=True)
    (out, aux), _ = gf(params, batch, state=st)
    gn = aux.group_norms["router_load_probe"]
    probe = out.pytree["router_load_probe"]
    # unclipped reference: lam * sum_x d(x) / Bbar
    B = batch["routes"].shape[0]
    ref = torch.stack([lam*(h_of(batch["routes"][i], batch["mask"][i]) - k/E) for i in range(B)]).sum(0)/Bbar
    print(f"[{label}] probe group norms = {[round(float(v),6) for v in gn]}  bound Ch={Ch:.6f}  max/Ch={float(gn.max())/Ch:.8f}  clipped? {(gn>Ch).tolist()}")
    print(f"      released probe leaf == unclipped sum/Bbar exactly: {torch.equal(probe, ref)}  max|diff|={float((probe-ref).abs().max()):.3e}  finite={bool(torch.isfinite(probe).all())}  aux.grad_norms(total, includes probe)={[round(float(v),5) for v in aux.grad_norms]}")
    return out, aux

same8 = torch.arange(k).view(1,1,k).expand(L,T,k).clone()            # every (l,t) -> experts 0..7  (the structural sup)
disj8 = (torch.arange(k)+k).view(1,1,k).expand(L,T,k).clone()        # disjoint set 8..15
rnd   = torch.stack([torch.stack([torch.randperm(E)[:k] for _ in range(T)]) for _ in range(L)])
ones  = torch.ones(T); one_tok = torch.zeros(T); one_tok[0]=1.0

# --- Attack 1: adversarial example saturating the bound (all tokens same 8 experts), full mask
b1 = {"routes": torch.stack([same8, rnd]), "mask": torch.stack([ones, ones]), "s": torch.tensor([0.1, 0.1])}
run(b1, make_loss(), "A1 sup example + random")
# --- Attack 2: T_x = 1 (one valid token) -> also exactly the sup
b2 = {"routes": torch.stack([rnd, rnd]), "mask": torch.stack([one_tok, ones]), "s": torch.tensor([0.1,0.1])}
run(b2, make_loss(), "A2 T_x=1")
# --- Attack 3: neighbouring pair D vs D+{sup}: difference of the released sums
b3a = {"routes": torch.stack([rnd]), "mask": torch.stack([ones]), "s": torch.tensor([0.1])}
o_a,_ = run(b3a, make_loss(), "A3 D")
o_b,_ = run(b1, make_loss(), "A3 D' = D + sup")
diff = o_b.pytree["router_load_probe"] - o_a.pytree["router_load_probe"]
print(f"      ||M(D')-M(D)|| on probe = {float(diff.norm()):.6e}  vs Ch/Bbar = {Ch/Bbar:.6e}  ratio={float(diff.norm())/(Ch/Bbar):.8f}")
# --- Attack 4: replace-one with disjoint expert sets
dA = h_of(same8, ones)-k/E; dB = h_of(disj8, ones)-k/E
print(f"[A4 replace-one] ||d-d'||={float((dA-dB).norm()):.4f}  sqrt(2k)={math.sqrt(2*k):.4f}  2*Dh={2*Dh:.4f}")
# --- Attack 5: non-binary (segment-id) mask 1,2,3 and float weights: does the structural bound survive?
seg = torch.tensor([1.]*6+[2.]*5+[3.]*5)
h_seg = h_of(same8, seg); print(f"[A5 segment-id mask] sum_e h={float(h_seg.sum()):.6f} max h={float(h_seg.max()):.6f} ||d||={float((h_seg-k/E).norm()):.6f} <= Dh {float((h_seg-k/E).norm())<=Dh}")
# --- Attack 6: additive float mask (0 / -1e9) passed where a 0/1 mask is expected
addm = torch.zeros(T); addm[8:] = -1e9
h_add = h_of(rnd, addm); print(f"[A6 additive mask] sum_e h={float(h_add.sum()):.4e} ||d||={float((h_add-k/E).norm()):.4e} <= Dh? {float((h_add-k/E).norm())<=Dh}  -> BOUND BROKEN (would be clipped; biased release)")
# --- Attack 7: fully-masked row T_x=0 (0/0) through the real clip: NaN propagation?
b7 = {"routes": torch.stack([rnd, rnd]), "mask": torch.stack([torch.zeros(T), ones]), "s": torch.tensor([0.1,0.1])}
run(b7, make_loss(), "A7 T_x=0 unguarded")
run(b7, make_loss(guard=True), "A7 T_x=0 guarded clamp(min=1)")
# --- Attack 8: recorder double-fire (2L captured logits, normalised by the configured L)
b8 = {"routes": torch.stack([torch.cat([same8,same8],0), torch.cat([rnd,rnd],0)]), "mask": torch.stack([ones,ones]), "s": torch.tensor([0.1,0.1])}
run(b8, make_loss(n_layers=L), "A8 2L captures / configured L")
run(b8, make_loss(n_layers=None), "A8 2L captures / actual count")
# --- Attack 9: h computed in bf16 then cast: does the sup example exceed the (1+1e-6) guard?
b9 = {"routes": torch.stack([same8, rnd]), "mask": torch.stack([ones, ones]), "s": torch.tensor([0.1,0.1])}
run(b9, make_loss(dtype=torch.bfloat16), "A9 bf16 h")
# --- Attack 10: the MPS-style roundoff: emulate _guard_scale relative term with fp32 sq accumulation, 113 leaves, widest 65536
from opaque.api.engine.clipping._pytree import _norm_roundoff, _blocked_reduction_terms
ro_cpu = _norm_roundoff(torch.float32, torch.float64, 113, _blocked_reduction_terms(65536))
ro_mps = _norm_roundoff(torch.float32, torch.float32, 113, _blocked_reduction_terms(65536))
eps32 = torch.finfo(torch.float32).eps
print(f"[A10 guard] norm_roundoff cpu/cuda={ro_cpu:.3e} mps={ro_mps:.3e}; _guard_scale relative shrink cpu={2*(eps32/2+ro_cpu):.3e} mps={2*(eps32/2+ro_mps):.3e}; design guard=1e-6 -> guard beats shrink on cpu/cuda: {1e-6>2*(eps32/2+ro_cpu)}, on mps: {1e-6>2*(eps32/2+ro_mps)}")
