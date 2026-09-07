"""Numerical validation of the derivative identities in phase1-math.md.

Everything is float64, CPU, tiny.  Each check prints a max-abs-diff.
"""
import math, json
import torch
torch.set_num_threads(2)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------
# Verbatim port of transformers 5.16.1 load_balancing_loss_func
# (.venv/.../transformers/models/mellum/modeling_mellum.py L540-606)
# ----------------------------------------------------------------------------
def hf_load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask):
    compute_device = gate_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float64, device=compute_device)
    router_prob_sum = torch.zeros(num_experts, dtype=torch.float64, device=compute_device)
    total_rows = 0.0
    flat_mask = attention_mask.reshape(-1).to(dtype=torch.float64)
    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
            num_experts, dtype=torch.float64).scatter_add_(0, selected_experts.reshape(-1), flat_mask.repeat_interleave(top_k))
        router_prob_sum = router_prob_sum + (routing_weights * flat_mask.unsqueeze(-1)).sum(dim=0)
        total_rows = total_rows + flat_mask.sum()
    tokens_per_expert = tokens_per_expert_sum / total_rows
    router_prob_per_expert = router_prob_sum / total_rows
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts

# ----------------------------------------------------------------------------
# Tiny "router only" model: L layers, hidden H, E experts, top-k.
# Router params W[l] (E,H); layer inputs X[l] (B,T,H) fixed.
# ----------------------------------------------------------------------------
B, T, L, E, K, H = 5, 7, 3, 6, 2, 8

def make(mask_kind):
    X = [torch.randn(B, T, H) for _ in range(L)]
    W = [torch.randn(E, H, requires_grad=True) for _ in range(L)]
    if mask_kind == "full":
        mask = torch.ones(B, T)
    else:
        mask = torch.ones(B, T)
        lens = [7, 5, 3, 6, 2]
        for b, n in enumerate(lens):
            mask[b, n:] = 0.0
    return X, W, mask

def logits_of(X, W):
    return [X[l] @ W[l].T for l in range(L)]  # (B,T,E)

def per_example_stats(logits, mask):
    """P_e(x), f_e(x) pooled over layers with per-example token-mean; T_x."""
    Tx = mask.sum(-1)                                    # (B,)
    P = torch.zeros(B, E); F = torch.zeros(B, E)
    for l in range(L):
        p = torch.softmax(logits[l], dim=-1)             # (B,T,E)
        _, sel = torch.topk(p, K, dim=-1)                # (B,T,K)
        ind = torch.zeros(B, T, E).scatter_(-1, sel, 1.0)
        P = P + (p * mask.unsqueeze(-1)).sum(1)
        F = F + (ind * mask.unsqueeze(-1)).sum(1)
    P = P / (L * Tx).unsqueeze(-1)
    F = F / (L * Tx).unsqueeze(-1)
    return P, F, Tx

def grads(loss, W):
    g = torch.autograd.grad(loss, W, allow_unused=True, retain_graph=True)
    return torch.cat([gi.reshape(-1) for gi in g])

results = {}
for mask_kind in ["full", "ragged"]:
    X, W, mask = make(mask_kind)
    logits = logits_of(X, W)
    gate_logits = tuple(lg.reshape(B * T, E) for lg in logits)
    aux_B = hf_load_balancing_loss_func(gate_logits, E, K, mask)
    G_B = grads(aux_B, W)

    P, F, Tx = per_example_stats(logits, mask)
    Ttot = Tx.sum()
    w = Tx / Ttot                                        # token weights
    F_B = (w.unsqueeze(-1) * F).sum(0).detach()          # f(B) = sum_x (T_x/T_tot) f(x)
    P_B = (w.unsqueeze(-1) * P).sum(0)
    aux_recon = E * (F_B * P_B).sum()
    r = {}
    r["sum_e f_e(B) == k"] = float(F_B.sum())
    r["aux_HF - E*sum f(B) P(B)"] = float((aux_B - aux_recon).detach().abs())

    # (1) surrogate S(x; f~) = E (T_x/T_tot) sum_e f~_e P_e(x), f~ = f(B) detached
    S = E * (w.unsqueeze(-1) * F_B.unsqueeze(0) * P).sum()
    G_S = grads(S, W)
    r["max|grad aux_HF - grad sum_x S(x;f(B))|"] = float((G_B - G_S).abs().max())
    r["|grad aux_HF|"] = float(G_B.norm())

    # per-example DP convention: (1/B) sum_x grad ell_x, ell_x = E sum_e f~_e P_e(x)
    ell = E * (F_B.unsqueeze(0) * P).sum(-1)             # (B,)
    G_eq = grads(ell.sum() / B, W)
    r["max|grad aux_HF - (1/B) sum grad ell_x (equal weights)|"] = float((G_B - G_eq).abs().max())
    # reweighted per-example loss: ell'_x = B (T_x/T_tot) ell_x
    G_rw = grads((B * w * ell).sum() / B, W)
    r["max|grad aux_HF - (1/B) sum grad B(T_x/T_tot) ell_x|"] = float((G_B - G_rw).abs().max())
    # f~ = plain example-mean of f(x) (what an equal-weight pipeline would release)
    F_bar = F.mean(0).detach()
    r["max|f(B) - mean_x f(x)|"] = float((F_B - F_bar).abs().max())

    # (2) per-example own aux: ell_x^own = E sum_e f_e(x) P_e(x); difference identity
    ell_own = E * (F.detach() * P).sum(-1)
    diff_id_max = 0.0
    for b in range(B):
        g_own = grads(ell_own[b], W)
        g_sur = grads(ell[b], W)
        d = (F[b] - F_B).detach()
        g_pred = grads(E * (d * P[b]).sum(), W)
        diff_id_max = max(diff_id_max, float(((g_own - g_sur) - g_pred).abs().max()))
    r["max|(grad own - grad sur) - E sum_e (f_e(x)-f_e(B)) grad P_e(x)|"] = diff_id_max
    G_own = grads(ell_own.sum() / B, W)
    r["rel |grad own-aux mean - grad aux_HF| / |grad aux_HF|"] = float((G_own - G_B).norm() / G_B.norm())
    results[mask_kind] = r

# ----------------------------------------------------------------------------
# token-level logit gradient: d/dz_j sum_e c_e p_e = p_j (c_j - sum_e c_e p_e)
# ----------------------------------------------------------------------------
z = torch.randn(E, requires_grad=True); c = torch.randn(E)
p = torch.softmax(z, 0)
g = torch.autograd.grad((c * p).sum(), z)[0]
pred = p * (c - (c * p).sum())
results["token_logit_grad_identity_maxdiff"] = float((g - pred).detach().abs().max())

# ----------------------------------------------------------------------------
# (2) specialization toy model: 4 doc types, E=8, k=2, each type uses its own
# expert pair -> f(B) exactly uniform (k/E) while f(x) is 1 on its pair.
# ----------------------------------------------------------------------------
E2, K2, L2, T2, H2 = 8, 2, 2, 16, 8
ntypes = 4; Bt = 8
Wt = [torch.randn(E2, H2, requires_grad=True) for _ in range(L2)]
# inputs chosen so that logits strongly favour the doc's own pair
Xt = []
for l in range(L2):
    Xl = torch.zeros(Bt, T2, H2)
    for b in range(Bt):
        typ = b % ntypes
        own = [2 * typ, 2 * typ + 1]
        # solve for x with W x = target logits (least squares)
        target = torch.full((E2,), -3.0); target[own] = 3.0
        target = target + 0.3 * torch.randn(T2, E2)
        Xl[b] = torch.linalg.lstsq(Wt[l].detach(), target.T).solution.T
    Xt.append(Xl)
maskt = torch.ones(Bt, T2)
logits_t = [Xt[l] @ Wt[l].T for l in range(L2)]
Pt = torch.zeros(Bt, E2); Ft = torch.zeros(Bt, E2)
for l in range(L2):
    p = torch.softmax(logits_t[l], -1); _, sel = torch.topk(p, K2, -1)
    Pt = Pt + p.sum(1) / (L2 * T2); Ft = Ft + torch.zeros(Bt, T2, E2).scatter_(-1, sel, 1.0).sum(1) / (L2 * T2)
F_Bt = Ft.mean(0).detach()
aux_batch = E2 * (F_Bt * Pt.mean(0)).sum()
aux_own = (E2 * (Ft.detach() * Pt).sum(-1)).mean()
g_batch = grads(aux_batch, Wt); g_own = grads(aux_own, Wt)
spec = {
    "f(B)": F_Bt.tolist(),
    "f(x=0)": Ft[0].tolist(),
    "aux_batch_value": float(aux_batch), "aux_own_value_mean": float(aux_own),
    "|grad aux_batch|": float(g_batch.norm()), "|grad aux_own_mean|": float(g_own.norm()),
}
# closed form: with f(x)=1 on S_x, grad_z of own-aux at token = (E/(L T_x)) grad_z P_S
b = 0; own = [0, 1]
zt = logits_t[0][b, 0].detach().requires_grad_(True)
pz = torch.softmax(zt, 0)
g_tok = torch.autograd.grad((Ft[b].detach() * pz).sum() * E2, zt, retain_graph=True)[0]
PS = pz[own].sum()
g_pred = E2 * torch.autograd.grad(PS, zt)[0]
spec["token closed-form check max|.| (requires f(x)=1 on S_x exactly)"] = float((g_tok - g_pred).abs().max())
spec["f(x=0) is exactly {1 on S,0 off}"] = bool(torch.allclose(Ft[0], torch.tensor([1.,1.,0,0,0,0,0,0])))
results["specialization_toy"] = spec

# ----------------------------------------------------------------------------
# (3) histogram norms: ||h||_2 <= sqrt(kL), ||h||_1 = kL; pooled: <= sqrt(k)
# ----------------------------------------------------------------------------
def hist_norms(L_, E_, K_, T_, trials=200):
    worst2 = 0.0; l1 = None
    for _ in range(trials):
        lg = torch.randn(L_, T_, E_) * (torch.rand(1) * 8)   # varying peakedness
        _, sel = torch.topk(lg, K_, -1)
        h = torch.zeros(L_, T_, E_).scatter_(-1, sel, 1.0).sum(1) / T_   # (L,E) fractions
        worst2 = max(worst2, float(h.norm()))
        l1 = float(h.abs().sum())
    # adversarial: every token -> same k experts in every layer
    h_adv = torch.zeros(L_, E_); h_adv[:, :K_] = 1.0
    return worst2, l1, float(h_adv.norm()), float((h_adv.mean(0)).norm())
w2, l1, adv2, adv_pool = hist_norms(4, 6, 2, 16)
results["hist_norms(L=4,E=6,k=2)"] = {"max random ||h||_2": w2, "sqrt(kL)": math.sqrt(8), "||h||_1": l1, "kL": 8,
                                     "adversarial ||h||_2": adv2, "adversarial pooled ||h_pool||_2": adv_pool, "sqrt(k)": math.sqrt(2)}

# ----------------------------------------------------------------------------
# (4b) per-group Mahalanobis: opaque's allocation vs naive per-group sigma*C_g
# ----------------------------------------------------------------------------
from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.types import PerGroup
pg = PerGroup(groups={"g": "grad", "h": "hist"}, values={"grad": 1.0, "hist": 0.3})
nm = 1.7
sd = per_group_noise_stddev(pg, nm)
mahal_opaque = sum(pg.values[k] ** 2 / sd.values[k] ** 2 for k in pg.values)
mahal_naive = sum(pg.values[k] ** 2 / (nm * pg.values[k]) ** 2 for k in pg.values)
mahal_iso = sum(pg.values[k] ** 2 / (nm * pg.effective) ** 2 for k in pg.values)
results["per_group_allocation"] = {
    "opaque sigma": sd.values, "formula nm*sqrt(C_g*S)": {k: nm * math.sqrt(v * sum(pg.values.values())) for k, v in pg.values.items()},
    "sum C_g^2/sigma_g^2 (opaque)": mahal_opaque, "1/nm^2": 1 / nm ** 2,
    "sum C_g^2/sigma_g^2 (naive sigma_g = nm*C_g)": mahal_naive, "K/nm^2": 2 / nm ** 2,
    "sum C_g^2/sigma_g^2 (isotropic nm*||C||_2)": mahal_iso,
}

# ----------------------------------------------------------------------------
# (6) routing pinning: gradient continuity under a tiny perturbation of theta
# near a routing tie.  Expert e is a linear map u_e; token output =
# sum_{e in topk} w_e (u_e . x); loss = sum of outputs.
# ----------------------------------------------------------------------------
Ep, Kp, Hp = 4, 2, 3
U = torch.randn(Ep, Hp)
x = torch.randn(Hp)
Wr = torch.randn(Ep, Hp)
# make experts 1 and 2 exactly tied at x: set row 2 so that (Wr x)[2] == (Wr x)[1]
with torch.no_grad():
    Wr[2] = Wr[1] + torch.cross(x, torch.randn(Hp)) * 0  # copy row
    Wr[0] += 5.0; # expert 0 dominates -> top-2 = {0, tie between 1 and 2}
def loss_route(Wr_, pin=None):
    z = Wr_ @ x
    p = torch.softmax(z, 0)
    if pin is None:
        _, sel = torch.topk(z, Kp)
    else:
        sel = pin
    wsel = p[sel] / p[sel].sum()
    return (wsel * (U[sel] @ x)).sum()
eps = 1e-9
Wa = Wr.clone().requires_grad_(True); Wb = Wr.clone(); Wb[1, 0] += eps; Wb.requires_grad_(True)
Wb2 = Wr.clone(); Wb2[1, 0] -= eps; Wb2.requires_grad_(True)
ga = torch.autograd.grad(loss_route(Wa), Wa)[0]
gb = torch.autograd.grad(loss_route(Wb), Wb)[0]
gb2 = torch.autograd.grad(loss_route(Wb2), Wb2)[0]
pin = torch.topk(Wr @ x, Kp).indices
gpa = torch.autograd.grad(loss_route(Wa, pin), Wa)[0]
gpb = torch.autograd.grad(loss_route(Wb, pin), Wb)[0]
gpb2 = torch.autograd.grad(loss_route(Wb2, pin), Wb2)[0]
results["routing_pinning"] = {
    "recomputed topk: |grad(theta+eps) - grad(theta-eps)|": float((gb - gb2).norm()),
    "pinned topk:     |grad(theta+eps) - grad(theta-eps)|": float((gpb - gpb2).norm()),
    "eps": eps,
}

# ----------------------------------------------------------------------------
# (7) Mellum2 worked numbers
# ----------------------------------------------------------------------------
Lm, Em, Km, Tm, Bm = 28, 64, 8, 1024, 256
num = {}
num["pooled Delta2 = sqrt(k)"] = math.sqrt(Km)
num["pooled centered Delta2 = sqrt(k(1-k/E))"] = math.sqrt(Km * (1 - Km / Em))
num["pooled Delta1 = k"] = Km
num["per-layer Delta2 = sqrt(kL)"] = math.sqrt(Km * Lm)
num["per-layer Delta1 = kL"] = Km * Lm
num["token-count pooled Delta2 = T sqrt(k)"] = Tm * math.sqrt(Km)
fe = Km / Em
for sh in [1.0, 2.0, 5.0]:
    num[f"separate release sigma_h={sh}: pooled per-entry std, rel err"] = (sh * math.sqrt(Km) / Bm, sh * math.sqrt(Km) / Bm / fe)
    num[f"separate release sigma_h={sh}: per-layer per-entry std, rel err"] = (sh * math.sqrt(Km * Lm) / Bm, sh * math.sqrt(Km * Lm) / Bm / fe)
    num[f"nm_eff = (1/nm^2 + 1/sigma_h^2)^-1/2 at nm=1, sigma_h={sh}"] = (1 + 1 / sh ** 2) ** -0.5
for beta in [0.9, 0.95, 0.99]:
    num[f"EMA beta={beta}: std shrink sqrt((1-b)/(1+b)), lag 1/(1-b)"] = (math.sqrt((1 - beta) / (1 + beta)), 1 / (1 - beta))
for rho in [0.1, 0.3, 0.5]:
    num[f"joint clip rho={rho}: rel err sigma E/(rho B sqrt k), grad budget sqrt(1-rho^2)"] = (Em / (rho * Bm * math.sqrt(Km)), math.sqrt(1 - rho ** 2))
    num[f"per-group optimal rho={rho}: rel err sigma E sqrt((1+rho)/rho)/(B sqrt k), grad noise x sqrt(1+rho)"] = (Em * math.sqrt((1 + rho) / rho) / (Bm * math.sqrt(Km)), math.sqrt(1 + rho))
    num[f"per-group isotropic rho={rho}: rel err sigma E sqrt(1+rho^2)/(rho B sqrt k), grad noise x sqrt(1+rho^2)"] = (Em * math.sqrt(1 + rho ** 2) / (rho * Bm * math.sqrt(Km)), math.sqrt(1 + rho ** 2))
num["aux value at balance = k"] = Km
num["alpha*aux at balance"] = 0.001 * Km
results["mellum2_numbers"] = num

print(json.dumps(results, indent=2, default=str))
