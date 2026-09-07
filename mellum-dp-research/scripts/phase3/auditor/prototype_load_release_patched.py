"""End-to-end prototype of the phase-2 v2 design (Mellum router-load release) on a tiny
Mellum with the REAL Opaque primitives. CPU only.

Mechanism (design v2 §1.1, §2.1-2.4):
  per-example loss  ell_x = CE_x + alpha*(S_x - sg[S_x]) + <z, sg[lam * w_x * d^{(L,E)}(x)]>
  S_x = E * w_x * sum_e (f_tilde_e - k/E) * P_e(x),  w_x = T_x / T_bar (T_bar = T_max)
  z: zero (L,E) probe parameter registered BEFORE make_functional(partition_trainable=True)
  clipping: PerGroup by direct construction {fallback: C_g, router_load_probe: C_h},
            C_h = lam*Delta_L*(1+1e-3), Delta_L = sqrt(k*L*(1-k/E)), lam = rho*C_g/Delta_L
  noise: opaque.dpsgd.noise.gaussian_noise (DP-SGD) / opaque.dpftrl.noise.mf_gaussian_noise (band-MF)
  post-processing: pool layers -> sum-zero projection -> bias-corrected EMA (beta=0.9 here) ->
                   known noise std s_t -> dead zone c=2 -> JS+ shrink -> clamp -> f_tilde_{t+1}

Checks V1..V7 are described in the report (phase3-prototype.md). Everything printed to stdout.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

ROOT = "/home/user/opaque"
SCRATCH = "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research"
OUT_DIR = os.path.join(SCRATCH, "auditor")  # AUDITOR PATCH: never overwrite the prototype directory
sys.path.insert(0, os.path.join(ROOT, "packages/opaque-patches/tests/transformers/models"))
sys.path.insert(0, os.path.join(SCRATCH, "empirical"))

from _test_utils import build_moe_model  # noqa: E402
from common import structured_batch  # noqa: E402  (phase-1 empirical helper)

import opaque.accounting as acc  # noqa: E402
import opaque.dpsgd.accounting as dpsgd_acc  # noqa: E402
import opaque.dpftrl.accounting as dpftrl_acc  # noqa: E402
from opaque.api.engine.noise_allocation import per_group_noise_stddev  # noqa: E402
from opaque.api.patches.kernels._linear_ce_chunked import linear_nll_sum_chunked  # noqa: E402
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise  # noqa: E402
from opaque.dpftrl.sampling import BMinSepSampler  # noqa: E402
from opaque.dpsgd.clipping import clipped_grad  # noqa: E402
from opaque.dpsgd.noise import gaussian_noise  # noqa: E402
from opaque.dpsgd.sampling import PoissonSampler  # noqa: E402
from opaque.functional import make_functional  # noqa: E402
from opaque.random import key as rng_key  # noqa: E402
from opaque.types import PerGroup  # noqa: E402

# ----------------------------------------------------------------------------- constants
E, K, L = 8, 2, 2
T_MAX = 32
N_TRAIN = 1024
N_HELD = 256
B_BAR = 32
N_STEPS = 300
N_STEPS_MF = 50
BETA = 0.9  # design default 0.99; 0.9 so a 300-step run reaches stationarity (lag 10 steps)
DEAD_ZONE_C = 2.0
GUARD = 1e-3
DELTA_ACC = 1e-5
EPS_TARGET = 3.0
ALPHA_LAB = 1.0  # Mellum2 uses 1e-3 / 1e-4; at toy scale the term is invisible under DP noise below ~1 (see report)
LR = 0.005
ROUTER_SCALE = 3.0  # induced imbalance: rows 0,1 of every router scaled by this factor
PROBE = "router_load_probe"
TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=10000.0,
    num_experts=E, num_experts_per_tok=K, moe_intermediate_size=32,
)
DELTA_H = math.sqrt(K * (1 - K / E))
DELTA_L = math.sqrt(K * L * (1 - K / E))
CTX: dict = {"f_tilde": torch.full((E,), K / E), "alpha": 0.0, "lam": 1.0, "t_bar": float(T_MAX)}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- model
def build_model(router_scale: float, seed: int = 0):
    """Tiny patched Mellum exactly as the repo tests build it (+ probe parameter)."""
    torch.manual_seed(seed)
    model, mod = build_moe_model("mellum", "cpu", **TINY)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_(any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")) or ".mlp.gate." in n)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".mlp.gate." in n:
                p[:2] *= router_scale
    # zero (L,E) probe leaf, registered BEFORE make_functional so it is a trainable leaf
    model.register_parameter(PROBE, torch.nn.Parameter(torch.zeros(L, E)))
    causal_cls = type(model)
    if not getattr(causal_cls.forward, "__proto_router_logits__", False):
        _orig = causal_cls.forward

        def forward(self, input_ids=None, attention_mask=None, labels=None, opaque_router_logits=False, **kw):
            # emulation of design §9.1: backbone with output_router_logits=True + chunked CE, no HF aux
            if not opaque_router_logits:
                return _orig(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
            nll = linear_nll_sum_chunked(out[0], self.lm_head.weight, labels, -100, 0, 0.0, False, chunk_vocab=None)
            n_valid = (labels[..., 1:] != -100).sum().float().clamp(min=1)
            return {"loss": nll / n_valid, "router_logits": out.router_logits}

        forward.__proto_router_logits__ = True
        causal_cls.forward = forward
    fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
    assert PROBE in trainable and tuple(trainable[PROBE].shape) == (L, E)
    return model, mod, fmodel, trainable, frozen


def make_data(n: int, seed_tokens: int, seed_len: int):
    ids = structured_batch(n, T_MAX, seed=seed_tokens)
    g = torch.Generator().manual_seed(seed_len)
    lengths = torch.randint(T_MAX // 2, T_MAX + 1, (n,), generator=g)  # T_x uniform in [T/2, T]
    mask = (torch.arange(T_MAX)[None] < lengths[:, None]).long()
    ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
    labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))  # right padding, pad labels -100
    return ids, mask, labels


# ----------------------------------------------------------------------------- statistics (vmap-safe)
def router_stats(router_logits, mask_row):
    """Per-example: h_layers (L,E), P (E,), T_x from the executed fp32 softmax top-k; binarised mask."""
    m = (mask_row != 0).float()
    Tx = m.sum()
    denom = Tx.clamp(min=1.0)
    hL, P = [], torch.zeros(E)
    for z in router_logits:
        z = z.reshape(-1, E)
        p = torch.softmax(z.float(), dim=-1)  # same op as MellumTopKRouter (softmax in fp32)
        idx = torch.topk(p, K, dim=-1).indices  # same op as the router -> identical executed set
        onehot = (idx[..., None] == torch.arange(E)).sum(-2).float()  # broadcast-compare (vmap-safe)
        hL.append((onehot * m[:, None]).sum(0) / denom)
        P = P + (p * m[:, None]).sum(0)
    hL = torch.stack(hL)
    P = P / (len(router_logits) * denom)
    valid = Tx > 0
    hL = torch.where(valid, hL, torch.zeros_like(hL))
    P = torch.where(valid, P, torch.zeros_like(P))
    return hL, P, Tx


def make_loss_fn(fmodel, frozen, include_ce=True):
    def loss_fn(tr, ids, mask, labels):
        params = {**frozen, **tr}
        out = fmodel(params, input_ids=ids[None], attention_mask=mask[None], labels=labels[None],
                     opaque_router_logits=True)
        hL, P, Tx = router_stats(out["router_logits"], mask)
        w = Tx / CTX["t_bar"]
        dL = torch.where(Tx > 0, hL - K / E, torch.zeros_like(hL))
        S = E * w * ((CTX["f_tilde"] - K / E) * P).sum()
        probe_term = (tr[PROBE] * (CTX["lam"] * w * dL).detach()).sum()
        ce = out["loss"] if include_ce else 0.0
        loss = ce + CTX["alpha"] * (S - S.detach()) + probe_term
        return loss, (hL, P, Tx)

    return loss_fn


def make_pergroup(trainable, C_g, C_h):
    groups = {(k,): "fallback" for k in trainable if k != PROBE}
    groups[(PROBE,)] = "router_load_probe"
    return PerGroup(groups=groups, values={"fallback": C_g, "router_load_probe": C_h})


def batched_forward(fmodel, frozen, params, ids, mask, labels):
    """Lab helper (not per-example): batched forward -> CE per example, hL (B,L,E), P (B,E), Tx (B,)."""
    out = fmodel({**frozen, **params}, input_ids=ids, attention_mask=mask, labels=labels, opaque_router_logits=True)
    B = ids.shape[0]
    m = (mask != 0).float()
    Tx = m.sum(1)
    hL, P = [], torch.zeros(B, E)
    for z in out["router_logits"]:
        z = z.reshape(B, -1, E)
        p = torch.softmax(z.float(), -1)
        idx = torch.topk(p, K, -1).indices
        oh = (idx[..., None] == torch.arange(E)).sum(-2).float()
        hL.append((oh * m[..., None]).sum(1) / Tx.clamp(min=1)[:, None])
        P = P + (p * m[..., None]).sum(1)
    hL = torch.stack(hL, 1)
    P = P / (L * Tx.clamp(min=1)[:, None])
    return out, hL, P, Tx


def hf_aux_out_of_place(router_logits, mask):
    """HF load_balancing_loss_func (modeling_mellum.py:540-606) with one-hot sums instead of scatter_add_."""
    flat = mask.reshape(-1).float()
    cnt, ps, tot = torch.zeros(E), torch.zeros(E), 0.0
    for z in router_logits:
        p = torch.softmax(z, dim=-1)
        idx = torch.topk(p, K, dim=-1).indices
        oh = (idx[..., None] == torch.arange(E)).sum(-2).float()
        cnt = cnt + (oh * flat[:, None]).sum(0)
        ps = ps + (p.float() * flat[:, None]).sum(0)
        tot = tot + flat.sum()
    return E * torch.sum((cnt / tot) * (ps / tot))


def token_weighted_f(hL, Tx):
    """HF pooled load f(B) = sum_x (T_x/T_tot) h(x), pooled over layers."""
    return (hL.mean(1) * Tx[:, None]).sum(0) / Tx.sum()


def per_example_ce(fmodel, frozen, params, ids, mask, labels):
    with torch.no_grad():
        out = fmodel({**frozen, **params}, input_ids=ids, attention_mask=mask, labels=labels, opaque_router_logits=False)
        logits = out.logits[:, :-1].float()
        tgt = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                ignore_index=-100, reduction="none").reshape(tgt.shape)
        valid = (tgt != -100).float()
        return (nll * valid).sum(1) / valid.sum(1).clamp(min=1)


def flat_cat(d, skip=(PROBE,)):
    return torch.cat([v.reshape(-1) for k, v in sorted(d.items()) if k not in skip])


def cosine(a, b):
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------- filter factors
def phi_table_sgd(n, beta):
    """phi_t (t=1..n): std factor of the EMA of iid unit noise after sum-zero projection."""
    w = (1 - beta) * beta ** np.arange(n)
    return np.sqrt(np.cumsum(w**2) * (E - 1) / E)


def mf_matrices(strategy, n):
    c = strategy.coefficients(n_steps=n).double().numpy()
    c = np.concatenate([c, np.zeros(max(0, n - len(c)))])[:n]
    C = np.zeros((n, n))
    for i in range(n):
        C[i:, i] = c[: n - i]
    Cinv = np.linalg.inv(C)
    return C, Cinv


def phi_table_mf(strategy, n, beta):
    _, Cinv = mf_matrices(strategy, n)
    w = (1 - beta) * beta ** np.arange(n)
    F = np.zeros((n, n))
    for i in range(n):
        F[i:, i] = w[: n - i]
    FC = F @ Cinv
    return np.linalg.norm(FC, axis=1) * math.sqrt((E - 1) / E), np.linalg.norm(Cinv, axis=1)


# ----------------------------------------------------------------------------- post-processing (design §2.4)
def postprocess(state, y, lam, sigma_h_base, phi, beta, c):
    """state: dict(m (E,), m_layers (L,E), t int). y: noised probe leaf (L,E). Returns new state + info."""
    t = state["t"] + 1
    d_hat_L = y / lam                                   # 1
    d_hat = d_hat_L.mean(0)                             # 2 pooled
    d_hat = d_hat - d_hat.mean()                        # 3 sum-zero projection
    m = beta * state["m"] + (1 - beta) * d_hat          # 4 EMA
    mL = beta * state["m_layers"] + (1 - beta) * (d_hat_L - d_hat_L.mean(-1, keepdim=True))
    corr = 1 - beta**t
    d_tilde = m / corr
    s = (sigma_h_base / (lam * math.sqrt(L))) * float(phi[t - 1]) / corr
    n2 = float(d_tilde.pow(2).sum())
    if n2 < c * E * s * s:                              # 5 dead zone
        d_plus = torch.zeros_like(d_tilde); dead = True; shrink = 0.0
    else:
        shrink = 1 - E * s * s / n2                     # JS+ toward balance
        d_plus = d_tilde * shrink; dead = False
    f_tilde = torch.clamp(K / E + d_plus, 0.0, 1.0)     # 6
    D = float(d_tilde.abs().max() / (K / E))            # 7 monitor
    D_layer = float((mL / corr).abs().max() / (K / E))
    new = {"m": m, "m_layers": mL, "t": t}
    return new, f_tilde, dict(s=s, dead=dead, shrink=shrink, D=D, D_layer=D_layer, d_tilde=d_tilde)


# ----------------------------------------------------------------------------- one training arm (V5 / V6)
def run_arm(cfg: dict) -> dict:
    torch.set_num_threads(2)
    name = cfg["name"]
    t0 = time.time()
    model, mod, fmodel, trainable, frozen = build_model(cfg["router_scale"], seed=0)
    ids, mask, labels = make_data(N_TRAIN, 1, 7)
    h_ids, h_mask, h_labels = make_data(N_HELD, 2, 8)
    params = {k: v.clone() for k, v in trainable.items()}
    C_g, rho, nm, alpha, f_mode = cfg["C_g"], cfg["rho"], cfg["nm"], cfg["alpha"], cfg["f_mode"]
    lam = rho * C_g / DELTA_L
    C_h = lam * DELTA_L * (1 + GUARD)
    CTX.update(f_tilde=torch.full((E,), K / E), alpha=alpha, lam=lam, t_bar=float(T_MAX))
    pg = make_pergroup(params, C_g, C_h)
    loss_fn = make_loss_fn(fmodel, frozen)
    gf, clip_state = clipped_grad(loss_fn, has_aux=True, clipping_norm=pg, normalize_by=float(B_BAR),
                                  batch_argnums=(1, 2, 3), return_aux=True)
    n_steps = cfg["n_steps"]
    q = B_BAR / N_TRAIN
    if cfg["noise"] == "gaussian":
        noise_fn, nstate = gaussian_noise(noise_multiplier=nm, key=rng_key(cfg["seed"]))
        sampler = PoissonSampler(ids, sample_rate=q, n_steps=n_steps, key=rng_key(100 + cfg["seed"]))
        phi = phi_table_sgd(n_steps, BETA)
        rownorm = np.ones(n_steps)
        strategy = None
    else:
        strategy = band_mf_strategy(bands=4, momentum=0.95)
        noise_fn, nstate = mf_gaussian_noise(params, strategy, n_steps=n_steps, noise_multiplier=nm,
                                             key=rng_key(cfg["seed"]))
        sampler = BMinSepSampler(ids, bands=4, sampling_prob=q / (1 - q * (4 - 1)), n_steps=n_steps, key=rng_key(100 + cfg["seed"]))  # AUDITOR PATCH: paper-p from p0 (design section 2.3; _b_min_sep.py:7-10; trainer _dpftrl.py:246-252 via amp.sampling_prob)
        phi, rownorm = phi_table_mf(strategy, n_steps, BETA)
    # base sigma_h of the probe group (design: sigma_h = nm*sqrt(C_h*(C_g+C_h))/B_bar)
    base = per_group_noise_stddev(PerGroup(pg.groups, {g: v / B_BAR for g, v in pg.values.items()}), nm)
    sigma_h_base = float(base.values["router_load_probe"])
    sigma_g_base = float(base.values["fallback"])
    state = {"m": torch.zeros(E), "m_layers": torch.zeros(L, E), "t": 0}
    f_tilde = torch.full((E,), K / E)
    hist = dict(step=[], delta_batch=[], track_err=[], track_err_rescaled=[], dead=[], D=[], s=[],
                shrink=[], clip_rate=[], probe_clip=[], batch_size=[], ce_batch=[], cos=[], cos_step=[],
                heldout_step=[], delta_heldout=[], ce_heldout=[], f_heldout=[], noise_std_probe=[],
                noise_std_fallback=[], probe_noise_sample=[], clipped_probe=[], f_tilde=[], cos_pop=[],
                cos_batch_vs_pop=[], track_err_pop=[], track_err_pop_rescaled=[], delta_pop_at_cos=[], dead_at_cos=[])

    def heldout_eval(step):
        with torch.no_grad():
            _, hL, P, Tx = batched_forward(fmodel, frozen, params, h_ids, h_mask, h_labels)
        f = token_weighted_f(hL, Tx)
        d = f - K / E
        hist["heldout_step"].append(step)
        hist["delta_heldout"].append(float(d.pow(2).mean().sqrt() / (K / E)))
        hist["f_heldout"].append([round(x, 4) for x in f.tolist()])
        hist["ce_heldout"].append(float(per_example_ce(fmodel, frozen, params, h_ids, h_mask, h_labels).mean()))

    heldout_eval(0)
    for step, idx in enumerate(sampler):
        if step >= n_steps:
            break
        idx = torch.as_tensor(list(idx), dtype=torch.long)
        b_ids, b_mask, b_labels = ids[idx], mask[idx], labels[idx]
        # ---- augment (once per step, outside vmap): the public constant f_tilde_t for this step
        if f_mode == "oracle":
            with torch.no_grad():
                _, hL_b, _, Tx_b = batched_forward(fmodel, frozen, params, b_ids, b_mask, b_labels)
            f_used = token_weighted_f(hL_b, Tx_b) if len(idx) > 0 else torch.full((E,), K / E)
        elif f_mode == "dp":
            f_used = f_tilde
        else:
            f_used = torch.full((E,), K / E)
        CTX["f_tilde"] = f_used
        assert torch.count_nonzero(params[PROBE]) == 0
        # ---- per-example vmap(grad) -> per-group clip -> sum / B_bar
        (clipped, aux), clip_state = gf(params, b_ids, b_mask, b_labels, state=clip_state)
        # ---- noise (one joint per-group Gaussian / MF release)
        noised, nstate = noise_fn(clipped, nstate)
        y = noised.pytree[PROBE].clone()
        hist["noise_std_probe"].append(float(noised.noise_stddev.values["router_load_probe"]))
        hist["noise_std_fallback"].append(float(noised.noise_stddev.values["fallback"]))
        hist["probe_noise_sample"].append((y - clipped.pytree[PROBE]).reshape(-1).tolist())
        hist["clipped_probe"].append(clipped.pytree[PROBE].reshape(-1).tolist())
        # ---- post-processing (public) -> f_tilde_{t+1}
        state, f_tilde, info = postprocess(state, y, lam, sigma_h_base, phi, BETA, DEAD_ZONE_C)
        noised.pytree[PROBE].zero_()  # probe update is exactly 0
        # ---- cosine metrics at the PRE-update parameters (the ones the loss used)
        if step % 10 == 0 and alpha > 0 and len(idx) > 0:
            with torch.no_grad():
                _, hL_p, _, Tx_p = batched_forward(fmodel, frozen, params, h_ids, h_mask, h_labels)
            f_pop = token_weighted_f(hL_p, Tx_p)

            def three(p_):
                out = fmodel({**frozen, **p_}, input_ids=b_ids, attention_mask=b_mask, labels=b_labels,
                             opaque_router_logits=True)
                B = b_ids.shape[0]; m = (b_mask != 0).float(); Tx = m.sum(1)
                hL, P = [], torch.zeros(B, E)
                for z in out["router_logits"]:
                    z = z.reshape(B, -1, E); p = torch.softmax(z.float(), -1)
                    ix = torch.topk(p, K, -1).indices
                    oh = (ix[..., None] == torch.arange(E)).sum(-2).float()
                    hL.append((oh * m[..., None]).sum(1) / Tx.clamp(min=1)[:, None])
                    P = P + (p * m[..., None]).sum(1)
                hL = torch.stack(hL, 1).detach(); P = P / (L * Tx.clamp(min=1)[:, None])
                f_exact = token_weighted_f(hL, Tx)
                w = Tx / T_MAX
                S_dp = (E * w[:, None] * (f_used - K / E)[None] * P).sum() / B_BAR
                S_ex = (E * w[:, None] * (f_exact - K / E)[None] * P).sum() / B_BAR
                S_pop = (E * w[:, None] * (f_pop - K / E)[None] * P).sum() / B_BAR
                return torch.stack([S_dp, S_ex, S_pop])
            _, vjp = torch.func.vjp(three, {k: v for k, v in params.items() if k != PROBE})
            g_dp = flat_cat(vjp(torch.tensor([1.0, 0.0, 0.0]))[0]); g_ex = flat_cat(vjp(torch.tensor([0.0, 1.0, 0.0]))[0])
            g_pop = flat_cat(vjp(torch.tensor([0.0, 0.0, 1.0]))[0])
            nz = g_dp.norm() > 0
            hist["cos"].append(cosine(g_dp, g_ex) if nz else float("nan")); hist["cos_step"].append(step)
            hist["cos_pop"].append(cosine(g_dp, g_pop) if nz else float("nan"))
            hist["cos_batch_vs_pop"].append(cosine(g_ex, g_pop))
            d_pop = f_pop - K / E
            hist["track_err_pop"].append(float((f_used - f_pop).norm() / d_pop.norm().clamp_min(1e-12)))
            hist["track_err_pop_rescaled"].append(float(((f_used - K / E) * (T_MAX / float(Tx_p.mean())) - d_pop).norm() / d_pop.norm().clamp_min(1e-12)))
            hist["delta_pop_at_cos"].append(float(d_pop.pow(2).mean().sqrt() / (K / E)))
            hist["dead_at_cos"].append(bool(info["dead"]))
        # ---- plain SGD on the other leaves
        with torch.no_grad():
            for k_ in params:
                if k_ != PROBE:
                    params[k_] -= LR * noised.pytree[k_]
        hist["f_tilde"].append([round(x, 5) for x in f_used.tolist()])
        # ---- lab-only metrics (from the private per-example aux; never released in a real run)
        hL_x, P_x, Tx_x = aux.loss_aux
        if len(idx) > 0:
            f_B = token_weighted_f(hL_x, Tx_x)
            d_B = f_B - K / E
            delta_b = float(d_B.pow(2).mean().sqrt() / (K / E))
            err = float((f_used - f_B).norm() / d_B.norm().clamp_min(1e-12))
            t_bar_true = float(Tx_x.mean())
            err_resc = float(((f_used - K / E) * (T_MAX / t_bar_true) - d_B).norm() / d_B.norm().clamp_min(1e-12))
        else:
            delta_b, err, err_resc = float("nan"), float("nan"), float("nan")
        hist["step"].append(step); hist["delta_batch"].append(delta_b)
        hist["track_err"].append(err); hist["track_err_rescaled"].append(err_resc)
        hist["dead"].append(bool(info["dead"])); hist["D"].append(info["D"]); hist["s"].append(info["s"])
        hist["shrink"].append(info["shrink"])
        gn = aux.group_norms
        hist["clip_rate"].append(float((gn["fallback"] > C_g).float().mean()) if len(idx) else float("nan"))
        hist["probe_clip"].append(float((gn["router_load_probe"] > C_h).float().sum()))
        hist["batch_size"].append(int(aux.batch_size))
        hist["ce_batch"].append(float(aux.loss_values.mean()) if len(idx) else float("nan"))
        if (step + 1) % 25 == 0:
            heldout_eval(step + 1)
    ns_all = np.array(hist["probe_noise_sample"]).reshape(-1, L, E)  # (steps, L, E)
    pooled = ns_all.mean(1) / lam
    pooled = pooled - pooled.mean(-1, keepdims=True)
    pred_pooled = sigma_h_base / (lam * math.sqrt(L)) * math.sqrt((E - 1) / E) * (np.mean(rownorm[: len(pooled)] ** 2) ** 0.5)
    emp_pooled = float(np.sqrt(np.mean(pooled**2))) if nm > 0 else 0.0
    sig_all = np.array(hist["clipped_probe"]).reshape(-1, L, E)[-100:].mean(1) / lam  # (steps, E): (1/B) sum_x w_x d(x)
    batch_signal_std = float(np.sqrt(np.mean(sig_all.var(axis=0))) / (K / E))
    res = dict(name=name, cfg=cfg, lam=lam, C_h=C_h, sigma_h_base=sigma_h_base, sigma_g_base=sigma_g_base,
               pooled_noise_std_empirical=emp_pooled, pooled_noise_std_predicted=float(pred_pooled),
               batch_signal_std_kE=batch_signal_std,
               phi_inf=float(phi[-1]), rownorm=rownorm.tolist() if cfg["noise"] == "mf" else None,
               s_inf=(sigma_h_base / (lam * math.sqrt(L))) * float(phi[-1]) / (1 - BETA**n_steps),
               elapsed=time.time() - t0, hist=hist, probe_param_final_nonzero=int(torch.count_nonzero(params[PROBE])))
    if cfg["noise"] == "mf":
        # V6: realised noise_stddev per group == base * ||row_t(C^-1)||
        pred_probe = sigma_h_base * rownorm[: len(hist["noise_std_probe"])]
        pred_fb = sigma_g_base * rownorm[: len(hist["noise_std_fallback"])]
        res["mf_sigma_relerr_probe"] = float(np.max(np.abs(np.array(hist["noise_std_probe"]) - pred_probe) / pred_probe))
        res["mf_sigma_relerr_fallback"] = float(np.max(np.abs(np.array(hist["noise_std_fallback"]) - pred_fb) / pred_fb))
        ns = np.array(hist["probe_noise_sample"])  # (steps, L*E)
        res["mf_probe_noise_empirical_std_over_pred"] = float(np.sqrt(np.mean((ns / pred_probe[:, None]) ** 2)))
        res["mf_probe_received_noise"] = bool(np.abs(ns).max() > 0)
        # AUDITOR PATCH: assert the latch actually holds the two-group PerGroup instead of hard-coding True
        res["mf_latch_ok"] = bool(nstate._first_max_norm is not None and nstate._first_max_norm == clipped.max_norm and isinstance(nstate._first_max_norm, PerGroup) and set(nstate._first_max_norm.values) == {"fallback", "router_load_probe"})
    return res


def summarize_arm(r):
    h = r["hist"]
    dead = np.array(h["dead"]); cos = np.array(h["cos"], dtype=float); cosp = np.array(h["cos_pop"], dtype=float)
    dac = np.array(h["dead_at_cos"], dtype=bool) if len(h["dead_at_cos"]) else np.zeros(0, dtype=bool)
    tep = np.array(h["track_err_pop"], dtype=float); tepr = np.array(h["track_err_pop_rescaled"], dtype=float)
    cbp = np.array(h["cos_batch_vs_pop"], dtype=float)
    te = np.array(h["track_err"], dtype=float); ter = np.array(h["track_err_rescaled"], dtype=float)
    return dict(
        name=r["name"], alpha=r["cfg"]["alpha"], rho=r["cfg"]["rho"], nm=r["cfg"]["nm"], f_mode=r["cfg"]["f_mode"],
        noise=r["cfg"]["noise"], lam=r["lam"], C_h=r["C_h"], sigma_h=r["sigma_h_base"], s_inf=r["s_inf"],
        s_inf_over_kE=r["s_inf"] / (K / E),
        delta_heldout=[round(x, 4) for x in h["delta_heldout"]], heldout_step=h["heldout_step"],
        ce_heldout=[round(x, 4) for x in h["ce_heldout"]],
        delta_heldout_final=h["delta_heldout"][-1], ce_heldout_final=h["ce_heldout"][-1],
        delta_batch_mean_last50=float(np.nanmean(h["delta_batch"][-50:])),
        mean_cos=float(np.nanmean(cos)) if len(cos) else None, n_cos=int(len(cos)),
        mean_cos_pop=float(np.nanmean(cosp)) if len(cosp) else None,
        mean_cos_pop_active=float(np.nanmean(cosp[~dac])) if len(cosp) and (~dac).any() else None,
        mean_cos_batch_vs_pop=float(np.nanmean(cbp)) if len(cbp) else None,
        track_err_pop_mean=float(np.nanmean(tep)) if len(tep) else None,
        track_err_pop_rescaled_mean=float(np.nanmean(tepr)) if len(tepr) else None,
        track_err_pop_rescaled_active=float(np.nanmean(tepr[~dac])) if len(tepr) and (~dac).any() else None,
        pooled_noise_std_empirical=r["pooled_noise_std_empirical"], pooled_noise_std_predicted=r["pooled_noise_std_predicted"],
        batch_signal_std_kE=r["batch_signal_std_kE"],
        dead_zone_rate=float(dead.mean()), track_err_mean=float(np.nanmean(te)),
        track_err_rescaled_mean=float(np.nanmean(ter)), track_err_last50=float(np.nanmean(te[-50:])),
        track_err_rescaled_last50=float(np.nanmean(ter[-50:])),
        D_mean_last50=float(np.mean(h["D"][-50:])), shrink_mean=float(np.mean(h["shrink"])),
        clip_rate_mean=float(np.nanmean(h["clip_rate"])), probe_clip_events=float(np.sum(h["probe_clip"])),
        batch_size_mean=float(np.mean(h["batch_size"])), ce_batch_mean_last50=float(np.nanmean(h["ce_batch"][-50:])),
        elapsed=r["elapsed"], probe_param_final_nonzero=r["probe_param_final_nonzero"],
    )


# ----------------------------------------------------------------------------- main
def main():
    T0 = time.time()
    checks = {}
    results = {"constants": dict(E=E, K=K, L=L, T_MAX=T_MAX, N_TRAIN=N_TRAIN, B_BAR=B_BAR, N_STEPS=N_STEPS,
                                 BETA=BETA, DEAD_ZONE_C=DEAD_ZONE_C, GUARD=GUARD, ALPHA_LAB=ALPHA_LAB, LR=LR,
                                 ROUTER_SCALE=ROUTER_SCALE, DELTA_H=DELTA_H, DELTA_L=DELTA_L)}
    log("=" * 100)
    log("PROTOTYPE: v2 router-load release on tiny Mellum (E=8,k=2,L=2,hidden=64), CPU")
    log("=" * 100)
    model, mod, fmodel, trainable, frozen = build_model(ROUTER_SCALE, seed=0)
    log(f"experts forward patched: {hasattr(mod.MellumExperts.forward, '__opaque_patched__')}; "
        f"experts impl: {getattr(model.config, '_experts_implementation', None)}; "
        f"trainable leaves: {len(trainable)} ({sum(v.numel() for v in trainable.values())} params); probe in trainable: {PROBE in trainable}")
    ids, mask, labels = make_data(N_TRAIN, 1, 7)
    h_ids, h_mask, h_labels = make_data(N_HELD, 2, 8)
    t_bar_true = float(h_mask.sum(1).float().mean())
    log(f"lengths: T_max={T_MAX}, held-out mean length T_bar_true={t_bar_true:.2f} (T_bar used = T_max -> release scale factor {t_bar_true / T_MAX:.3f})")

    # ---------------- calibration on the disjoint held-out split (design §4.4): C_g = median per-example norm at alpha=0
    CTX.update(f_tilde=torch.full((E,), K / E), alpha=0.0, lam=1.0, t_bar=float(T_MAX))
    loss_fn = make_loss_fn(fmodel, frozen)
    pg_big = make_pergroup(trainable, 1e9, 1e9)
    gf_cal, st = clipped_grad(loss_fn, has_aux=True, clipping_norm=pg_big, normalize_by=float(N_HELD),
                              batch_argnums=(1, 2, 3), return_aux=True)
    (_, aux_cal), st = gf_cal(trainable, h_ids, h_mask, h_labels, state=st)
    gn0 = aux_cal.group_norms["fallback"]
    C_g = float(gn0.median())
    with torch.no_grad():
        _, hL_h, _, Tx_h = batched_forward(fmodel, frozen, trainable, h_ids, h_mask, h_labels)
    f0 = token_weighted_f(hL_h, Tx_h)
    delta0 = float((f0 - K / E).pow(2).mean().sqrt() / (K / E))
    D0 = float((f0 - K / E).abs().max() / (K / E))
    log(f"calibration (held-out, alpha=0): per-example grad norm p10/p50/p90/max = {gn0.quantile(0.1):.3f}/{gn0.median():.3f}/{gn0.quantile(0.9):.3f}/{gn0.max():.3f} -> C_g={C_g:.4f}")
    # per-example norm change from the aux term at the lab alpha
    CTX.update(alpha=ALPHA_LAB, f_tilde=f0)
    (_, aux_cal1), st = gf_cal(trainable, h_ids, h_mask, h_labels, state=st)
    gn1 = aux_cal1.group_norms["fallback"]
    log(f"  per-example norm with alpha={ALPHA_LAB} (f_tilde=f_heldout): median {gn1.median():.3f} (x{gn1.median() / gn0.median():.3f}); "
        f"aux.loss_values == CE exactly (value-neutral surrogate): {torch.equal(aux_cal1.loss_values, aux_cal.loss_values)}")
    log(f"induced imbalance (router rows 0,1 x{ROUTER_SCALE}): f_heldout={[round(x, 3) for x in f0.tolist()]} delta_0={delta0:.3f} D_0={D0:.3f}")
    results["calibration"] = dict(C_g=C_g, delta0=delta0, D0=D0, f0=f0.tolist(), t_bar_true=t_bar_true,
                                  norm_median_alpha0=float(gn0.median()), norm_median_alpha_lab=float(gn1.median()))

    # ---------------- noise multiplier at eps=3 (delta=1e-5) for q=B/N, T=300 (real accountant)
    q = B_BAR / N_TRAIN
    t1 = time.time()
    cal = acc.calibrate(acc.epsilon_budget(EPS_TARGET, DELTA_ACC),
                        lambda x: dpsgd_acc.poisson(dpsgd_acc.gaussian(x), sample_rate=q) * N_STEPS, 0.3, 8.0, tolerance=1e-3)
    NM = float(cal.param)
    log(f"accountant: poisson(gaussian(nm), q={q:.5f}) * {N_STEPS} at eps={EPS_TARGET}, delta={DELTA_ACC} -> nm={NM:.4f} ({time.time() - t1:.1f}s)")

    # ---------------- rho* from the design's r formula
    phi_inf = math.sqrt((1 - BETA) / (1 + BETA)) * math.sqrt((E - 1) / E)

    def smoothed_rel_noise(rho):  # per-entry smoothed noise on the pooled d_hat, in units of k/E
        return NM * DELTA_H * (1 + GUARD) * math.sqrt(1 + 1 / rho) / (B_BAR * K / E) * phi_inf

    signal = delta0 * t_bar_true / T_MAX  # what the release sees with T_bar = T_max
    rho_star = next(r for r in np.arange(0.01, 2.0, 0.01) if smoothed_rel_noise(r) / signal < 0.30)
    rho_star = float(round(rho_star, 2))
    log(f"rho*: smallest rho with smoothed noise / released imbalance < 30%: rho*={rho_star} "
        f"(smoothed noise {smoothed_rel_noise(rho_star):.4f} k/E units vs released imbalance {signal:.4f}; "
        f"ratio {smoothed_rel_noise(rho_star) / signal:.3f}; gradient-noise inflation sqrt(1+rho)={math.sqrt(1 + rho_star):.4f}). "
        f"At rho=0.02: smoothed noise {smoothed_rel_noise(0.02):.4f} -> ratio {smoothed_rel_noise(0.02) / signal:.3f}, single-release r1={smoothed_rel_noise(0.02) / phi_inf:.3f}")
    results["rho_star"] = dict(rho_star=rho_star, NM=NM, phi_inf=phi_inf, smoothed_noise_rho_star=smoothed_rel_noise(rho_star),
                               smoothed_noise_rho002=smoothed_rel_noise(0.02), released_signal=signal,
                               dead_zone_engages_below_delta_rho_star=math.sqrt(DEAD_ZONE_C) * smoothed_rel_noise(rho_star) / (t_bar_true / T_MAX),
                               dead_zone_engages_below_delta_rho002=math.sqrt(DEAD_ZONE_C) * smoothed_rel_noise(0.02) / (t_bar_true / T_MAX))

    # ================================================================= V1 surrogate identity (patched model, ragged)
    log("\n--- V1: surrogate identity on the PATCHED model with ragged lengths")
    B1 = 16
    b_ids, b_mask, b_labels = ids[:B1], mask[:B1], labels[:B1]
    with torch.no_grad():
        out_b, hL_b, P_b, Tx_b = batched_forward(fmodel, frozen, trainable, b_ids, b_mask, b_labels)
    fB = token_weighted_f(hL_b, Tx_b)
    T_tot = float(Tx_b.sum())
    CTX.update(alpha=1.0, f_tilde=fB, lam=1.0, t_bar=float(T_MAX))
    surr_fn = make_loss_fn(fmodel, frozen, include_ce=False)
    vg = torch.func.vmap(torch.func.grad(surr_fn, has_aux=True), in_dims=(None, 0, 0, 0))
    g_per, _ = vg(trainable, b_ids, b_mask, b_labels)
    g_mean = flat_cat({k: v.mean(0) for k, v in g_per.items()})
    non_probe = {k: v for k, v in trainable.items() if k != PROBE}

    def hf_aux_own(p_):  # HF's own function (scatter_add_), outside vmap
        out = fmodel({**frozen, **p_}, input_ids=b_ids, attention_mask=b_mask, labels=b_labels, opaque_router_logits=True)
        return mod.load_balancing_loss_func(out["router_logits"], E, K, b_mask)

    def hf_aux_oop(p_):  # same formula, out-of-place
        out = fmodel({**frozen, **p_}, input_ids=b_ids, attention_mask=b_mask, labels=b_labels, opaque_router_logits=True)
        return hf_aux_out_of_place(out["router_logits"], b_mask)

    g_hf = flat_cat(torch.func.grad(hf_aux_own)(non_probe)); g_oop = flat_cat(torch.func.grad(hf_aux_oop)(non_probe))
    v_own, v_oop = float(hf_aux_own(non_probe)), float(hf_aux_oop(non_probe))
    cos1 = cosine(g_mean, g_hf); ratio1 = float(g_mean.norm() / g_hf.norm()); expect1 = T_tot / (B1 * T_MAX)
    rel_oop = float((g_oop - g_hf).norm() / g_hf.norm())
    log(f"  lengths: {b_mask.sum(1).tolist()}")
    log(f"  HF aux value (own fn) {v_own:.6f} vs out-of-place {v_oop:.6f}; grad rel-L2 oop vs own {rel_oop:.2e}")
    log(f"  cos(batch-mean surrogate grad, grad HF aux) = {cos1:.12f}; norm ratio {ratio1:.6f} vs T_tot/(B*T_bar) = {expect1:.6f} (rel diff {abs(ratio1 - expect1) / expect1:.2e})")
    checks["V1"] = dict(result="PASS" if (1 - cos1) < 1e-6 and abs(ratio1 - expect1) / expect1 < 1e-4 else "FAIL",
                        cos=cos1, ratio=ratio1, expected_ratio=expect1, rel_oop_vs_hf=rel_oop, aux_value=v_own)

    # ================================================================= V2 probe leaf identity + structural bound
    log("\n--- V2: probe leaf == (lam/B_bar) sum_x w_x d^{(L,E)}(x); group norm <= C_h for every example incl. adversarial")
    rho2 = rho_star
    lam2 = rho2 * C_g / DELTA_L; C_h2 = lam2 * DELTA_L * (1 + GUARD)
    adv_ids = torch.full((1, T_MAX), 5, dtype=torch.long); adv_mask = torch.ones(1, T_MAX, dtype=torch.long)
    adv_labels = adv_ids.clone()
    b2_ids = torch.cat([b_ids, adv_ids]); b2_mask = torch.cat([b_mask, adv_mask]); b2_labels = torch.cat([b_labels, adv_labels])
    CTX.update(alpha=ALPHA_LAB, f_tilde=fB, lam=lam2, t_bar=float(T_MAX))
    pg2 = make_pergroup(trainable, C_g, C_h2)
    gf2, st2 = clipped_grad(make_loss_fn(fmodel, frozen), has_aux=True, clipping_norm=pg2, normalize_by=float(B_BAR),
                            batch_argnums=(1, 2, 3), return_aux=True)
    (cl2, aux2), st2 = gf2(trainable, b2_ids, b2_mask, b2_labels, state=st2)
    hL2, P2, Tx2 = aux2.loss_aux
    w2 = Tx2 / T_MAX
    expected_leaf = (lam2 / B_BAR) * (w2[:, None, None] * (hL2 - K / E)).sum(0)
    leaf_err = float((cl2.pytree[PROBE] - expected_leaf).abs().max())
    pgn = aux2.group_norms["router_load_probe"]
    adv_norm = float(pgn[-1]); adv_d_norm = float((hL2[-1] - K / E).norm())
    log(f"  max|probe leaf - expected| = {leaf_err:.2e}; sum_e leaf = {float(cl2.pytree[PROBE].sum()):.2e}")
    log(f"  adversarial all-same-token example: h per layer = {hL2[-1].tolist()}; ||d^(L,E)|| = {adv_d_norm:.6f} vs Delta_L = {DELTA_L:.6f}")
    log(f"  probe group norms: max = {float(pgn.max()):.6f} (adversarial {adv_norm:.6f}) vs C_h = {C_h2:.6f}; lam*Delta_L = {lam2 * DELTA_L:.6f}; "
        f"examples with norm > C_h: {int((pgn > C_h2).sum())}; max_norm stored: {dict(cl2.max_norm.values)}")
    v2_pass = leaf_err < 1e-6 and float(pgn.max()) <= C_h2 and abs(adv_d_norm - DELTA_L) < 1e-5 and int((pgn > C_h2).sum()) == 0
    checks["V2"] = dict(result="PASS" if v2_pass else "FAIL", leaf_maxabs_err=leaf_err, probe_norm_max=float(pgn.max()),
                        adversarial_norm=adv_norm, adversarial_d_norm=adv_d_norm, Delta_L=DELTA_L, C_h=C_h2, lam=lam2,
                        n_over=int((pgn > C_h2).sum()))

    # ================================================================= V3 Mahalanobis identity with the real allocator
    log("\n--- V3: Mahalanobis identity with the real per_group_noise_stddev")
    nf3, ns3 = gaussian_noise(noise_multiplier=NM, key=rng_key(3))
    noised3, ns3 = nf3(cl2, ns3)
    sg, sh = float(noised3.noise_stddev.values["fallback"]), float(noised3.noise_stddev.values["router_load_probe"])
    maha = (C_g / B_BAR) ** 2 / sg**2 + (C_h2 / B_BAR) ** 2 / sh**2
    closed_g = NM * math.sqrt(C_g * (C_g + C_h2)) / B_BAR; closed_h = NM * math.sqrt(C_h2 * (C_g + C_h2)) / B_BAR
    log(f"  sigma_g={sg:.6f} (closed form {closed_g:.6f}), sigma_h={sh:.6f} (closed form {closed_h:.6f})")
    log(f"  (C_g/B)^2/sigma_g^2 + (C_h/B)^2/sigma_h^2 = {maha:.10f} vs 1/nm^2 = {1 / NM**2:.10f}; ratio {maha * NM**2:.12f}")
    log(f"  gradient-noise inflation sigma_g/(nm*C_g/B) = {sg / (NM * C_g / B_BAR):.6f} = sqrt(1+rho) = {math.sqrt(1 + C_h2 / C_g):.6f}")
    checks["V3"] = dict(result="PASS" if abs(maha * NM**2 - 1) < 1e-9 else "FAIL", maha_times_nm2=maha * NM**2, sigma_g=sg, sigma_h=sh,
                        inflation=sg / (NM * C_g / B_BAR))

    # ================================================================= V4 accountant invariance
    log("\n--- V4: accountant invariance (the mechanism object never sees the pytree / PerGroup)")
    sig = {n: str(inspect.signature(f)) for n, f in (("dpsgd.gaussian", dpsgd_acc.gaussian), ("dpsgd.poisson", dpsgd_acc.poisson),
                                                      ("dpftrl.mf_gaussian", dpftrl_acc.mf_gaussian))}
    for n, s in sig.items():
        log(f"  {n}{s}")
    m_without = dpsgd_acc.poisson(dpsgd_acc.gaussian(NM), sample_rate=q) * N_STEPS
    m_with = dpsgd_acc.poisson(dpsgd_acc.gaussian(NM), sample_rate=q) * N_STEPS  # identical construction: nothing about the probe enters
    e_wo, e_w = m_without.epsilon_at(DELTA_ACC), m_with.epsilon_at(DELTA_ACC)
    strat4 = band_mf_strategy(bands=4, momentum=0.95)
    mf_wo = dpftrl_acc.mf_gaussian(NM, strat4, n_steps=N_STEPS_MF); mf_w = dpftrl_acc.mf_gaussian(NM, band_mf_strategy(bands=4, momentum=0.95), n_steps=N_STEPS_MF)
    t4 = time.time(); emf_wo, emf_w = mf_wo.epsilon_at(DELTA_ACC), mf_w.epsilon_at(DELTA_ACC); t4 = time.time() - t4
    log(f"  DP-SGD: eps without probe {e_wo:.6f} == with probe {e_w:.6f}: {e_wo == e_w}; type {type(m_without).__name__}; built from (nm, q, T) only")
    log(f"  band-MF(4, .95) n_steps={N_STEPS_MF}: mf_gaussian(nm, strategy) eps without {emf_wo:.6f} == with {emf_w:.6f}: {emf_wo == emf_w} "
        f"(sensitivity {strat4.sensitivity(n_steps=N_STEPS_MF):.4f}, {t4:.1f}s, un-amplified)")
    checks["V4"] = dict(result="PASS" if (e_wo == e_w and emf_wo == emf_w) else "FAIL", eps_dpsgd=e_wo, eps_mf=emf_wo, signatures=sig,
                        no_pytree_arg=all("pytree" not in s and "PerGroup" not in s and "max_norm" not in s for s in sig.values()))

    # ================================================================= V7 gradient checkpointing on/off equality
    log("\n--- V7: gradient checkpointing on/off equality of the probe leaf and all grads")
    model.gradient_checkpointing_enable()
    ck_flags = [bool(getattr(m, "gradient_checkpointing", False)) for m in model.model.layers]
    (cl7, aux7), _ = gf2(trainable, b2_ids, b2_mask, b2_labels, state=st2)
    model.gradient_checkpointing_disable()
    (cl7b, _), _ = gf2(trainable, b2_ids, b2_mask, b2_labels, state=st2)
    rel7 = {k: float((cl7.pytree[k] - cl7b.pytree[k]).norm() / cl7b.pytree[k].norm().clamp_min(1e-30)) for k in cl7.pytree}
    n_layers7 = None
    log(f"  checkpointing flags per layer: {ck_flags}; probe leaf rel-L2 ckpt vs no-ckpt {rel7[PROBE]:.2e}; max over all leaves {max(rel7.values()):.2e}; "
        f"vs the earlier non-ckpt run: {float((cl7b.pytree[PROBE] - cl2.pytree[PROBE]).abs().max()):.2e}")
    checks["V7"] = dict(result="PASS" if max(rel7.values()) < 1e-6 and all(ck_flags) else "FAIL", max_rel=max(rel7.values()), probe_rel=rel7[PROBE], flags=ck_flags)

    # ================================================================= V5 / V6 training arms
    log("\n--- V5/V6: training arms (B_bar=32, N=1024, Poisson; alpha lab=1.0; lr=0.005)")
    common = dict(C_g=C_g, n_steps=N_STEPS, router_scale=ROUTER_SCALE, noise="gaussian", seed=11)
    arms = [
        dict(common, name="OFF", alpha=0.0, rho=rho_star, nm=NM, f_mode="none"),
        dict(common, name="ORACLE", alpha=ALPHA_LAB, rho=rho_star, nm=NM, f_mode="oracle"),
        dict(common, name=f"DP_rho{rho_star}", alpha=ALPHA_LAB, rho=rho_star, nm=NM, f_mode="dp"),
        dict(common, name="DP_rho0.02", alpha=ALPHA_LAB, rho=0.02, nm=NM, f_mode="dp"),
        # lab-only ablation: identical pipeline with the noise multiplier set to 0 (clip only)
        dict(common, name="OFF_nm0", alpha=0.0, rho=rho_star, nm=0.0, f_mode="none"),
        dict(common, name="ORACLE_nm0", alpha=ALPHA_LAB, rho=rho_star, nm=0.0, f_mode="oracle"),
        dict(common, name=f"DP_rho{rho_star}_nm0", alpha=ALPHA_LAB, rho=rho_star, nm=0.0, f_mode="dp"),
        # V6: band-MF(4, .95) + b-min-sep for 50 steps
        dict(common, name="MF_DP_rho{}".format(rho_star), alpha=ALPHA_LAB, rho=rho_star, nm=NM, f_mode="dp", noise="mf", n_steps=N_STEPS_MF),
    ]
    t5 = time.time()
    arm_results = []
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(2) as pool:
            arm_results = pool.map(run_arm, arms)
    except Exception as exc:  # noqa: BLE001
        log(f"  [pool failed: {type(exc).__name__}: {exc}; running arms sequentially]")
        arm_results = [run_arm(a) for a in arms]
    log(f"  arms done in {time.time() - t5:.0f}s")
    summaries = [summarize_arm(r) for r in arm_results]
    results["arms"] = summaries
    results["arm_hist"] = {r["name"]: {k: v for k, v in r["hist"].items() if k not in ("probe_noise_sample", "clipped_probe")} for r in arm_results}
    log("\n  per-arm summary")
    hdr = (f"  {'arm':20s} {'alpha':>5s} {'rho':>5s} {'nm':>6s} | {'d_0':>5s} {'d_T':>5s} {'CE_0':>6s} {'CE_T':>6s} | "
           f"{'cosB':>6s} {'cosPop':>6s} {'cosPopA':>7s} {'cosB/P':>6s} | {'dead%':>6s} {'trkPop':>6s} {'trkPopR':>7s} {'trkPopRA':>8s} {'s/kE':>6s} {'noise e/p':>9s} | {'clip%':>6s} {'pclip':>5s}")
    log(hdr)
    fmt = lambda v, w=6, d=3: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v == v else f"{'n/a':>{w}s}")
    for s in summaries:
        log(f"  {s['name']:20s} {s['alpha']:5.2f} {s['rho']:5.2f} {s['nm']:6.3f} | {s['delta_heldout'][0]:5.3f} {s['delta_heldout_final']:5.3f} {s['ce_heldout'][0]:6.3f} {s['ce_heldout_final']:6.3f} | "
            f"{fmt(s['mean_cos'])} {fmt(s['mean_cos_pop'])} {fmt(s['mean_cos_pop_active'], 7)} {fmt(s['mean_cos_batch_vs_pop'])} | {100 * s['dead_zone_rate']:6.1f} "
            f"{fmt(s['track_err_pop_mean'])} {fmt(s['track_err_pop_rescaled_mean'], 7)} {fmt(s['track_err_pop_rescaled_active'], 8)} {s['s_inf_over_kE']:6.3f} "
            f"{(s['pooled_noise_std_empirical'] / s['pooled_noise_std_predicted'] if s['pooled_noise_std_predicted'] > 0 else float('nan')):9.3f} | {100 * s['clip_rate_mean']:6.1f} {s['probe_clip_events']:5.0f}")
    log("  per-step batch-sampling std of the released signal (1/B)sum_x w_x d(x), per entry in k/E units, last 100 steps: " +
        ", ".join(f"{s['name']}={s['batch_signal_std_kE']:.3f}" for s in summaries))
    log("  cosB = cos(surrogate grad at f_used, exact batch aux grad at f(B_t)); cosPop = same vs the held-out population load; cosPopA = cosPop over steps with the dead zone NOT engaged;")
    log("  cosB/P = cos(batch aux grad, population aux grad) [how noisy f(B_t) itself is at B=32]; trkPop = ||f_used - f_pop|| / ||f_pop - k/E||; R = rescaled by T_max/T_bar_true; A = dead zone not engaged; noise e/p = empirical/predicted pooled probe-noise std")
    log("\n  delta_t on held-out every 25 steps:")
    for s in summaries:
        log(f"  {s['name']:22s} " + " ".join(f"{x:.3f}" for x in s["delta_heldout"]))
    log("  CE on held-out every 25 steps:")
    for s in summaries:
        log(f"  {s['name']:22s} " + " ".join(f"{x:.3f}" for x in s["ce_heldout"]))
    by = {s["name"]: s for s in summaries}
    dp_name = f"DP_rho{rho_star}"
    # V5 verdict: mechanism ran end to end; probe never clipped; probe param stayed 0; DP arm tracks the oracle
    off, orc, dp, dp02 = by["OFF"], by["ORACLE"], by[dp_name], by["DP_rho0.02"]
    recovered = (off["delta_heldout_final"] - dp["delta_heldout_final"]) / max(off["delta_heldout_final"] - orc["delta_heldout_final"], 1e-9)
    crit = dict(
        probe_never_clipped_and_param_zero=all(s["probe_clip_events"] == 0 and s["probe_param_final_nonzero"] == 0 for s in summaries),
        oracle_cos_batch_is_one=abs(orc["mean_cos"] - 1) < 1e-4,
        dp_recovers_half_of_oracle_reduction=recovered >= 0.5,
        dp_cos_pop_active_gt_0p8=(dp["mean_cos_pop_active"] or 0) > 0.8,
        rho002_dead_zone_engaged_majority=dp02["dead_zone_rate"] > 0.5,
        pooled_noise_matches_prediction=abs(dp["pooled_noise_std_empirical"] / dp["pooled_noise_std_predicted"] - 1) < 0.15,
    )
    log(f"\n  V5 criteria: {crit}; imbalance reduction recovered by DP vs ORACLE = {recovered:.3f}")
    checks["V5"] = dict(result="PASS" if all(crit.values()) else "FAIL", criteria=crit, recovered_fraction=recovered,
                        dp_mean_cos_batch=dp["mean_cos"], dp_mean_cos_pop=dp["mean_cos_pop"], dp_mean_cos_pop_active=dp["mean_cos_pop_active"],
                        dp_dead_rate=dp["dead_zone_rate"], rho002_dead_rate=dp02["dead_zone_rate"], rho002_mean_cos_pop=dp02["mean_cos_pop"],
                        dp_track_err_pop_rescaled_active=dp["track_err_pop_rescaled_active"],
                        delta_final={s["name"]: s["delta_heldout_final"] for s in summaries},
                        ce_final={s["name"]: s["ce_heldout_final"] for s in summaries},
                        surrogate_reduces_imbalance_nm0=by["ORACLE_nm0"]["delta_heldout_final"] < by["OFF_nm0"]["delta_heldout_final"],
                        surrogate_reduces_imbalance_dp=dp["delta_heldout_final"] < off["delta_heldout_final"])
    mf = next(r for r in arm_results if r["cfg"]["noise"] == "mf")
    log(f"\n  V6 band-MF(4,.95)+b-min-sep {N_STEPS_MF} steps: latch ok {mf['mf_latch_ok']}; realised noise_stddev vs base*||row_t(C^-1)|| max rel err: "
        f"probe {mf['mf_sigma_relerr_probe']:.2e}, fallback {mf['mf_sigma_relerr_fallback']:.2e}; probe received noise: {mf['mf_probe_received_noise']}; "
        f"empirical probe-noise RMS / predicted sigma = {mf['mf_probe_noise_empirical_std_over_pred']:.3f}; row norms t=0..4: {[round(x, 4) for x in mf['rownorm'][:5]]}; "
        f"phi_mf(last)={mf['phi_inf']:.4f} vs DP-SGD phi_inf={phi_inf:.4f}; batch size mean {by[mf['name']]['batch_size_mean']:.1f}")
    checks["V6"] = dict(result="PASS" if (mf["mf_latch_ok"] and mf["mf_sigma_relerr_probe"] < 1e-6 and mf["mf_sigma_relerr_fallback"] < 1e-6 and mf["mf_probe_received_noise"]) else "FAIL",
                        sigma_relerr_probe=mf["mf_sigma_relerr_probe"], sigma_relerr_fallback=mf["mf_sigma_relerr_fallback"],
                        empirical_over_pred=mf["mf_probe_noise_empirical_std_over_pred"], phi_mf=mf["phi_inf"], phi_sgd=phi_inf,
                        mean_cos=by[mf["name"]]["mean_cos"], dead_rate=by[mf["name"]]["dead_zone_rate"], delta_final=by[mf["name"]]["delta_heldout_final"])

    results["checks"] = checks
    results["elapsed_total_s"] = time.time() - T0
    log("\n" + "=" * 100)
    for k in sorted(checks):
        log(f"  {k}: {checks[k]['result']}")
    log(f"total elapsed {results['elapsed_total_s']:.0f}s")
    with open(os.path.join(OUT_DIR, "prototype_results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    log("wrote", os.path.join(OUT_DIR, "prototype_results.json"))


if __name__ == "__main__":
    main()
