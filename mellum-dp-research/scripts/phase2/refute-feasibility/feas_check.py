"""CPU feasibility check for the phase-2 final design's statistics/probe path.

2-layer tiny Mellum (E=8, k=2) built with the repo's build_moe_model (patched exactly
like the Mellum tests). Emulates the design's `opaque_router_logits=True` forward
(backbone with output_router_logits=True + chunked CE; no HF aux), the vmap-safe
router statistics, the probe leaf, and checks: vmap(grad) vs eager loop, recorder
count under gradient checkpointing (double-fire?) and gradient equality through the
checkpoint region, clipped_grad PerGroup + microbatching, gaussian / mf_gaussian per-group
sigma, surrogate identity vs HF load_balancing_loss_func on the patched model.
"""
import json
import math
import sys
import time

import torch

torch.set_num_threads(4)
sys.path.insert(0, "/home/user/opaque/packages/opaque-patches/tests/transformers/models")
from _test_utils import build_moe_model  # noqa: E402

E, K, L = 8, 2, 2
TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=10000.0,
    num_experts=E, num_experts_per_tok=K, moe_intermediate_size=32,
)
RESULTS = {}
t_start = time.time()

torch.manual_seed(0)
model, mod = build_moe_model("mellum", "cpu", **TINY)
model.train()
for n, p in model.named_parameters():
    p.requires_grad_(any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")))
model.register_parameter("router_load_probe", torch.nn.Parameter(torch.zeros(E)))
RESULTS["experts_impl"] = getattr(model.config, "_experts_implementation", None)
RESULTS["experts_forward_patched"] = hasattr(mod.MellumExperts.forward, "__opaque_patched__")

# ---- emulate the design's `opaque_router_logits` kwarg on the fused/chunked-CE forward
from opaque.api.patches.kernels._linear_ce_chunked import linear_nll_sum_chunked  # noqa: E402

CausalLM = type(model)
_orig_forward = CausalLM.forward
RESULTS["causal_lm_forward_is_opaque_patched"] = hasattr(_orig_forward, "__opaque_patched__")


def forward(self, input_ids=None, attention_mask=None, labels=None, opaque_router_logits=False, **kw):
    if not opaque_router_logits:
        return _orig_forward(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
    hidden = outputs[0]
    nll = linear_nll_sum_chunked(hidden, self.lm_head.weight, labels, -100, 0, 0.0, False, chunk_vocab=None)
    n_valid = (labels[..., 1:] != -100).sum().float().clamp(min=1)
    return {"loss": nll / n_valid, "router_logits": outputs.router_logits}


CausalLM.forward = forward


# ---- vmap-safe statistics (design §9.1 moe_stats)
def router_load_and_probs(router_logits, attention_mask, top_k):
    num_experts = router_logits[0].shape[-1]
    h = torch.zeros(num_experts, dtype=torch.float32)
    P = torch.zeros(num_experts, dtype=torch.float32)
    m = attention_mask.reshape(-1).to(torch.float32) if attention_mask is not None else None
    for z in router_logits:
        z = z.reshape(-1, num_experts)
        p = torch.softmax(z.float(), dim=-1)
        _, idx = torch.topk(p, top_k, dim=-1)
        onehot = (idx[..., None] == torch.arange(num_experts)).sum(-2).float()
        if m is not None:
            onehot = onehot * m[:, None]
            p = p * m[:, None]
        h = h + onehot.sum(0)
        P = P + p.sum(0)
    T = m.sum() if m is not None else torch.tensor(float(z.shape[0]))
    denom = len(router_logits) * T
    return h / denom, P / denom


from opaque.functional import make_functional  # noqa: E402

fmodel, trainable, frozen = make_functional(model, disable_autograd_tracking=True, partition_trainable=True)
RESULTS["trainable_keys"] = sorted(trainable.keys())[:3] + ["...", "router_load_probe" in trainable]

C_g = 0.9
rho = 0.02
Delta_h = math.sqrt(K * (1 - K / E))
lam = rho * C_g / Delta_h
C_h = lam * Delta_h * (1 + 1e-6)
alpha = 1e-4
torch.manual_seed(1)
f_tilde = torch.full((E,), K / E) + 0.05 * torch.randn(E)
f_tilde = f_tilde - (f_tilde.mean() - K / E)


def loss_fn(trainable, input_ids, attention_mask, labels, alpha=alpha):
    params = {**frozen, **trainable}
    out = fmodel(params, input_ids=input_ids[None], attention_mask=attention_mask[None],
                 labels=labels[None], opaque_router_logits=True)
    h, P = router_load_and_probs(out["router_logits"], attention_mask, K)
    d = h - K / E
    loss = out["loss"] + alpha * E * ((f_tilde - K / E) * P).sum() \
        + (trainable["router_load_probe"] * (lam * d).detach()).sum()
    nlog = torch.tensor(float(len(out["router_logits"])))
    return loss, (h, P, nlog, out["loss"])


def loss_only(trainable, input_ids, attention_mask, labels):
    return loss_fn(trainable, input_ids, attention_mask, labels)[0]


B, T = 4, 16
g = torch.Generator().manual_seed(2)
ids = torch.randint(3, 128, (B, T), generator=g)
lengths = [T, 12, 10, T]
mask = torch.zeros(B, T, dtype=torch.long)
for i, ln in enumerate(lengths):
    mask[i, :ln] = 1
ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def flat(gdict):
    return torch.cat([v.reshape(-1) for k, v in sorted(gdict.items())])


# ---- A) vmap(grad)
vg = torch.func.vmap(torch.func.grad(loss_fn, has_aux=True), in_dims=(None, 0, 0, 0))
t0 = time.time()
grads_v, (h_v, P_v, nlog_v, ce_v) = vg(trainable, ids, mask, labels)
RESULTS["A_vmap_ok"] = True
RESULTS["A_vmap_time_s"] = round(time.time() - t0, 3)
RESULTS["A_nlog_per_example"] = nlog_v.tolist()
RESULTS["A_probe_grad_eq_lam_d"] = torch.allclose(grads_v["router_load_probe"], lam * (h_v - K / E), atol=1e-7)
RESULTS["A_sum_h_eq_k"] = torch.allclose(h_v.sum(-1), torch.full((B,), float(K)), atol=1e-5)
RESULTS["A_sum_P_eq_1"] = torch.allclose(P_v.sum(-1), torch.ones(B), atol=1e-5)
RESULTS["A_norm_d_max"] = (h_v - K / E).norm(dim=-1).max().item()
RESULTS["A_bound_Delta_h"] = Delta_h

# ---- B) eager per-example loop (functional, no vmap) and plain autograd on the module
h_e, P_e, gl = [], [], []
for i in range(B):
    gi, (hi, Pi, _, _) = torch.func.grad(loss_fn, has_aux=True)(trainable, ids[i], mask[i], labels[i])
    h_e.append(hi); P_e.append(Pi); gl.append(gi)
h_e = torch.stack(h_e); P_e = torch.stack(P_e)
RESULTS["B_h_vmap_vs_eager_maxabs"] = (h_v - h_e).abs().max().item()
RESULTS["B_P_vmap_vs_eager_maxabs"] = (P_v - P_e).abs().max().item()
RESULTS["B_grad_vmap_vs_eager_rel_per_example"] = [
    rel(flat({k: v[i] for k, v in grads_v.items()}), flat(gl[i])) for i in range(B)
]
# plain autograd on the module (HF-loop analogue at equal precision, same patched module)
plain = []
for i in range(B):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1], labels=labels[i:i + 1], opaque_router_logits=True)
    h, P = router_load_and_probs(out["router_logits"], mask[i], K)
    loss = out["loss"] + alpha * E * ((f_tilde - K / E) * P).sum() \
        + (model.router_load_probe * (lam * (h - K / E)).detach()).sum()
    loss.backward()
    plain.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
RESULTS["B_grad_vmap_vs_module_backward_rel"] = [
    rel(flat({k: v[i] for k, v in grads_v.items()}), flat(plain[i])) for i in range(B)
]
model.zero_grad(set_to_none=True)

# ---- C) clipped_grad with PerGroup + microbatching
from opaque.dpsgd.clipping import clipped_grad, per_group  # noqa: E402

pg = per_group(trainable, router_load_probe=C_h, fallback=C_g)
RESULTS["C_pergroup_values"] = dict(pg.values)
B_bar = 5.0
res_c = {}
for mb in (None, 2, 3):
    grad_fn, clip_state = clipped_grad(loss_only, clipping_norm=pg, normalize_by=B_bar,
                                       batch_argnums=(1, 2, 3), return_aux=True, microbatch_size=mb)
    (clipped, aux), clip_state = grad_fn(trainable, ids, mask, labels, state=clip_state)
    res_c[str(mb)] = clipped
    probe_leaf = clipped.pytree["router_load_probe"]
    expected = (lam / B_bar) * (h_v - K / E).sum(0)
    RESULTS[f"C_mb{mb}_probe_leaf_eq_expected_maxabs"] = (probe_leaf - expected).abs().max().item()
    RESULTS[f"C_mb{mb}_probe_group_norm_max"] = aux.group_norms["router_load_probe"].max().item()
    RESULTS[f"C_mb{mb}_probe_clip_rate"] = float((aux.group_norms["router_load_probe"] > C_h).sum().item()) / B
    RESULTS[f"C_mb{mb}_fallback_norms"] = [round(x, 4) for x in aux.group_norms["fallback"].tolist()]
    RESULTS[f"C_mb{mb}_max_norm_values"] = {k: float(v) for k, v in clipped.max_norm.values.items()}
RESULTS["C_mb2_vs_none_rel"] = rel(flat(res_c["2"].pytree), flat(res_c["None"].pytree))
RESULTS["C_mb3_vs_none_rel"] = rel(flat(res_c["3"].pytree), flat(res_c["None"].pytree))
RESULTS["C_probe_sum_zero"] = res_c["None"].pytree["router_load_probe"].sum().item()

# ---- D) noise allocation: gaussian_noise and mf_gaussian_noise (PerGroup latch over steps)
from opaque.random import key  # noqa: E402
from opaque.dpsgd.noise import gaussian_noise  # noqa: E402
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise  # noqa: E402

nm = 0.5622
noise_fn, nstate = gaussian_noise(noise_multiplier=nm, key=key(0))
noised, nstate = noise_fn(res_c["None"], nstate)
S = C_g + C_h
RESULTS["D_gauss_sigma"] = {k: float(v) for k, v in noised.noise_stddev.values.items()}
RESULTS["D_gauss_sigma_closed_form"] = {"fallback": nm * math.sqrt(C_g * S) / B_bar,
                                        "router_load_probe": nm * math.sqrt(C_h * S) / B_bar}
sg, sh = noised.noise_stddev.values["fallback"], noised.noise_stddev.values["router_load_probe"]
RESULTS["D_mahalanobis_nm2"] = ((C_g / B_bar) ** 2 / sg ** 2 + (C_h / B_bar) ** 2 / sh ** 2) * nm ** 2
RESULTS["D_probe_noise_std_over_lam_in_kE_units"] = (sh / lam) / (K / E)

mf_fn, mf_state = mf_gaussian_noise(trainable, band_mf_strategy(bands=4, momentum=0.95), n_steps=6,
                                    noise_multiplier=nm, key=key(1))
mf_sig = []
try:
    for step in range(3):
        noised_mf, mf_state = mf_fn(res_c["None"], mf_state)
        mf_sig.append({k: float(v) for k, v in noised_mf.noise_stddev.values.items()})
    RESULTS["D_mf_latch_ok_3_steps"] = True
    RESULTS["D_mf_sigma_per_step"] = mf_sig
except Exception as e:  # noqa: BLE001
    RESULTS["D_mf_latch_ok_3_steps"] = f"FAILED: {type(e).__name__}: {e}"

# ---- E) gradient checkpointing (non-reentrant, opaque HF glue)
model.gradient_checkpointing_enable()
RESULTS["E_ckpt_flags"] = [bool(getattr(m, "gradient_checkpointing", False)) for m in model.model.layers]
try:
    t0 = time.time()
    grads_c, (h_c, P_c, nlog_c, ce_c) = vg(trainable, ids, mask, labels)
    RESULTS["E_ckpt_vmap_ok"] = True
    RESULTS["E_ckpt_time_s"] = round(time.time() - t0, 3)
    RESULTS["E_ckpt_nlog_per_example"] = nlog_c.tolist()
    RESULTS["E_ckpt_h_maxabs_vs_nockpt"] = (h_c - h_v).abs().max().item()
    RESULTS["E_ckpt_grad_rel_vs_nockpt"] = rel(flat(grads_c), flat(grads_v))
    RESULTS["E_ckpt_probe_grad_eq"] = torch.allclose(grads_c["router_load_probe"], grads_v["router_load_probe"], atol=1e-7)
    # does the surrogate term (through router logits captured inside the checkpoint region) carry gradient?
    g_a0 = torch.func.vmap(torch.func.grad(lambda *a: loss_fn(*a, alpha=0.0)[0]), in_dims=(None, 0, 0, 0))(trainable, ids, mask, labels)
    g_a1 = torch.func.vmap(torch.func.grad(lambda *a: loss_fn(*a, alpha=1.0)[0]), in_dims=(None, 0, 0, 0))(trainable, ids, mask, labels)
    diff_ckpt = flat({k: v for k, v in g_a1.items() if k != "router_load_probe"}) - flat({k: v for k, v in g_a0.items() if k != "router_load_probe"})
    model.gradient_checkpointing_disable()
    g_a0n = torch.func.vmap(torch.func.grad(lambda *a: loss_fn(*a, alpha=0.0)[0]), in_dims=(None, 0, 0, 0))(trainable, ids, mask, labels)
    g_a1n = torch.func.vmap(torch.func.grad(lambda *a: loss_fn(*a, alpha=1.0)[0]), in_dims=(None, 0, 0, 0))(trainable, ids, mask, labels)
    diff_nock = flat({k: v for k, v in g_a1n.items() if k != "router_load_probe"}) - flat({k: v for k, v in g_a0n.items() if k != "router_load_probe"})
    RESULTS["E_surrogate_grad_norm_ckpt"] = diff_ckpt.norm().item()
    RESULTS["E_surrogate_grad_norm_nockpt"] = diff_nock.norm().item()
    RESULTS["E_surrogate_grad_rel_ckpt_vs_nockpt"] = rel(diff_ckpt, diff_nock)
except Exception as e:  # noqa: BLE001
    import traceback
    RESULTS["E_ckpt_vmap_ok"] = f"FAILED: {type(e).__name__}: {e}"
    RESULTS["E_traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    model.gradient_checkpointing_disable()

# ---- F) surrogate identity vs HF batched aux on the patched model (fp32), ragged lengths
model.zero_grad(set_to_none=True)
out_b = model.model(input_ids=ids, attention_mask=mask, output_router_logits=True)
aux_hf = mod.load_balancing_loss_func(out_b.router_logits, E, K, mask)
aux_hf.backward()
g_hf = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
model.zero_grad(set_to_none=True)
# f(B): pooled load from the same logits with the flat mask (HF definition)
fB, _ = router_load_and_probs(out_b.router_logits, mask.reshape(-1), K)
T_tot = mask.sum().float()
out_b2 = model.model(input_ids=ids, attention_mask=mask, output_router_logits=True)
surr = 0.0
for i in range(B):
    per_ex_logits = [z.reshape(B, T, E)[i] for z in out_b2.router_logits]
    _, Pi = router_load_and_probs(per_ex_logits, mask[i], K)
    w = mask[i].sum().float() / T_tot
    surr = surr + E * w * (fB * Pi).sum()
surr.backward()
g_su = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
RESULTS["F_aux_hf_value"] = aux_hf.item()
RESULTS["F_surrogate_value"] = surr.item()
RESULTS["F_fB_sum"] = fB.sum().item()
RESULTS["F_grad_rel_surrogate_vs_hf"] = rel(flat(g_su), flat(g_hf))
# equal-example-weight variant (Opaque convention) vs HF token weighting under ragged lengths
model.zero_grad(set_to_none=True)
out_b3 = model.model(input_ids=ids, attention_mask=mask, output_router_logits=True)
surr_eq = 0.0
for i in range(B):
    per_ex_logits = [z.reshape(B, T, E)[i] for z in out_b3.router_logits]
    _, Pi = router_load_and_probs(per_ex_logits, mask[i], K)
    surr_eq = surr_eq + E * (1.0 / B) * (fB * Pi).sum()
surr_eq.backward()
g_eq = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
RESULTS["F_grad_rel_equalweight_vs_hf_ragged"] = rel(flat(g_eq), flat(g_hf))
RESULTS["F_lengths"] = lengths
model.zero_grad(set_to_none=True)

# ---- G) HF's own output_router_logits=True under vmap with a mask (the path the design rejects)
CausalLM.forward = _orig_forward
def hf_loss(trainable, input_ids, attention_mask, labels):
    params = {**frozen, **trainable}
    out = fmodel(params, input_ids=input_ids[None], attention_mask=attention_mask[None], labels=labels[None], output_router_logits=True)
    return out["loss"]
try:
    torch.func.vmap(torch.func.grad(hf_loss), in_dims=(None, 0, 0, 0))(trainable, ids, mask, labels)
    RESULTS["G_hf_output_router_logits_under_vmap"] = "ran (unexpected)"
except Exception as e:  # noqa: BLE001
    RESULTS["G_hf_output_router_logits_under_vmap"] = f"{type(e).__name__}: {str(e)[:120]}"

RESULTS["total_time_s"] = round(time.time() - t_start, 1)
print(json.dumps(RESULTS, indent=1, default=str))
