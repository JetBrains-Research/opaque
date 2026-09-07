"""refute-composition check 1: the design's PLAUSIBLE T5 claim.

Does the HF OutputRecorder path (backbone ``output_router_logits=True``) under
``vmap(grad)`` (a) return exactly L router-logit tensors per example, (b) do so
with opaque's non-reentrant gradient checkpointing WITHOUT a double append, and
(c) carry gradient through the captured logits (surrogate + probe) so that the
released probe leaf and the LoRA-side gradients equal the non-checkpointed run?

Also: (d) does a zeroed probe leaf stay zero through AdamW-BC with PerGroup
noise_stddev metadata (design §2.2 step 7 / risk 8)?
"""
import torch

torch.set_num_threads(2)
torch.manual_seed(0)
from transformers import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM, MellumTopKRouter

import opaque.patches as patches
from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
from opaque.random import key

cfg = MellumConfig(
    vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
    mlp_layer_types=["sparse", "sparse"], layer_types=["full_attention", "full_attention"],
    max_position_embeddings=64, router_aux_loss_coef=0.001, output_router_logits=False,
    sliding_window=16, use_cache=False,
)
E, K, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
B, T = 3, 12
ids = torch.randint(0, 64, (B, T))
mask = torch.ones(B, T, dtype=torch.long)
mask[1, 8:] = 0  # one ragged example
labels = ids.clone()
labels[mask == 0] = -100
ALPHA = 1e-3
LAM = 0.1
f_tilde = torch.full((E,), K / E) + torch.tensor([0.05, -0.05, 0.02, -0.02])


def build(ckpt: bool):
    torch.manual_seed(0)
    model = MellumForCausalLM(cfg).float()
    patches.apply_runtime_patches()
    patches.apply_model_patches(model)
    if ckpt:
        model.gradient_checkpointing_enable()
    for n_, p in model.named_parameters():
        p.requires_grad = ("q_proj" in n_ or "v_proj" in n_)
    # a hook counter to detect double-firing under recompute
    fired = {"n": 0}
    for m in model.modules():
        if isinstance(m, MellumTopKRouter):
            m.register_forward_hook(lambda *_: fired.__setitem__("n", fired["n"] + 1))
    fmodel, trainable, frozen = make_functional(
        model.model, disable_autograd_tracking=True, partition_trainable=True
    )
    lm_w = model.lm_head.weight.detach()
    trainable = dict(trainable, router_load_probe=torch.zeros(E))
    n_rec = {"n": None}

    def loss_fn(params, ids1, mask1, lab1):
        p_ = {k: v for k, v in params.items() if k != "router_load_probe"}
        out = fmodel({**frozen, **p_}, input_ids=ids1[None], attention_mask=mask1[None],
                     output_router_logits=True)
        rl = out.router_logits
        n_rec["n"] = len(rl)
        hid = out.last_hidden_state[0]
        logits = hid @ lm_w.T
        ce = torch.nn.functional.cross_entropy(logits[:-1], lab1[1:], ignore_index=-100)
        m = mask1.float()
        Tx = m.sum().clamp_min(1.0)
        P = torch.zeros(E)
        h = torch.zeros(E)
        for z in rl:  # (T, E)
            p = torch.softmax(z.float(), -1)
            P = P + (p * m[:, None]).sum(0)
            top = torch.topk(p, K, -1).indices
            oh = (top[..., None] == torch.arange(E)).float().sum(1)  # (T, E)
            h = h + (oh * m[:, None]).sum(0)
        P = P / (L * Tx)
        h = h / (L * Tx)
        d = h - K / E
        sur = E * ((f_tilde - K / E) * P).sum()
        probe = (params["router_load_probe"] * (LAM * d).detach()).sum()
        return ce + ALPHA * sur + probe, {"d": d.detach(), "h": h.detach()}

    return model, fmodel, trainable, loss_fn, fired, n_rec


res = {}
for ckpt in (False, True):
    model, fmodel, trainable, loss_fn, fired, n_rec = build(ckpt)
    pg = per_group(trainable, router_load_probe=LAM * (K * (1 - K / E)) ** 0.5 * (1 + 1e-6), fallback=1.0)
    gf, st = clipped_grad(loss_fn, argnums=0, has_aux=True, batch_argnums=(1, 2, 3),
                          clipping_norm=pg, normalize_by=B, return_aux=True)
    fired["n"] = 0
    (g, aux), st = gf(trainable, ids, mask, labels, state=st)
    res[ckpt] = (g, aux, fired["n"], n_rec["n"])
    print(f"ckpt={ckpt}: recorder returned {n_rec['n']} router-logit tensors (L={L}); "
          f"router hook fired {fired['n']} times for B={B} (expect {B * L} fwd"
          f"{' + ' + str(B * L) + ' recompute' if ckpt else ''});"
          f" group_norms[probe] max={aux.group_norms['router_load_probe'].max():.5f} "
          f"<= C_h={pg.values['router_load_probe']:.5f}: "
          f"{bool((aux.group_norms['router_load_probe'] <= pg.values['router_load_probe']).all())}")

g0, a0, _, _ = res[False]
g1, a1, _, _ = res[True]
for k in g0.pytree:
    diff = (g0.pytree[k] - g1.pytree[k]).norm() / (g0.pytree[k].norm() + 1e-30)
    print(f"  rel-L2 ckpt vs no-ckpt  {k:45s} {diff:.2e}  (|g|={g0.pytree[k].norm():.4f})")
print("  probe leaf (= (LAM/B) sum_x d_x):", g0.pytree["router_load_probe"])
print("  sum_e probe leaf (expect 0):", float(g0.pytree["router_load_probe"].sum()))
print("  per-example d sums (expect 0):", a0.loss_aux["d"].sum(-1))
print("  probe leaf == (LAM/B)*sum d_x:",
      torch.allclose(g0.pytree["router_load_probe"], LAM * a0.loss_aux["d"].sum(0) / B, atol=1e-6))

# (d) AdamW-BC with a zeroed probe leaf and PerGroup noise_stddev
from opaque.optimizers import adamw

nf, ns = gaussian_noise(noise_multiplier=1.0, key=key(0))
noisy, ns = nf(g0, ns)
print("  noise_stddev per group:", dict(noisy.noise_stddev.values))
noisy.pytree["router_load_probe"].zero_()  # design §2.2 step 7
opt = adamw(learning_rate=1e-2, weight_decay=0.1, noise_bias_correction=True)
params = {k: v.clone() for k, v in trainable.items()}
ost = opt.init(params)
for _ in range(3):
    upd, ost = opt.update(noisy, ost, params=params)
    print("  AdamW-BC probe update:", upd["router_load_probe"], " finite:",
          bool(torch.isfinite(upd["router_load_probe"]).all()))
