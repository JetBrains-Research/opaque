import copy, math, sys, torch
torch.set_num_threads(2)
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
def log(*a): print(*a, flush=True)
def rel_l2(a, b):
    a = a.float().flatten(); b = b.float().flatten(); den = b.norm().item(); return (a - b).norm().item() / den if den > 0 else (a-b).norm().item()
T = 16; window = int(sys.argv[1]) if len(sys.argv) > 1 else 8
kw = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, num_experts=8, num_experts_per_tok=2,
          moe_intermediate_size=64, norm_topk_prob=True, layer_types=["sliding_attention", "full_attention"], sliding_window=window)
torch.manual_seed(0)
cfg = MellumConfig(**kw); cfg._attn_implementation = "sdpa"
ref = MellumForCausalLM(cfg).train()
ids = torch.randint(3, cfg.vocab_size, (1, T))
from opaque.patches import apply_runtime_patches, apply_model_patches
def hf_hidden(model, mask, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    return [h[0].detach() for h in out.hidden_states], out.logits[0].detach()
res = {}
for npad in range(0, 8):
    mask = torch.ones(1, T, dtype=torch.long); mask[0, :npad] = 0
    hs_g, lg_g = hf_hidden(copy.deepcopy(ref), mask, True)
    hs_n, lg_n = hf_hidden(copy.deepcopy(ref), mask, False)
    # eager attention HF
    ref_e = copy.deepcopy(ref); ref_e.config._attn_implementation = "eager"
    hs_e, lg_e = hf_hidden(ref_e, mask, True)
    res[npad] = (hs_g, lg_g, hs_n, lg_n, hs_e, lg_e, mask)
apply_runtime_patches(compat=True)
m = copy.deepcopy(ref); apply_model_patches(m, compat=True, performance=True, kernels=False)
backbone = m.model; params = {**dict(backbone.named_parameters()), **dict(backbone.named_buffers())}
def f(i, mk):
    out = torch.func.functional_call(backbone, params, (), {"input_ids": i[None], "attention_mask": mk[None], "output_hidden_states": True})
    return tuple(h[0] for h in out.hidden_states)
for npad, (hs_g, lg_g, hs_n, lg_n, hs_e, lg_e, mask) in res.items():
    with torch.no_grad():
        hs_v = torch.func.vmap(f, in_dims=(0, 0))(ids, mask)
    hs_v = [h[0] for h in hs_v]
    valid = mask[0] == 1
    l0_valid = rel_l2(hs_v[1][valid], hs_g[1][valid]); l0_pad = rel_l2(hs_v[1][~valid], hs_g[1][~valid]) if npad else float("nan")
    fin = [bool(torch.isfinite(hs_g[1]).all()), bool(torch.isfinite(hs_n[1]).all()), bool(torch.isfinite(hs_v[1]).all())]
    log(f"window={window} npad={npad}: L0 valid-pos rel-L2 vmap-vs-HF(grad) {l0_valid:.1e}; pad-pos {l0_pad:.1e}; "
        f"HF grad-vs-nograd L0 {rel_l2(hs_g[1], hs_n[1]):.1e}; HF sdpa-vs-eager L0 valid {rel_l2(hs_g[1][valid], hs_e[1][valid]):.1e}; "
        f"final valid: vmap-vs-HF {rel_l2(hs_v[2][valid], hs_g[2][valid]):.1e}, HF sdpa-vs-eager {rel_l2(hs_g[2][valid], hs_e[2][valid]):.1e}; finite(HFgrad,HFnograd,vmap)={fin}")
    if npad and l0_valid > 1e-5:
        per_pos = [rel_l2(hs_v[1][t], hs_g[1][t]) for t in range(T)]
        log("    per-position L0 rel-L2:", ["%.0e" % e for e in per_pos])
        per_pos_e = [rel_l2(hs_e[1][t], hs_g[1][t]) for t in range(T)]
        log("    per-position L0 HF eager-vs-sdpa:", ["%.0e" % e for e in per_pos_e])
