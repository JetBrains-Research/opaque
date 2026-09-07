"""Shared helpers for the tiny-Mellum empirical experiments (CPU-only)."""
import copy, sys, time, json
from pathlib import Path
import torch

torch.set_num_threads(2)
ROOT = Path("/home/user/opaque")
sys.path.insert(0, str(ROOT / "packages/opaque-patches/tests/transformers/models"))
OUT = Path("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical")

TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=10000.0,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
)


def build_unpatched(seed=0, attn_impl="sdpa", **overrides):
    """Fresh, UNPATCHED HF MellumForCausalLM (module-level functions restored)."""
    from transformers.models.mellum.modeling_mellum import MellumForCausalLM
    from transformers.models.mellum.configuration_mellum import MellumConfig
    kw = dict(TINY); kw.update(overrides)
    cfg = MellumConfig(**kw)
    cfg._attn_implementation = attn_impl
    torch.manual_seed(seed)
    return MellumForCausalLM(cfg)


def build_patched(seed=0, attn_impl="sdpa", **overrides):
    """Same as tests: build_moe_model('mellum', 'cpu', ...) — applies opaque patches."""
    from _test_utils import build_moe_model
    kw = dict(TINY); kw.update(overrides)
    torch.manual_seed(seed)
    model, mod = build_moe_model("mellum", "cpu", attn_impl=attn_impl, **kw)
    return model, mod


def copy_weights(src, dst):
    dst.load_state_dict(src.state_dict())
    return dst


def random_batch(B, T, vocab=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab, (B, T), generator=g)
    return ids


def structured_batch(B, T, vocab=128, subset=6, seed=1):
    """Each example draws its tokens from its own small vocabulary subset
    (mimics within-document topical routing)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(B, T, dtype=torch.long)
    for b in range(B):
        sub = torch.randperm(vocab - 3, generator=g)[:subset] + 3
        ids[b] = sub[torch.randint(0, subset, (T,), generator=g)]
    return ids


class RouterCapture:
    """Forward hook on every *TopKRouter capturing (logits, scores, indices)."""
    def __init__(self, model, detach=True):
        from transformers.models.mellum.modeling_mellum import MellumTopKRouter
        self.records = []; self.detach = detach
        self.handles = [m.register_forward_hook(self._hook) for m in model.modules()
                        if isinstance(m, MellumTopKRouter)]
    def _hook(self, mod, inp, out):
        logits, scores, idx = out
        if self.detach: self.records.append((logits.detach(), scores.detach(), idx.detach()))
        else: self.records.append((logits, scores, idx))
    def clear(self):
        self.records = []
    def remove(self):
        for h in self.handles: h.remove()


def set_trainable(model, partition):
    """partition in {'attn','attn+router','attn+experts','all'}"""
    for n, p in model.named_parameters():
        attn = ".self_attn." in n and (".q_proj" in n or ".k_proj" in n or ".v_proj" in n or ".o_proj" in n)
        router = ".mlp.gate." in n
        experts = ".mlp.experts." in n
        if partition == "attn": p.requires_grad_(attn)
        elif partition == "attn+router": p.requires_grad_(attn or router)
        elif partition == "attn+experts": p.requires_grad_(attn or experts)
        elif partition == "all": p.requires_grad_(True)
        else: raise ValueError(partition)


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# vmap-safe (out-of-place) Switch-style load-balancing loss. Numerically the
# same formula as HF load_balancing_loss_func (modeling_mellum.py:540-608) but
# uses one_hot sums instead of bincount / in-place scatter_add_, so it can run
# per example under torch.func.vmap (with or without an attention mask).
# ---------------------------------------------------------------------------
def aux_loss_vmap_safe(gate_logits, num_experts=None, top_k=2, attention_mask=None):
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    counts = torch.zeros(num_experts, dtype=torch.float32)
    probsum = torch.zeros(num_experts, dtype=torch.float32)
    total = torch.zeros((), dtype=torch.float32)
    flat_mask = None if attention_mask is None else attention_mask.reshape(-1).to(torch.float32)
    for lg in gate_logits:
        p = torch.softmax(lg.to(torch.float32), dim=-1)
        _, sel = torch.topk(p, top_k, dim=-1)
        # one_hot via comparison (F.one_hot does data-dependent checks that break under vmap)
        oh = (sel[..., :, None] == torch.arange(num_experts)).to(torch.float32).sum(dim=-2)  # (rows, E)
        if flat_mask is not None:
            oh = oh * flat_mask[:, None]
            p = p * flat_mask[:, None]
            total = total + flat_mask.sum()
        else:
            total = total + float(lg.shape[0])
        counts = counts + oh.sum(0)
        probsum = probsum + p.sum(0)
    return num_experts * torch.sum((counts / total) * (probsum / total))


def load_stats_from_logits(gate_logits, num_experts, top_k, attention_mask=None):
    """Return (f, P, total_rows): load fractions, mean probs (pooled over layers)."""
    counts = torch.zeros(num_experts); probsum = torch.zeros(num_experts); total = 0.0
    flat_mask = None if attention_mask is None else attention_mask.reshape(-1).float()
    for lg in gate_logits:
        p = torch.softmax(lg.float(), -1)
        _, sel = torch.topk(p, top_k, -1)
        oh = (sel[..., :, None] == torch.arange(num_experts)).float().sum(-2)
        if flat_mask is not None:
            oh = oh * flat_mask[:, None]; p = p * flat_mask[:, None]; total = total + flat_mask.sum()
        else:
            total = total + float(lg.shape[0])
        counts = counts + oh.sum(0); probsum = probsum + p.sum(0)
    return counts / total, probsum / total, total


def topk_sets(logits, k):
    """Top-k index set per row from router logits (softmax is monotone; same as router)."""
    p = torch.softmax(logits.float(), -1)
    return torch.topk(p, k, -1).indices


def route_flip_fraction(logits_a, logits_b, k):
    """Fraction of rows whose top-k SET differs between two logit tensors."""
    sa = torch.sort(topk_sets(logits_a, k), -1).values
    sb = torch.sort(topk_sets(logits_b, k), -1).values
    return (sa != sb).any(-1).float().mean().item(), (sa != sb).any(-1)


def margins(logits, k):
    p = torch.softmax(logits.float(), -1)
    s = torch.sort(p, -1, descending=True).values
    return s[:, k - 1] - s[:, k]
