"""Exp B: HF load_balancing_loss_func recomputes top-k from softmax(logits) in the LOGITS dtype
(modeling_mellum.py: routing_weights = softmax(layer_gate, dim=-1)), whereas the forward router uses
softmax(..., dtype=float). For a bf16 model, does the aux loss's assignment count f(B) match the
routes actually executed in the forward? Synthetic bf16 logits, E=64, k=8, at several logit scales."""
import torch, json
torch.manual_seed(0)
E, k, N = 64, 8, 200_000
res = {}
for scale in (0.5, 1.0, 2.0, 4.0):
    z32 = torch.randn(N, E) * scale
    z16 = z32.to(torch.bfloat16)           # what F.linear produces in a bf16 model
    p_fwd = torch.softmax(z16, dim=-1, dtype=torch.float)   # forward router (fp32 softmax)
    p_aux = torch.softmax(z16, dim=-1)                       # aux func: softmax in logits dtype (bf16)
    top_fwd = torch.topk(p_fwd, k, dim=-1).indices
    top_aux = torch.topk(p_aux, k, dim=-1).indices
    same = (torch.sort(top_fwd, -1).values == torch.sort(top_aux, -1).values).all(-1)
    # count distinct-set mismatches and how many bf16-softmax ties exist at the k/k+1 boundary
    srt = torch.sort(p_aux, dim=-1, descending=True).values
    ties = (srt[:, k-1] == srt[:, k]).float().mean().item()
    f_fwd = torch.zeros(E).index_add_(0, top_fwd.reshape(-1), torch.ones(N*k)) / N
    f_aux = torch.zeros(E).index_add_(0, top_aux.reshape(-1), torch.ones(N*k)) / N
    res[str(scale)] = dict(frac_tokens_topk_set_differs=1-same.float().mean().item(),
                           frac_bf16_softmax_tie_at_boundary=ties,
                           max_abs_f_diff_over_kE=((f_fwd-f_aux).abs().max()/(k/E)).item(),
                           median_top1_prob=p_fwd.max(-1).values.median().item())
    print(scale, res[str(scale)])
json.dump(res, open("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/critic/expB.json","w"), indent=1)
