"""K4: loss = CE + alpha*(S - S.detach()) has value bit-identical to CE and gradient identical to
grad(CE + alpha*S), under torch.func.grad and under vmap."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, grad_and_value, vmap

torch.set_num_threads(2)
torch.manual_seed(0)
D, H, V, E, k, T, B = 16, 32, 40, 8, 2, 12, 6
ALPHA = 1e-3


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.router = nn.Linear(D, E, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(D, H), nn.SiLU(), nn.Linear(H, D)) for _ in range(E)])
        self.head = nn.Linear(D, V)

    def forward(self, ids):
        x = self.emb(ids)                                  # (T, D)
        logits = self.router(x)
        p = torch.softmax(logits.float(), -1)
        idx = torch.topk(p, k, -1).indices
        onehot = (torch.arange(E)[None, None, :] == idx[..., None]).any(-2).float()
        wts = p * onehot
        wts = wts / wts.sum(-1, keepdim=True)
        out = sum(wts[:, e:e + 1] * self.experts[e](x) for e in range(E))
        return self.head(out), p


model = Tiny()
params = {n: p.detach() for n, p in model.named_parameters()}
ids = torch.randint(0, V, (B, T))
labels = torch.randint(0, V, (B, T))
f_tilde = torch.softmax(torch.randn(E) * 2, 0) * k  # imbalanced load vector
w_x = torch.tensor(1.0)


def parts(p, ids_x, labels_x):
    logits, probs = functional_call(model, p, (ids_x,))
    ce = F.cross_entropy(logits, labels_x)
    P = probs.mean(0)
    S = E * w_x * ((f_tilde - k / E) * P).sum()
    return ce, S


def loss_ce(p, ids_x, labels_x):
    return parts(p, ids_x, labels_x)[0]


def loss_neutral(p, ids_x, labels_x):
    ce, S = parts(p, ids_x, labels_x)
    return ce + ALPHA * (S - S.detach())


def loss_full(p, ids_x, labels_x):
    ce, S = parts(p, ids_x, labels_x)
    return ce + ALPHA * S


def compare(tag, g_a, v_a, g_b, v_b, g_ce, v_ce):
    same_val = torch.equal(v_a, v_ce)
    maxdiff = max(float((g_a[n] - g_b[n]).abs().max()) for n in g_a)
    bit = all(torch.equal(g_a[n], g_b[n]) for n in g_a)
    gnorm = max(float(g_b[n].abs().max()) for n in g_b)
    s_contrib = max(float((g_b[n] - g_ce[n]).abs().max()) for n in g_b)
    print(f"{tag}: value(neutral)==value(CE) bit-identical: {same_val}; value(full)-value(CE) = {float((v_b - v_ce).abs().max()):.3e}")
    print(f"      grad(neutral) vs grad(full): bit-identical={bit}, max|diff|={maxdiff:.3e} (max|grad|={gnorm:.3e}); "
          f"alpha*grad S contribution max={s_contrib:.3e} (nonzero => the surrogate gradient is really there)")
    return same_val and maxdiff == 0.0


ok = True
for dtype in (torch.float32, torch.float64):
    p = {n: t.to(dtype) for n, t in params.items()}
    model.to(dtype)
    f_tilde = f_tilde.to(dtype); w_x = w_x.to(dtype)
    # single example, torch.func.grad
    g_n, v_n = grad_and_value(loss_neutral)(p, ids[0], labels[0])
    g_f, v_f = grad_and_value(loss_full)(p, ids[0], labels[0])
    g_c, v_c = grad_and_value(loss_ce)(p, ids[0], labels[0])
    ok &= compare(f"[{dtype}] grad ", g_n, v_n, g_f, v_f, g_c, v_c)
    # vmap over the batch
    vg = vmap(grad_and_value(loss_neutral), in_dims=(None, 0, 0))
    g_n, v_n = vg(p, ids, labels)
    g_f, v_f = vmap(grad_and_value(loss_full), in_dims=(None, 0, 0))(p, ids, labels)
    g_c, v_c = vmap(grad_and_value(loss_ce), in_dims=(None, 0, 0))(p, ids, labels)
    ok &= compare(f"[{dtype}] vmap ", g_n, v_n, g_f, v_f, g_c, v_c)
    # per-example vmap grad vs loop grad (vmap exactness for this neutral loss)
    loop = [grad_and_value(loss_neutral)(p, ids[i], labels[i]) for i in range(B)]
    md = max(float((g_n[n][i] - loop[i][0][n]).abs().max()) for n in g_n for i in range(B))
    print(f"      vmap vs per-example loop grad max|diff| = {md:.3e}")
print("K4 RESULT:", "PASS" if ok else "FAIL")
