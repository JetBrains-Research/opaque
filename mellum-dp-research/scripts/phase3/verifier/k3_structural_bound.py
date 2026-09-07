"""K3: brute-force the per-layer centred load bound ||d^{(L,E)}(x)||_2 <= sqrt(k*L*(1-k/E)) and the
replace-one bound, with the §1.1 definitions and a binarised mask; random + adversarial routings."""
import itertools, math
import numpy as np
import torch

torch.set_num_threads(2)
rng = np.random.default_rng(0)


def load_vec(logits, mask, k):
    """logits: (L, T, E) fp32; mask: (T,) any dtype -> d^(L,E) per §1.1 with explicit T_x=0 -> 0."""
    L, T, E = logits.shape
    m = (mask != 0).to(torch.float32)
    T_x = m.sum()
    p = torch.softmax(logits.float(), dim=-1)
    idx = torch.topk(p, k, dim=-1).indices                       # (L,T,k) executed set
    onehot = (torch.arange(E)[None, None, None, :] == idx[..., None]).any(-2).to(torch.float32)  # (L,T,E)
    h = torch.einsum("t,lte->le", m, onehot) / T_x.clamp(min=1)
    d = h - k / E
    d = torch.where(T_x > 0, d, torch.zeros_like(d))
    return d, T_x


results = []
for E in (8, 64):
    for k in (2, 8):
        L = 2
        bound = math.sqrt(k * L * (1 - k / E))
        maxn = 0.0
        # random routings, random masks, random T
        for trial in range(400):
            T = int(rng.integers(1, 65))
            logits = torch.tensor(rng.normal(size=(L, T, E)) * rng.choice([0.1, 1.0, 10.0]), dtype=torch.float32)
            mask = torch.tensor(rng.random(T) < rng.choice([0.3, 0.7, 1.0]))
            d, T_x = load_vec(logits, mask, k)
            n = float(d.norm())
            assert n <= bound + 1e-6, (E, k, n, bound)
            if T_x == 0:
                assert n == 0.0
            maxn = max(maxn, n)
        # adversarial: every token, every layer routes to the same k experts (sharp logits)
        T = 64
        logits = torch.full((L, T, E), -20.0)
        S = rng.choice(E, size=k, replace=False)
        logits[..., S] = 20.0
        d_adv, _ = load_vec(logits, torch.ones(T), k)
        n_adv = float(d_adv.norm())
        # adversarial: T=1 with random logits (any single token attains the vertex)
        d_one, _ = load_vec(torch.tensor(rng.normal(size=(L, 1, E)), dtype=torch.float32), torch.ones(1), k)
        n_one = float(d_one.norm())
        # exhaustive over all k-subset vertex combos for small E (E=8): each layer independently a vertex
        if E == 8:
            verts = [np.isin(np.arange(E), c).astype(float) - k / E for c in itertools.combinations(range(E), k)]
            ex_max = max(math.sqrt(sum(float(v @ v) for v in vs)) for vs in itertools.product(verts, repeat=L))
        else:
            ex_max = float("nan")
        # replace-one: max over pairs. random pairs + adversarial disjoint sets
        ro_max = 0.0
        for trial in range(300):
            T1, T2 = int(rng.integers(1, 65)), int(rng.integers(1, 65))
            l1 = torch.tensor(rng.normal(size=(L, T1, E)) * 5, dtype=torch.float32)
            l2 = torch.tensor(rng.normal(size=(L, T2, E)) * 5, dtype=torch.float32)
            d1, T1x = load_vec(l1, torch.ones(T1), k); d2, T2x = load_vec(l2, torch.ones(T2), k)
            w1, w2 = float(T1x) / 64, float(T2x) / 64  # T_bar = T_max = 64
            ro_max = max(ro_max, float((w1 * d1 - w2 * d2).norm()))
        # adversarial disjoint subsets (or maximally different when E < 2k), full weight
        S1 = np.arange(k); S2 = (np.arange(k) + k) % E if E >= 2 * k else np.arange(E - k, E)
        la = torch.full((L, 8, E), -20.0); la[..., S1] = 20.0
        lb = torch.full((L, 8, E), -20.0); lb[..., S2] = 20.0
        da, _ = load_vec(la, torch.ones(8), k); db, _ = load_vec(lb, torch.ones(8), k)
        ro_adv = float((da - db).norm())
        # adversarial: w1 = 1 (full), w2 -> other example fully masked? (T_x=0 -> d=0) gives sqrt(kL(1-k/E))
        ro_adv2 = float(da.norm())
        b_2kL = math.sqrt(2 * k * L)
        b_2kL_c = math.sqrt(2 * k * L * (1 - k / E))
        b_exact = math.sqrt(2 * min(k, E - k) * L)
        results.append((E, k, L, bound, maxn, n_adv, n_one, ex_max, ro_max, ro_adv, ro_adv2, b_2kL, b_2kL_c, b_exact))

print("E   k  L  sqrt(kL(1-k/E))  max_rand   adv_same_k   T=1     exhaustive(E=8)  | replace-one: max_rand  adv_disjoint  adv_vs_empty  sqrt(2kL)  sqrt(2kL(1-k/E))  sqrt(2min(k,E-k)L)")
for r in results:
    E, k, L, b, mr, na, n1, ex, rom, roa, roa2, b1, b2, b3 = r
    print(f"{E:<3} {k:<2} {L:<2} {b:<16.6f} {mr:<10.6f} {na:<12.6f} {n1:<7.4f} {ex:<16.6f} | {rom:<21.6f} {roa:<13.6f} {roa2:<13.6f} {b1:<10.4f} {b2:<17.4f} {b3:.4f}")
