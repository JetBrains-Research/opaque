"""refute-composition check 2: does a perfectly correlated inclusion coin between two
records (what the trainer's own comment says happens across ranks after a DDP resume,
_dp_trainer.py ~3750-3757) leave the tight Poisson-subsampled-Gaussian PLD intact?

1-D projection, sigma = nm, sensitivity 1.  Standard pair (independent coins):
    P = N(0,s^2)             vs   Q = (1-q) N(0,s^2) + q N(1,s^2)
Correlated pair (record y's coin == record x's coin, y's contribution u aligned with x):
    A = (1-q) N(0,s^2) + q N(u,s^2)   vs   B = (1-q) N(0,s^2) + q N(u+1,s^2)
delta(eps) = hockey-stick divergence, computed by numerical integration on a grid.
"""
import numpy as np
from scipy.stats import norm

def hs(p, q_, eps, x):
    return np.trapz(np.maximum(q_ - np.exp(eps) * p, 0.0), x)

for s, q in [(1.0, 0.01), (0.5622, 256 / 5e5)]:
    x = np.linspace(-12, 14, 400001)
    P = norm.pdf(x, 0, s)
    Q = (1 - q) * norm.pdf(x, 0, s) + q * norm.pdf(x, 1, s)
    print(f"sigma={s}, q={q}")
    for eps in (0.5, 1.0, 2.0, 3.0):
        d_std = max(hs(P, Q, eps, x), hs(Q, P, eps, x))
        rows = []
        for u in (0.0, 0.5, 1.0):
            A = (1 - q) * norm.pdf(x, 0, s) + q * norm.pdf(x, u, s)
            Bm = (1 - q) * norm.pdf(x, 0, s) + q * norm.pdf(x, u + 1, s)
            d_corr = max(hs(A, Bm, eps, x), hs(Bm, A, eps, x))
            rows.append((u, d_corr))
        print(f"  eps={eps}: delta_standard={d_std:.3e}   " +
              "  ".join(f"delta_corr(u={u})={d:.3e}" for u, d in rows))
