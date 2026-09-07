"""Auditor: shared-coin hockey stick for partner mass u in {0,1,3,7} (W=2,4,8 ranks with all co-indexed records duplicates of x).
Pair: A=(1-q)N(0,s^2)+qN(u,s^2)  vs  B=(1-q)N(0,s^2)+qN(u+1,s^2); sensitivity 1; grid rectangle rule, float64."""
import numpy as np
from scipy.stats import norm
s, q = 0.5622, 256 / 5e5
x = np.linspace(-14 * s - 1, 14 * s + 12, 6_000_001); dx = x[1] - x[0]
def hs(p, q_, eps): return float(np.maximum(q_ - np.exp(eps) * p, 0.0).sum() * dx)
def delta(u, eps):
    A = (1 - q) * norm.pdf(x, 0, s) + q * norm.pdf(x, u, s)
    B = (1 - q) * norm.pdf(x, 0, s) + q * norm.pdf(x, u + 1, s)
    return max(hs(A, B, eps), hs(B, A, eps))
dG = lambda e: norm.cdf(1 / (2 * s) - e * s) - np.exp(e) * norm.cdf(-1 / (2 * s) - e * s)
print(f"cap q*delta_G(eps=3) = {q*dG(3):.4e}")
print(" u | delta(eps=3) | eps_1(delta=1e-6)")
for u in (0, 1, 3, 7):
    d3 = delta(u, 3.0)
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if delta(u, mid) > 1e-6: lo = mid
        else: hi = mid
    print(f" {u} | {d3:.4e} | {hi:.3f}")
