"""Whole-run consequences of the shared coin (worst-case dataset: y a duplicate of x, u=1).
(1) rigorous LOWER bound: eps_T(delta) >= eps_1(delta) (post-processing to the first step); solve eps_1(1e-6).
(2) RDP upper bound over T steps for both pairs (independent vs shared coin) for scale."""
import math
import numpy as np
from scipy.stats import norm
SIGMA, Q, T, DELTA = 0.5622, 256/5e5, 15625, 1e-6
x = np.linspace(-14*SIGMA-1, 14*SIGMA+3, 4_000_001); w = x[1]-x[0]
A = norm.pdf(x, 0, SIGMA)
pairs = {
  "independent": (A, (1-Q)*A + Q*norm.pdf(x, 1, SIGMA)),
  "shared u=1":  ((1-Q)*A + Q*norm.pdf(x, 1, SIGMA), (1-Q)*A + Q*norm.pdf(x, 2, SIGMA)),
}
def hs(p, q_, eps):
    e = math.exp(eps); return max(np.maximum(q_-e*p,0).sum()*w, np.maximum(p-e*q_,0).sum()*w)
def eps_at(p, q_, delta):
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = (lo+hi)/2
        if hs(p, q_, mid) > delta: lo = mid
        else: hi = mid
    return hi
def renyi(p, q_, alpha):
    """D_alpha(Q||P) and D_alpha(P||Q) (max)."""
    m = (p > 1e-300) & (q_ > 1e-300)
    d1 = math.log((q_[m]**alpha * p[m]**(1-alpha)).sum()*w)/(alpha-1)
    d2 = math.log((p[m]**alpha * q_[m]**(1-alpha)).sum()*w)/(alpha-1)
    return max(d1, d2)
for name, (p, q_) in pairs.items():
    e1 = eps_at(p, q_, DELTA)
    best = min((T*renyi(p, q_, a) + math.log(1/DELTA)/(a-1), a) for a in [1.5,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256])
    print(f"{name:>12}: single-step eps(delta=1e-6) = {e1:.3f}  -> whole-run eps(1e-6) >= {e1:.3f} (rigorous lower bound); "
          f"RDP upper bound over T={T}: eps <= {best[0]:.2f} (alpha={best[1]})")
    print(f"{'':>12}  single-step delta(eps=3) = {hs(p,q_,3.0):.3e}  -> whole-run delta(eps=3) >= that")
