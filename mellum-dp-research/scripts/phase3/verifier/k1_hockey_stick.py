"""K1 privacy effect: hockey-stick divergence at eps=3 for the Poisson-subsampled Gaussian
(sigma=0.5622, q=256/5e5), (a) independent coins, (b) coin shared between the differing record x
and one other fixed record y (pair ((1-q)A + qB0, (1-q)A + qB1)).
Own numerical integration on a fine grid, float64, plus analytic cross-checks.
"""
import math
import numpy as np
from scipy.stats import norm

SIGMA = 0.5622
Q = 256 / 5e5
EPS = 3.0


def hs(p, q_, grid_w, eps):
    """hockey-stick H_{e^eps}(Q||P) = int max(Q - e^eps P, 0); and reverse."""
    e = math.exp(eps)
    d1 = np.maximum(q_ - e * p, 0.0).sum() * grid_w
    d2 = np.maximum(p - e * q_, 0.0).sum() * grid_w
    return d1, d2


def pdf(x, mu):
    return norm.pdf(x, loc=mu, scale=SIGMA)


# grid wide enough: contributions up to 2 (u + 1) plus 14 sigma tails
lo, hi, n = -14 * SIGMA - 1.0, 14 * SIGMA + 3.0, 4_000_001
x = np.linspace(lo, hi, n)
w = x[1] - x[0]

# (a) independent coins: P = A (x out), Q = (1-q)A + q N(1, sigma^2) (x in)
A = pdf(x, 0.0)
P_a = A
Q_a = (1 - Q) * A + Q * pdf(x, 1.0)
add_a, rem_a = hs(P_a, Q_a, w, EPS)
print(f"(a) independent coins, eps={EPS}: delta(add)={add_a:.4e}  delta(remove)={rem_a:.4e}  max={max(add_a, rem_a):.4e}")

# analytic cross-check for the unsubsampled Gaussian: delta_G(eps)=Phi(1/(2s)-eps s)-e^eps Phi(-1/(2s)-eps s)
dG = norm.cdf(1 / (2 * SIGMA) - EPS * SIGMA) - math.exp(EPS) * norm.cdf(-1 / (2 * SIGMA) - EPS * SIGMA)
print(f"    unsubsampled Gaussian delta_G(eps=3, sigma=0.5622) analytic = {dG:.4e};  q*delta_G = {Q*dG:.4e}")
# exact 1-D check of the grid: delta of the Gaussian pair by integration
dG_num = hs(pdf(x, 0.0), pdf(x, 1.0), w, EPS)
print(f"    grid delta_G = {dG_num[0]:.4e} (rel err {abs(dG_num[0]-dG)/dG:.2e})")

# (b) shared coin between x and y: M(D) = (1-q)A + q B0, M(D') = (1-q)A + q B1
#   A = N(0), B0 = N(u) (y in, x out), B1 = N(u+1) (y in, x in, x aligned with y)
for u in (0.0, 0.5, 1.0):
    B0 = pdf(x, u)
    B1 = pdf(x, u + 1.0)
    P_b = (1 - Q) * A + Q * B0
    Q_b = (1 - Q) * A + Q * B1
    add_b, rem_b = hs(P_b, Q_b, w, EPS)
    print(f"(b) shared coin, u={u}: delta(add)={add_b:.4e}  delta(remove)={rem_b:.4e}  max={max(add_b, rem_b):.4e}"
          f"   ratio to (a): {max(add_b, rem_b)/max(add_a, rem_a):.3e}")

# also the "y anti-aligned" and 2-D orthogonal cases for completeness
B0 = pdf(x, 1.0); B1 = pdf(x, 0.0)  # x's contribution cancels y's (anti-aligned): B1 = N(u-1) with u=1
add_b, rem_b = hs((1 - Q) * A + Q * B0, (1 - Q) * A + Q * B1, w, EPS)
print(f"(b') shared coin, u=1 anti-aligned x=-1: max delta={max(add_b, rem_b):.4e}")

# What eps does the shared-coin pair reach at the SAME delta as (a)?  (bisection on eps)
target = max(add_a, rem_a)
B0 = pdf(x, 1.0); B1 = pdf(x, 2.0)
P_b = (1 - Q) * A + Q * B0; Q_b = (1 - Q) * A + Q * B1
lo_e, hi_e = 0.0, 40.0
for _ in range(60):
    mid = 0.5 * (lo_e + hi_e)
    if max(hs(P_b, Q_b, w, mid)) > target:
        lo_e = mid
    else:
        hi_e = mid
print(f"eps at which shared-coin (u=1) pair reaches delta={target:.3e}: eps={hi_e:.3f}  (independent coins: eps=3)")

# Table over eps for (a) and (b,u=1)
print("\n eps | delta indep | delta shared(u=1) | ratio")
for e in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
    da = max(hs(P_a, Q_a, w, e)); db = max(hs(P_b, Q_b, w, e))
    print(f" {e:>3} | {da:.3e} | {db:.3e} | {db/da:.2e}")

# Whole-run intuition: per-step delta * T (simple union bound), T=15625
T = 15625
print(f"\nunion-bound over T={T} steps at eps=3: indep {T*max(add_a,rem_a):.2e}; shared(u=1) {T*max(hs(P_b,Q_b,w,3.0)):.2e}")
