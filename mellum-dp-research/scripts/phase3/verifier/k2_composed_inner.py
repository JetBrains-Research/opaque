import math, time
import opaque.dpsgd.accounting as A
NM, Q, T, DELTA = 0.5622, 256/5e5, 15625, 1e-6
rho = 0.02
nm_h = NM*math.sqrt(1+1/rho); nm_eff = 1/math.sqrt(1/NM**2 + 1/nm_h**2)
print("nm_h", nm_h, "nm_eff", nm_eff)
# 1. unsubsampled: composed inner vs joint gaussian (should be identical: Gaussian PLRVs add)
e1 = ((A.gaussian(NM) | A.gaussian(nm_h)) * T).epsilon_at(DELTA)
e2 = (A.gaussian(nm_eff) * T).epsilon_at(DELTA)
print(f"unsubsampled: (g|g_h)*T eps={e1:.4f}   g(nm_eff)*T eps={e2:.4f}")
# 2. subsampled: composed inner vs separate coins vs joint
e3 = (A.poisson(A.gaussian(NM) | A.gaussian(nm_h), Q) * T).epsilon_at(DELTA)
e4 = ((A.poisson(A.gaussian(NM), Q) | A.poisson(A.gaussian(nm_h), Q)) * T).epsilon_at(DELTA)
e5 = (A.poisson(A.gaussian(nm_eff), Q) * T).epsilon_at(DELTA)
print(f"poisson(g|g_h,q)*T eps={e3:.4f}   (poisson(g,q)|poisson(g_h,q))*T eps={e4:.4f}   poisson(g(nm_eff),q)*T eps={e5:.4f}")
# 3. single-step delta at eps=3 from the accountant vs my hockey stick 3.1597e-11
step = A.poisson(A.gaussian(NM), Q)
d = step.delta_at(3.0)
print(f"accountant single-step delta_at(eps=3) = {d:.4e}  (mine: 3.1597e-11)")
print("types:", type(A.gaussian(NM) | A.gaussian(nm_h)).__name__, type(A.poisson(A.gaussian(NM) | A.gaussian(nm_h), Q)).__name__)
