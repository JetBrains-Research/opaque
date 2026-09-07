import math
import opaque.dpsgd.accounting as A
NM, Q, T, DELTA = 0.5622, 256/5e5, 15625, 1e-6
for rho in (0.1, 0.02):
    nm_h = NM*math.sqrt(1+1/rho); nm_eff = NM*math.sqrt((1+rho)/(1+2*rho))
    print(f"\nrho={rho}: nm_h={nm_h:.4f} nm_eff={nm_eff:.5f}")
    comp = A.gaussian(NM) | A.gaussian(nm_h); joint = A.gaussian(nm_eff)
    print(" single-step UNsubsampled delta_at(eps):  composed(g|g_h)   joint g(nm_eff)   analytic Gaussian")
    for eps in (0.5, 1.0, 2.0, 3.0):
        from scipy.stats import norm
        dG = norm.cdf(1/(2*nm_eff) - eps*nm_eff) - math.exp(eps)*norm.cdf(-1/(2*nm_eff) - eps*nm_eff)
        print(f"   eps={eps}: {comp.delta_at(eps):.6e}   {joint.delta_at(eps):.6e}   {dG:.6e}")
    pc = A.poisson(comp, Q); pj = A.poisson(joint, Q)
    print(" single-step POISSON delta_at(eps):  poisson(composed)   poisson(joint)")
    for eps in (1.0, 2.0, 3.0):
        print(f"   eps={eps}: {pc.delta_at(eps):.6e}   {pj.delta_at(eps):.6e}")
    print(f" T-fold eps@1e-6: poisson(composed)*T = {(pc*T).epsilon_at(DELTA):.4f}   poisson(joint)*T = {(pj*T).epsilon_at(DELTA):.4f}")
    # sanity: separate coins (two independent Poisson draws) is a *different* mechanism
    sep = (A.poisson(A.gaussian(NM), Q) | A.poisson(A.gaussian(nm_h), Q)) * T
    print(f" separate-coin composition (poisson(g)|poisson(g_h))*T eps = {sep.epsilon_at(DELTA):.4f}")
