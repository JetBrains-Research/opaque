"""Cost table for the load release at the preset regime (design-optimal).
q=256/5e5, T=15625, delta=1e-6, nm=0.5622 (eps=3.0 baseline).
Routes: (A) same-batch joint/PerGroup at ratio rho (eps if nm held; grad-noise x sqrt(1+rho) if eps held)
        (B) independent Poisson draw, forward-only, sigma_h = c*nm, every m steps, batch multiplier bm
        (C) amortised same-batch every m steps (DP-SGD heterogeneous composition)
"""
import math, json, time
import opaque.accounting as acc
import opaque.dpsgd.accounting as d

q = 256/500_000; T = 15_625; delta = 1e-6; nm = 0.5622
E, k, B, Dh = 64, 8, 256, math.sqrt(8*(1-8/64))
base = d.poisson(d.gaussian(nm), sample_rate=q) * T
eps0 = base.epsilon_at(delta)
print(f"baseline eps={eps0:.4f}")
r_unit = nm*Dh/(B*k/E)   # single-release relative noise per entry at multiplier nm, batch 256
print(f"r_unit (sigma_h=nm, B=256, centred) = {r_unit:.4f}")
out = {"eps0": eps0, "A": [], "B": [], "C": []}
t0 = time.time()
print("\n(A) same-batch joint release, ratio rho = C_h/C_g")
for rho in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
    c = math.sqrt(1 + 1/rho)
    eps = (d.poisson(d.gaussian(nm) | d.gaussian(c*nm), sample_rate=q) * T).epsilon_at(delta)
    r1 = r_unit * c
    out["A"].append(dict(rho=rho, c=c, eps_nm_held=eps, grad_infl=math.sqrt(1+rho), r1=r1))
    print(f"  rho={rho:<5} c={c:6.3f}  eps(nm held)={eps:.3f}  grad x{math.sqrt(1+rho):.3f} (eps held)  r1={100*r1:5.1f}%  r EMA.95={100*r1*0.1601:4.1f}%")
print(f"[{time.time()-t0:.0f}s]")
print("\n(B) independent Poisson draw (forward-only), sigma_h=c*nm, batch multiplier bm, every m steps")
for c, bm, m in ((1,1,1),(2,1,1),(1,1,4),(2,1,4),(1,4,4),(1,1,16),(2,1,16),(1,4,16),(0.5,4,16)):
    q2 = bm*q; n2 = T//m
    proc = base | (d.poisson(d.gaussian(c*nm), sample_rate=q2) * n2)
    eps = proc.epsilon_at(delta)
    try:
        res = acc.calibrate(acc.epsilon_budget(eps0, delta),
            lambda x, _c=c, _q2=q2, _n2=n2: (d.poisson(d.gaussian(x), sample_rate=q)*T) | (d.poisson(d.gaussian(_c*nm), sample_rate=_q2)*_n2),
            0.4, 4.0, tolerance=1e-4)
        infl = float(res.param)/nm
    except Exception as e:
        infl = float('nan')
    r1 = r_unit*c/bm
    out["B"].append(dict(c=c, bm=bm, m=m, eps_nm_held=eps, grad_infl_eps_matched=infl, r1=r1))
    print(f"  c={c:<4} bm={bm} m={m:<3} eps(nm held)={eps:.3f}  eps-matched grad x{infl:.4f}  r1={100*r1:5.1f}%  r EMA.9-over-releases={100*r1*math.sqrt(0.1/1.9):4.1f}%  fwd overhead ~{100*bm/m/3:.0f}% of step")
print(f"[{time.time()-t0:.0f}s]")
print("\n(C) amortised same-batch (DP-SGD): joint release only every m steps, rho on those steps")
for rho, m in ((0.05,4),(0.2,4),(0.2,16),(0.5,16)):
    c = math.sqrt(1+1/rho); n2 = T//m
    proc = (d.poisson(d.gaussian(nm), sample_rate=q)*(T-n2)) | (d.poisson(d.gaussian(nm)|d.gaussian(c*nm), sample_rate=q)*n2)
    eps = proc.epsilon_at(delta)
    r1 = r_unit*c
    out["C"].append(dict(rho=rho, m=m, eps_nm_held=eps, r1=r1))
    print(f"  rho={rho} m={m:<3} eps(nm held)={eps:.3f}  (grad noise x{math.sqrt(1+rho):.3f} on 1/{m} of steps if eps held)  r1={100*r1:5.1f}%")
print(f"[{time.time()-t0:.0f}s]")
print("\n(D) randomised response on a 64-bit sign vector (LFB): per-step eps for flip prob p, worst case all 64 bits change")
for p in (0.25, 0.4, 0.45):
    print(f"  p={p}: eps_step = 64*ln((1-p)/p) = {64*math.log((1-p)/p):.1f}  (vs Gaussian route total extra eps < 0.5 over 15625 steps)")
json.dump(out, open(__file__.replace('.py','.json'),'w'), indent=1)
