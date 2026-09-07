"""E1: epsilon cost of an extra per-step Gaussian release under Poisson DP-SGD.

q=256/500000, T=15625 (8 epochs), sigma=1.0, delta=1e-6.
Extra release: sensitivity s, noise sigma_h*s  => noise multiplier sigma_h.
"""
import math, time, json
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

q = 256 / 500_000
T = 15_625
delta = 1e-6
nm = 1.0

t0 = time.time()
base = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q) * T
eps_base = base.epsilon_at(delta)
print(f"baseline  poisson(gaussian({nm}), q={q:.3e}) * {T}: eps={eps_base:.4f}  [{time.time()-t0:.1f}s]")

rows = []
for sh in (1.0, 2.0, 4.0, 8.0):
    # (b) heterogeneous composition INSIDE the Poisson step (both releases on the same sampled batch)
    inner = dpsgd_acc.gaussian(nm) | dpsgd_acc.gaussian(sh)
    het = dpsgd_acc.poisson(inner, sample_rate=q) * T
    t1 = time.time(); eps_het = het.epsilon_at(delta); t_het = time.time() - t1
    # (c) analytic collapse: joint Gaussian with (1/nm_eff)^2 = (1/nm)^2 + (1/sh)^2
    nm_eff = 1.0 / math.sqrt(1.0 / nm**2 + 1.0 / sh**2)
    col = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm_eff), sample_rate=q) * T
    eps_col = col.epsilon_at(delta)
    # (d) composing OUTSIDE the Poisson (two independently-subsampled releases per step) — looser, but also valid
    outside = (dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q)
               | dpsgd_acc.poisson(dpsgd_acc.gaussian(sh), sample_rate=q)) * T
    eps_out = outside.epsilon_at(delta)
    # (e) eps-matched: what gradient noise multiplier nm' restores eps_base when the extra release (sh) is present?
    try:
        res = acc.calibrate(
            acc.epsilon_budget(eps_base, delta),
            lambda x, _sh=sh: dpsgd_acc.poisson(dpsgd_acc.gaussian(x) | dpsgd_acc.gaussian(_sh), sample_rate=q) * T,
            0.5, 64.0, tolerance=1e-4,
        )
        nm_matched = res.param
    except Exception as e:
        nm_matched = float("nan"); print("   eps-match unreachable:", str(e)[:120])
    rows.append(dict(sigma_h=sh, eps_het=eps_het, eps_collapse=eps_col, nm_eff=nm_eff,
                     eps_outside=eps_out, nm_matched=float(nm_matched), t_het=t_het))
    print(f"sigma_h={sh:>4}: eps(inside-Poisson het)={eps_het:.4f}  eps(gaussian(nm_eff={nm_eff:.4f}))={eps_col:.4f}"
          f"  eps(outside-Poisson)={eps_out:.4f}  eps-matched grad nm'={float(nm_matched):.4f}  [{t_het:.1f}s]")

# ---- preset regime: eps=3 target (examples/train_dpftrl.py mellum2-kstack uses eps=3, 8 epochs, batch 256)
res3 = acc.calibrate(acc.epsilon_budget(3.0, delta), lambda x: dpsgd_acc.poisson(dpsgd_acc.gaussian(x), sample_rate=q) * T, 0.3, 4.0, tolerance=1e-4)
nm3 = float(res3.param)
print(f"\neps=3 regime: baseline nm={nm3:.4f} (eps={ (dpsgd_acc.poisson(dpsgd_acc.gaussian(nm3), sample_rate=q)*T).epsilon_at(delta):.4f})")
rows3 = []
for mult in (1.0, 2.0, 4.0, 8.0):
    sh = mult * nm3     # extra release noise multiplier expressed relative to the gradient's nm
    eps_h = (dpsgd_acc.poisson(dpsgd_acc.gaussian(nm3) | dpsgd_acc.gaussian(sh), sample_rate=q) * T).epsilon_at(delta)
    try:
        r = acc.calibrate(acc.epsilon_budget(3.0, delta),
                          lambda x, _sh=sh: dpsgd_acc.poisson(dpsgd_acc.gaussian(x) | dpsgd_acc.gaussian(_sh), sample_rate=q) * T,
                          0.3, 64.0, tolerance=1e-4)
        nmm = float(r.param)
    except Exception as e:
        nmm = float("nan")
    nm_eff = 1/math.sqrt(1/nm3**2 + 1/sh**2)
    rows3.append(dict(mult=mult, sigma_h=sh, eps_with_extra=eps_h, nm_matched=nmm, grad_noise_increase=nmm/nm3 if nmm==nmm else None))
    print(f"  sigma_h={mult}x nm ({sh:.4f}): eps rises 3.0 -> {eps_h:.4f}; eps-matched grad nm'={nmm:.4f} (x{nmm/nm3 if nmm==nmm else float('nan'):.4f}); analytic nm_eff={nm_eff:.4f}")
json.dump(dict(nm3=nm3, rows3=rows3), open(__file__.replace(".py", "_eps3.json"), "w"), indent=1)
print("\n(f) joint per-group release (accounting stays gaussian(nm)); bounds C (grad), lambda (load):")
for ratio in (0.1, 0.25, 0.5, 1.0):
    C, lam = 1.0, ratio
    S = C + lam
    sig_g = nm * math.sqrt(C * S); sig_l = nm * math.sqrt(lam * S)
    iso = nm * math.sqrt(C**2 + lam**2)
    print(f"  lambda/C={ratio:<5} sigma_grad/(nm*C)={sig_g/(nm*C):.4f}  sigma_load/(nm*lambda)={sig_l/(nm*lam):.4f}"
          f"  isotropic sigma/(nm*C)={iso/(nm*C):.4f}  Mahalanobis check={(C/sig_g)**2+(lam/sig_l)**2:.6f} (=1/nm^2={1/nm**2})")
json.dump(dict(eps_base=eps_base, rows=rows), open(__file__.replace(".py", ".json"), "w"), indent=1)
print(f"total {time.time()-t0:.1f}s")
print("\n(g) context: eps of poisson(gaussian(sigma), q)*T vs sigma (why the sigma/sqrt2 collapse is expensive at small sigma):")
for s_ in (0.5, 0.6, 0.7071, 0.8, 1.0, 1.5):
    print(f"  sigma={s_}: eps={(dpsgd_acc.poisson(dpsgd_acc.gaussian(s_), sample_rate=q)*T).epsilon_at(delta):.4f}")
print("(h) independent second Poisson draw for the load statistic (2 subsampled releases/step):")
for sh in (1.0, 2.0, 4.0):
    e2 = ((dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q) | dpsgd_acc.poisson(dpsgan := dpsgd_acc.gaussian(sh), sample_rate=q)) * T).epsilon_at(delta)
    print(f"  sigma_h={sh}: eps={e2:.4f} (baseline {eps_base:.4f})")
