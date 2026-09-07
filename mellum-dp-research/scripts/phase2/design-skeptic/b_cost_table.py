"""B: skeptic cost table at the preset regime (nm=0.5622, B=256, k=8, E=64, q=256/5e5, T=15625, delta=1e-6).

Joint PerGroup release (accountant unchanged): probe leaf carries s*h_x, bound C_h = s*sqrt(k) = rho*C.
Optimal allocation: sigma_g = nm*C*sqrt(1+rho); sigma_h = nm*C_h*sqrt(1+1/rho).
Released f_hat = leaf/(s*Bbar): per-entry noise std = nm*sqrt(k)*sqrt(1+1/rho)/Bbar; r = that/(k/E).
'eps if nm held' = the equivalent separate same-batch release gaussian(nm*sqrt(1+1/rho)) composed inside Poisson.
Also: the monitor-only regime rho in {0.01, 0.02} and the decision-rule detection power.
"""
import math, time
import torch
torch.set_num_threads(2)
from opaque.dpsgd.accounting import gaussian, poisson
from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.types import PerGroup

NM, Q, T, DELTA = 0.5622, 256 / 500_000, 15625, 1e-6
E, K, BBAR, C = 64, 8, 256, 0.9
t0 = time.time()
eps0 = (poisson(gaussian(NM), Q) * T).epsilon_at(DELTA)
print(f"baseline eps = {eps0:.4f} [{time.time()-t0:.1f}s]")
base_r = NM * E / (BBAR * math.sqrt(K))
print(f"nm*E/(B*sqrt(k)) = {base_r:.4f}")
ema99 = math.sqrt(0.01 / 1.99); ema95 = math.sqrt(0.05 / 1.95)
print(f"{'rho':>5} {'grad x':>7} {'r1':>7} {'r W=256':>8} {'r ema.99':>9} {'r ema.95':>9} {'sig_h':>7} {'eps hold nm':>11} {'maha':>6}")
for rho in (0.01, 0.02, 0.05, 0.1, 0.25):
    lam = rho * C
    mn = PerGroup({("fallback",): "fallback", ("probe",): "probe"}, {"fallback": C / BBAR, "probe": lam / BBAR})
    sd = per_group_noise_stddev(mn, NM)
    sg, sh = sd.values["fallback"], sd.values["probe"]
    maha = ((C / BBAR) ** 2 / sg ** 2 + (lam / BBAR) ** 2 / sh ** 2) * NM ** 2
    infl = sg / (NM * C / BBAR)
    s = lam / math.sqrt(K)
    r1 = (sh / s) / (K / E)
    sig_h = NM * math.sqrt(1 + 1 / rho)
    t1 = time.time()
    eps_hold = (poisson(gaussian(NM) | gaussian(sig_h), Q) * T).epsilon_at(DELTA)
    print(f"{rho:5.2f} {infl:7.3f} {100*r1:6.1f}% {100*r1/16:7.1f}% {100*r1*ema99:8.1f}% {100*r1*ema95:8.1f}% {sig_h:7.3f} {eps_hold:11.3f} {maha:6.3f} [{time.time()-t1:.1f}s]")

# decision rule: declare "imbalanced" when max_e |f~_e - k/E|/(k/E) > tau with f~ a W-step window mean.
# noise per entry after W-step mean = r1/sqrt(W) (DP-SGD) in k/E units; false-alarm prob per check with 64 entries.
from statistics import NormalDist
nd = NormalDist()
for rho in (0.01, 0.02, 0.05):
    r1 = base_r * math.sqrt(1 + 1 / rho)
    for W in (64, 256):
        sig = r1 / math.sqrt(W)
        for tau in (0.25, 0.5):
            pfa = 1 - (2 * nd.cdf(tau / sig) - 1) ** E  # any of 64 entries exceeding tau under exact balance
            pmiss = nd.cdf((tau - 2 * tau) / sig)         # miss when true imbalance is 2*tau on one expert
            print(f"rho={rho:.2f} W={W:>3} tau={tau}: noise std {100*sig:.1f}% of k/E; P(false alarm)={pfa:.2e}; P(miss | true dev 2tau)={pmiss:.2e}")

# band-MF bands=64 sanity (n=1024): row norms and the W=256 window-mean / EMA(.99) factors vs iid
from opaque.dpftrl.noise import band_mf_strategy
n = 1024
t1 = time.time()
st = band_mf_strategy(bands=64)
c = st.coefficients(n_steps=n).double()
c = torch.cat([c, torch.zeros(n - len(c), dtype=torch.float64)])
Cm = torch.zeros(n, n, dtype=torch.float64)
for i in range(n):
    Cm[i:, i] = c[: n - i]
Cinv = torch.linalg.inv(Cm)
def toep(col):
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i:, i] = col[: n - i]
    return M
print(f"band-MF n={n}: sens={st.sensitivity(n_steps=n):.4f} ||row_t(C^-1)|| t=0,63,n-1: "
      f"{[round(float(Cinv[t].norm()),4) for t in (0,63,n-1)]} [{time.time()-t1:.1f}s]")
for W in (64, 256):
    col = torch.zeros(n, dtype=torch.float64); col[:W] = 1.0 / W
    Fm = toep(col) @ Cinv
    print(f"  window mean W={W}: MF factor {float(Fm[n-1].norm()):.4f} vs iid {1/math.sqrt(W):.4f}")
for b in (0.95, 0.99):
    Fm = (1 - b) * toep(torch.tensor([b ** i for i in range(n)], dtype=torch.float64)) @ Cinv
    print(f"  EMA beta={b}: MF factor {float(Fm[n-1].norm()):.4f} vs iid {math.sqrt((1-b)/(1+b)):.4f}")
print(f"total {time.time()-t0:.1f}s")
