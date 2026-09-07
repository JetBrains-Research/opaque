"""Cost table for the design-minimal note (CPU, tiny).

(a) DP-SGD / Poisson at the preset regime: nm=0.5622, q=256/5e5, T=15625, delta=1e-6.
    Joint PerGroup route (accounting unchanged) vs "hold nm, pay epsilon" equivalent.
(b) band-MF bands=64: ||row_t(C^-1)|| and the noise factor of load-stream filters.
"""

from __future__ import annotations

import math
import time

import torch

torch.set_num_threads(2)

from opaque.dpsgd.accounting import gaussian, poisson  # noqa: E402
from opaque.api.engine.noise_allocation import per_group_noise_stddev  # noqa: E402
from opaque.types import PerGroup  # noqa: E402

NM, Q, T, DELTA = 0.5622, 256 / 500_000, 15625, 1e-6
E, K, BBAR, C = 64, 8, 256, 0.9

print("== (a) DP-SGD / Poisson ==")
t0 = time.time()
base = (poisson(gaussian(NM), Q) * T).epsilon_at(DELTA)
print(f"baseline eps(nm={NM}) = {base:.4f}   [{time.time()-t0:.1f}s]")

base_r = NM * E / (BBAR * math.sqrt(K))
print(f"single-release relative error at sigma_h=nm: {100*base_r:.2f}%")
ema = {b: math.sqrt((1 - b) / (1 + b)) for b in (0.9, 0.95, 0.99)}
print("EMA factors:", {b: round(v, 4) for b, v in ema.items()})

rows = []
for rho in (0.05, 0.1, 0.25, 0.5):
    lam = rho * C
    # normalized bounds as the trainer builds them (normalize_by = expected batch)
    mn = PerGroup({("fallback",): "fallback", ("router_load_probe",): "router_load_probe"},
                  {"fallback": C / BBAR, "router_load_probe": lam / BBAR})
    sd = per_group_noise_stddev(mn, NM)
    sg, sh = sd.values["fallback"], sd.values["router_load_probe"]
    # Mahalanobis check
    maha = (C / BBAR) ** 2 / sg**2 + (lam / BBAR) ** 2 / sh**2
    infl = sg / (NM * C / BBAR)
    # leaf carries s*f_x with s = lam/sqrt(K); estimate f_hat = leaf/s
    s = lam / math.sqrt(K)
    per_entry = sh / s
    r = per_entry / (K / E)
    # "hold nm on gradient, release load same-batch at sigma_h = nm*sqrt(1+1/rho)" equivalent
    sigma_h_equiv = NM * math.sqrt(1 + 1 / rho)
    t0 = time.time()
    eps_hold = (poisson(gaussian(NM) | gaussian(sigma_h_equiv), Q) * T).epsilon_at(DELTA)
    dt = time.time() - t0
    rows.append((rho, infl, r, r * ema[0.95], r * ema[0.99], sigma_h_equiv, eps_hold, maha, dt))

print(f"{'rho':>5} {'grad x':>7} {'r 1-step':>9} {'r ema.95':>9} {'r ema.99':>9} {'sig_h eq':>9} {'eps hold':>9} {'1/nm^2 chk':>11}")
for rho, infl, r, r95, r99, sh_eq, eps_hold, maha, dt in rows:
    print(f"{rho:5.2f} {infl:7.3f} {100*r:8.1f}% {100*r95:8.1f}% {100*r99:8.1f}% {sh_eq:9.3f} {eps_hold:9.3f} {maha*NM**2:11.4f}  [{dt:.1f}s]")

print("\n== (b) band-MF bands=64 ==")
from opaque.dpftrl.noise import band_mf_strategy  # noqa: E402

for n in (2048, 15625):
    t0 = time.time()
    try:
        st = band_mf_strategy(bands=64)
        c = st.coefficients(n_steps=n).to(torch.float64)
        sens = st.sensitivity(n_steps=n)
    except Exception as exc:  # noqa: BLE001
        print(f"n={n}: strategy failed: {exc!r}")
        continue
    dt = time.time() - t0
    b = len(c)
    # inverse Toeplitz coefficients d (lower-triangular Toeplitz)
    d = torch.zeros(n, dtype=torch.float64)
    d[0] = 1.0 / c[0]
    for i in range(1, n):
        acc = 0.0
        for j in range(1, min(i, b - 1) + 1):
            acc += c[j].item() * d[i - j].item()
        d[i] = -acc / c[0].item()
    rown = torch.sqrt(torch.cumsum(d**2, 0))
    print(f"n={n}: bands={b} sens={sens:.4f} c[0]={c[0]:.4f} c[1]={c[1]:.4f} c[-1]={c[-1]:.5f}  [{dt:.1f}s]")
    print("  ||row_t(C^-1)|| at t=0,1,63,64,127,512,n-1:",
          [round(rown[i].item(), 4) for i in (0, 1, 63, 64, 127, 512, n - 1)])
    # noise factor of linear filters applied to the noised leaf stream x_t = leaf_t + s*(C^-1 z)_t
    # filter with coefficients w (w[0] = weight on current step): noise std = s*||(w * d)[:t+1]||
    def filt_factor(w: torch.Tensor, t: int) -> float:
        L = t + 1
        wv = torch.zeros(L, dtype=torch.float64)
        m = min(L, len(w))
        wv[:m] = w[:m]
        e = torch.zeros(L, dtype=torch.float64)
        # convolution e_i = sum_j w_j d_{i-j}
        for i in range(L):
            jmax = min(i, m - 1)
            e[i] = torch.dot(wv[: jmax + 1], d[i - jmax : i + 1].flip(0))
        return math.sqrt(float(torch.dot(e, e)))

    ts = [63, 127, 511, n - 1]
    for name, w, iid in (
        ("EMA b=.95", torch.tensor([(1 - 0.95) * 0.95**i for i in range(n)]), ema[0.95]),
        ("EMA b=.99", torch.tensor([(1 - 0.99) * 0.99**i for i in range(n)]), ema[0.99]),
        ("win-mean 64", torch.full((64,), 1 / 64, dtype=torch.float64), 1 / 8),
        ("win-mean 256", torch.full((256,), 1 / 256, dtype=torch.float64), 1 / 16),
    ):
        vals = [round(filt_factor(w, t), 4) for t in ts]
        print(f"  {name:13s} noise factor at t={ts}: {vals}   (iid DP-SGD equivalent {iid:.4f})")
