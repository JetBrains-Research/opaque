"""judge-utility: independent recomputation of cost-table numbers (CPU, tiny)."""
import math, time, sys
import torch

from opaque.types import PerGroup
from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.dpsgd import accounting as acc

nm, Bbar, k, E, C = 0.5622, 256, 8, 64, 0.9
q, T, delta = 256 / 5e5, 15625, 1e-6
Dh_c = math.sqrt(k * (1 - k / E))   # centred structural bound
Dh_u = math.sqrt(k)                 # uncentred

print(f"Dh centred={Dh_c:.4f} uncentred={Dh_u:.4f}")
t0 = time.time()
eps0=3.0
print(f"baseline eps = {eps0:.4f}  [{time.time()-t0:.1f}s]")

print("\n(1) per-group allocation via the REAL function, centred release")
for rho in ():
    Ch = rho * C
    lam = Ch / Dh_c
    pg = PerGroup({"g": "fallback", "z": "probe"}, {"fallback": C, "probe": Ch})
    sig = per_group_noise_stddev(pg, nm)
    sg, sh = sig.values["fallback"], sig.values["probe"]
    mahal = (C / sg) ** 2 + (Ch / sh) ** 2
    infl = sg / (nm * C)
    r1 = (sh / (lam * Bbar)) / (k / E)          # per-entry noise on d-hat in k/E units
    r1_u = (nm * Dh_u * math.sqrt(1 + 1 / rho) / Bbar) / (k / E)  # uncentred variant
    c = math.sqrt(1 + 1 / rho)
    t0 = time.time()
    eps_hold = (acc.poisson(acc.gaussian(nm) | acc.gaussian(c * nm), q) * T).epsilon_at(delta)
    print(f" rho={rho:<5} grad x{infl:.4f} (sqrt(1+rho)={math.sqrt(1+rho):.4f})  mahal*nm^2={mahal*nm**2:.6f}"
          f"  r1 centred={100*r1:.1f}%  r1 uncentred={100*r1_u:.1f}%  eps if nm held={eps_hold:.3f} [{time.time()-t0:.1f}s]")

print("\n(2) DP-SGD iid filter factors")
for b in (0.95, 0.99):
    print(f" EMA {b}: {math.sqrt((1-b)/(1+b)):.4f}")
for W in (64, 256):
    print(f" window {W}: {1/math.sqrt(W):.4f}")

print("\n(3) band-MF bands=64 filter factors: preset momentum 0.95 vs library default 1.0")
from opaque.dpftrl.noise import band_mf_strategy
n = 1024
for mom in (0.95, 1.0):
    t0 = time.time()
    s = band_mf_strategy(bands=64, momentum=mom)
    coef = s.coefficients(n_steps=n).double()
    # build lower-triangular Toeplitz C and invert
    Cm = torch.zeros(n, n, dtype=torch.float64)
    nb = coef.numel()
    for i in range(n):
        m = min(nb, n - i)
        Cm[i:i + m, i] = coef[:m]
    Cinv = torch.linalg.solve_triangular(Cm, torch.eye(n, dtype=torch.float64), upper=False)
    row = Cinv.norm(dim=1)
    def filt_factor(w):  # w[j] = weight on x_{t-j}; F = lower-Toeplitz(w); factor = ||row_t(F Cinv)|| at t=n-1
        Fm = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            Fm[i, : i + 1] = torch.flip(w[: i + 1], dims=[0])
        return (Fm @ Cinv).norm(dim=1)[-1].item()
    ema95 = filt_factor((1 - 0.95) * torch.tensor([0.95 ** j for j in range(n)], dtype=torch.float64))
    ema99 = filt_factor((1 - 0.99) * torch.tensor([0.99 ** j for j in range(n)], dtype=torch.float64))
    win256 = filt_factor(torch.tensor([1 / 256 if j < 256 else 0.0 for j in range(n)], dtype=torch.float64))
    win64 = filt_factor(torch.tensor([1 / 64 if j < 64 else 0.0 for j in range(n)], dtype=torch.float64))
    print(f" momentum={mom}: sens={coef.norm():.4f} c0={coef[0]:.4f}  ||row_t(C^-1)||: t=0 {row[0]:.3f}, t=7 {row[7]:.3f}, t=63 {row[63]:.3f}, t=n-1 {row[-1]:.3f}"
          f"  EMA.95={ema95:.4f} EMA.99={ema99:.4f} win64={win64:.4f} win256={win256:.4f} [{time.time()-t0:.1f}s]")

print("\n(4) optimal's shrinkage threshold in per-coordinate RMS delta units")
for name, s_kE in (("MF rho=.02", 0.0083), ("SGD rho=.02", 0.0235)):
    # shrink engages when ||d||^2 < (E-1) s^2 ; ||d|| = delta*(k/E)*sqrt(E)
    delta_thr = math.sqrt((E - 1) / E) * s_kE
    print(f" {name}: s={s_kE} k/E-units -> threshold on delta (per-coord RMS) = {delta_thr:.4f}; on ||d||/(k/E) = {math.sqrt(E-1)*s_kE:.4f}")

print("\n(5) misc arithmetic")
d = 28 * (16 * (2304 + 4096) * 2 + 16 * (2304 + 512) * 2)
print(f" LoRA params r=16 q/k/v/o: {d}  sqrt={math.sqrt(d):.1f}  noise norm nm*C*sqrt(d)/B = {nm*C*math.sqrt(d)/Bbar:.2f}")
mac_expert = 2304 * 2 * 896 + 896 * 2304
print(f" expert MAC/token = {mac_expert/1e6:.2f}M; routed x8 = {8*mac_expert/1e6:.1f}M; dense x64 = {64*mac_expert/1e6:.0f}M; router fp32 GEMM {64*2304/1e6:.3f}M MAC = {100*64*2304/(8*mac_expert):.2f}% of routed")
print(f" 64 ln(.55/.45) = {64*math.log(.55/.45):.1f};  64 ln 3 = {64*math.log(3):.1f}")
