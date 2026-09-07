"""reviser checks: (1) dead-zone chi2 tails, (2) per-layer carrier pooled-noise identity,
(3) token-weighted surrogate identity with a public T-bar, (4) bias-corrected EMA gain."""
import numpy as np
from scipy import stats
E, k, L = 64, 8, 28
# (1) after sum-zero projection noise total ||n||^2 = sigma^2 * chi2_{E-1}; s^2 = (E-1)/E sigma^2
# dead zone ||d~||^2 < c * E * s^2  <=>  chi2_{E-1} < c*(E-1)
for c in (1.0, 1.5, 2.0):
    thr = c * (E - 1)
    p_pass = stats.chi2.sf(thr, df=E - 1)
    print(f"c={c}: P(pure-noise passes dead zone)={p_pass:.3e}; over 15625 steps expected passes={15625*p_pass:.3g}")
# (2) per-layer carrier: lambda_L = rho*C_g/sqrt(7L); pooled noise per entry = sigma_h/(lambda_L*sqrt(L)) vs sigma_h/lambda
rho, Cg, nm, Bbar = 0.02, 0.9, 0.5622, 256
Dh, DhL = np.sqrt(k * (1 - k / E)), np.sqrt(k * L * (1 - k / E))
lam, lamL = rho * Cg / Dh, rho * Cg / DhL
Ch = lam * Dh; S = Cg + Ch
sig_h = nm * np.sqrt(Ch * S) / Bbar
rng = np.random.default_rng(0)
n = 20000
noise_pooled = rng.normal(0, sig_h / lam, size=(n, E))
noise_layer = rng.normal(0, sig_h / lamL, size=(n, L, E)).mean(axis=1)
print(f"pooled-release per-entry noise std {noise_pooled.std():.5f}; per-layer carrier pooled std {noise_layer.std():.5f}; closed forms {sig_h/lam:.5f} / {sig_h/(lamL*np.sqrt(L)):.5f}; per-layer entry std {sig_h/lamL:.5f} (x{np.sqrt(L):.2f})")
# (3) token-weighted identity: HF aux grad direction = sum_x (T_x/T_tot) grad P(x); DP surrogate uses T_x/Tbar (public)
B = 6; T = rng.integers(200, 1025, size=B); Ttot = T.sum(); Tbar = 700.0
G = rng.normal(size=(B, E, 5))  # grad P_e(x) w.r.t. 5 fake params
f = rng.dirichlet(np.ones(E)) * k
hf = np.einsum('e,xep->p', f - k / E, np.einsum('x,xep->xep', T / Ttot, G))
dp = np.einsum('e,xep->p', f - k / E, np.einsum('x,xep->xep', T / (B * Tbar), G))
print(f"token-weighted: cos(HF, DP)={hf@dp/np.linalg.norm(hf)/np.linalg.norm(dp):.12f}; scale DP/HF={np.linalg.norm(dp)/np.linalg.norm(hf):.6f} vs T_tot/(B*Tbar)={Ttot/(B*Tbar):.6f}")
eq = np.einsum('e,xep->p', f - k / E, G) / B
print(f"equal-weight: cos(HF, equal)={hf@eq/np.linalg.norm(hf)/np.linalg.norm(eq):.4f} (not 1 under ragged lengths)")
# (4) EMA gain and corrected std
beta = 0.99
for t in (100, 200, 300, 460):
    gain = 1 - beta ** t
    phi2 = sum(((1 - beta) * beta ** i) ** 2 for i in range(t)) * (E - 1) / E
    print(f"t={t}: uncorrected gain {gain:.3f}; phi={np.sqrt(phi2):.4f}; corrected phi/(1-beta^t)={np.sqrt(phi2)/gain:.4f}")
