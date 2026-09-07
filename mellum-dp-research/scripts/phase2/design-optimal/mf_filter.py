"""G7: noise factor of linear filters applied to the band-MF load stream (bands=64, momentum=0.95).
Released stream: xhat = d + sigma_h * (C^{-1} Z)_t. Filter F (lower-triangular Toeplitz). Noise std of (F xhat)_t
= sigma_h * ||row_t(F C^{-1})||. Compare EMA(beta_f) = (1-beta_f) Toeplitz(beta_f^i), and DP-SGD equivalents."""
import torch, math, time
from opaque.dpftrl.noise import band_mf_strategy
torch.set_num_threads(4)
n = 2048
t0 = time.time()
strat = band_mf_strategy(bands=64, momentum=0.95)
c = strat.coefficients(n_steps=n).double(); c = torch.cat([c, torch.zeros(n - len(c), dtype=torch.float64)])
print(f"coefficients computed in {time.time()-t0:.1f}s; sensitivity ||c||={strat.sensitivity(n_steps=n):.4f}; c[:4]={c[:4].tolist()}")
def toeplitz(col):
    n = len(col); M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i:, i] = col[: n - i]
    return M
C = toeplitz(c)
Cinv = torch.linalg.inv(C)
def rownorm(M, t): return float(M[t].norm())
print("||row_t(C^-1)||:", [round(rownorm(Cinv, t),4) for t in (0,1,7,31,255,n-1)])
def ema_filter(beta):
    return (1-beta)*toeplitz(torch.tensor([beta**i for i in range(n)], dtype=torch.float64))
for bf in (0.9, 0.95, 0.99, 0.995):
    F = ema_filter(bf) @ Cinv
    mf = rownorm(F, n-1)
    sgd = math.sqrt((1-bf)/(1+bf))
    print(f"EMA beta_f={bf}: MF stationary factor={mf:.4f} (t=255: {rownorm(F,255):.4f})  DP-SGD factor={sgd:.4f}  ratio MF/SGD={mf/sgd:.3f}  lag={1/(1-bf):.0f} steps")
for W in (16, 64, 256):
    col = torch.zeros(n, dtype=torch.float64); col[:W] = 1.0/W
    F = toeplitz(col) @ Cinv
    print(f"running mean W={W}: MF factor={rownorm(F,n-1):.4f}  DP-SGD factor={1/math.sqrt(W):.4f}")
# noise variance of the sum-zero projection is (1-1/E) of the raw
print("sum-zero projection variance factor:", 1-1/64)
print(f"total {time.time()-t0:.1f}s")
