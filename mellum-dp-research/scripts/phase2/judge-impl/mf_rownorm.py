import torch, math, time
from opaque.dpftrl.noise import band_mf_strategy
torch.set_num_threads(4)
n = 1024
def toeplitz(col):
    n = len(col); M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n): M[i:, i] = col[: n - i]
    return M
for mom in (1.0, 0.95):
    t0 = time.time()
    st = band_mf_strategy(bands=64, momentum=mom)
    c = st.coefficients(n_steps=n).double(); c = torch.cat([c, torch.zeros(n - len(c), dtype=torch.float64)])
    Cinv = torch.linalg.inv(toeplitz(c))
    rn = lambda M, t: float(M[t].norm())
    out = {"row_norm_t=n-1": rn(Cinv, n-1), "row_norm_t=0": rn(Cinv, 0)}
    for bf in (0.95, 0.99):
        F = (1-bf)*toeplitz(torch.tensor([bf**i for i in range(n)], dtype=torch.float64)) @ Cinv
        out[f"EMA{bf}"] = rn(F, n-1)
    col = torch.zeros(n, dtype=torch.float64); col[:256] = 1/256
    out["W256"] = rn(toeplitz(col) @ Cinv, n-1)
    print(f"momentum={mom} n={n} sens={st.sensitivity(n_steps=n):.4f} ({time.time()-t0:.1f}s):", {k: round(v,4) for k,v in out.items()})
