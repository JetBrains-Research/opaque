import time, torch
torch.set_num_threads(2)
from opaque.dpftrl.noise import band_mf_strategy
from opaque.dpftrl.accounting import mf_gaussian, b_min_sep
st = band_mf_strategy(bands=64)
q = 256/500_000; T = 15625
for nm in (0.5622,):
    t0 = time.time()
    proc = b_min_sep(mf_gaussian(nm, st), n_steps=T, p0=q)
    eps = proc.epsilon_at(1e-6)
    print(f"band-MF b-min-sep nm={nm}: eps={eps:.4f} [{time.time()-t0:.1f}s]")
