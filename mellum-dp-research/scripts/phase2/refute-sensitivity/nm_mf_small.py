"""Bounded probe: at a SMALL horizon, compare the nm needed for eps=3 under Poisson-DP-SGD vs b-min-sep band-MF(64,.95).
Only the RATIO is of interest (the design's nm_MF caveat). Wall-clock bounded by the caller."""
import time, math
t0=time.time()
from opaque.dpsgd import accounting as A
from opaque.dpftrl import accounting as F
from opaque.dpftrl.noise import band_mf_strategy
q=256/5e5; d=1e-6
for n in (256, 1024):
    s=band_mf_strategy(bands=64, momentum=0.95)
    for nm in (0.5622, 1.0, 2.0):
        try:
            e_sgd=(A.poisson(A.gaussian(nm),q)*n).epsilon_at(d)
            e_mf=F.b_min_sep(F.mf_gaussian(nm, s), n_steps=n, p0=q).epsilon_at(d)
            print(f"n={n} nm={nm}: eps_sgd={e_sgd:.4f} eps_mf_bminsep={e_mf:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            print(f"n={n} nm={nm}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
