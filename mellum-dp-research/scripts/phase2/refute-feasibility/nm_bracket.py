import time
from opaque.dpftrl.accounting import mf_gaussian
from opaque.dpftrl.noise import band_mf_strategy
from opaque.dpsgd.accounting import gaussian, poisson
t0=time.time()
strat = band_mf_strategy(bands=64, momentum=0.95)
def eps(nm, mp):
    return mf_gaussian(nm, strat, n_steps=15625, min_sep=64, max_participations=mp).epsilon_at(1e-6)
for mp in (8, 245):
    lo, hi = 0.3, 30.0
    for _ in range(30):
        mid = (lo*hi)**0.5
        if eps(mid, mp) > 3.0: lo = mid
        else: hi = mid
    print(f"un-amplified band_mf(64,0.95) n=15625 min_sep=64 max_part={mp}: nm at eps=3 ~ {hi:.4f} (eps={eps(hi,mp):.3f}); sens={strat.sensitivity(n_steps=15625, min_sep=64, max_participations=mp):.4f}", flush=True)
print("dpsgd baseline check:", poisson(gaussian(0.5622), 256/5e5).__mul__(15625).epsilon_at(1e-6))
print(f"wall {time.time()-t0:.1f}s")
