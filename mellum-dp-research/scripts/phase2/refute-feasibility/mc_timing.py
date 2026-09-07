import time, sys
from opaque.dpftrl.accounting import mf_gaussian, b_min_sep
from opaque.dpftrl.noise import band_mf_strategy
n = int(sys.argv[1]); res = float(sys.argv[2]); nm = float(sys.argv[3])
proc = b_min_sep(mf_gaussian(nm, band_mf_strategy(bands=64, momentum=0.95)), n_steps=n, p0=256/5e5)
t0 = time.time()
try:
    eps = proc.epsilon_at(1e-6, mc_resolution=res, mc_failure_probability=1e-6)
except TypeError as e:
    print("TypeError", e); eps = proc.epsilon_at(1e-6)
print(f"n={n} res={res} nm={nm} eps={eps:.4f} wall={time.time()-t0:.1f}s", flush=True)
