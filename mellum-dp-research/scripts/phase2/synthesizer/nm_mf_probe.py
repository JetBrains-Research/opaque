"""Best-effort probe of the preset's DP-FTRL accountant: b_min_sep(mf_gaussian(nm, band_mf(64, 0.95)), n_steps=15625, p0=256/5e5).
Prints eps at delta=1e-6 for a few nm; each MC-PLD evaluation is slow (minimal reported >170 s)."""
import time, sys
from opaque.dpftrl.noise import band_mf_strategy
from opaque.dpftrl.accounting import mf_gaussian, b_min_sep
strat = band_mf_strategy(bands=64, momentum=0.95)
for nm in (0.5622, 1.5, 3.0):
    t0 = time.time()
    p = b_min_sep(mf_gaussian(nm, strat), n_steps=15625, p0=256/5e5)
    try:
        e = p.epsilon_at(1e-6)
        print(f"nm={nm}: eps={e:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    except Exception as ex:
        print(f"nm={nm}: FAILED {type(ex).__name__}: {ex}  [{time.time()-t0:.0f}s]", flush=True)
