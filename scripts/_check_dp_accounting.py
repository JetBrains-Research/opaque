"""Quick check of dp_accounting vs opaque.accounting precision."""
from dp_accounting.pld import privacy_loss_distribution as pld_lib
import opaque.accounting as acc

def ref_gaussian(sigma, sampling_prob=1.0):
    return pld_lib.from_gaussian_mechanism(sigma, sampling_prob=sampling_prob)

# --- Gaussian ---
for sigma in [0.1, 0.25, 0.3, 0.35, 0.5, 0.65, 0.8, 1.2]:
    ref = ref_gaussian(sigma)
    eps_ref = ref.get_epsilon_for_delta(1e-5)
    eps_ours = acc.gaussian(sigma).epsilon_at(1e-5)
    diff = abs(eps_ref - eps_ours)
    flag = " <<<" if diff > 1e-6 else ""
    print(f"Gaussian({sigma:4.2f}) eps@1e-5  ref={eps_ref:.10f}  ours={eps_ours:.10f}  diff={diff:.2e}{flag}")

print()

# --- Poisson with various params ---
for sigma in [0.5, 0.8, 1.2]:
    for q in [0.001, 0.0005, 0.0001]:
        for steps in [10, 50, 200, 500, 1000]:
            ref = ref_gaussian(sigma, sampling_prob=q).self_compose(steps)
            eps_ref = ref.get_epsilon_for_delta(1e-5)
            eps_ours = (acc.poisson(acc.gaussian(sigma), q) * steps).epsilon_at(1e-5)
            diff = abs(eps_ref - eps_ours)
            flag = " <<<" if diff > 1e-6 else ""
            print(f"Poisson(G({sigma}),{q})*{steps:4d}  ref={eps_ref:.8f}  ours={eps_ours:.8f}  diff={diff:.2e}{flag}")

print()

# --- delta_at roundtrip ---
proc = acc.gaussian(0.8)
eps = proc.epsilon_at(1e-5)
delta = proc.delta_at(eps)
print(f"Roundtrip: epsilon_at(1e-5)={eps:.10f}, delta_at(eps)={delta:.2e} (should be ~1e-5)")

# --- advantage = delta_at(0) ---
adv = proc.advantage()
d0 = proc.delta_at(0.0)
print(f"advantage={adv:.10f}, delta_at(0)={d0:.10f}, diff={abs(adv - d0):.2e}")
