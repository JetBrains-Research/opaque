"""Complete example demonstrating opaque.accounting API.

This example shows:
1. Basic mechanism creation and composition
2. Privacy budget queries (epsilon, delta, advantage, beta, risk)
3. Calibration for target privacy budgets
4. Multi-phase training with heterogeneous noise
5. Discretization configuration
6. All available mechanisms
"""

import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque.accounting.discretization import DiscretizationConfig

print("=" * 70)
print("OPAQUE ACCOUNTING: Complete API Demonstration")
print("=" * 70)

# =============================================================================
# 1. Basic DP-SGD: Poisson-subsampled Gaussian
# =============================================================================
print("\n1️⃣  BASIC DP-SGD")
print("-" * 70)

# Standard DP-SGD step
step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
print(f"Single step: {step}")

# Compose 1000 training steps
training = step * 1000
print(f"1000 steps:  {training}")

# Query privacy at delta=1e-5
epsilon = training.epsilon_at(1e-5)
print(f"Privacy: (ε={epsilon:.6f}, δ=1e-5)")

# =============================================================================
# 2. Production DP-SGD: Truncated Poisson (capped batch size)
# =============================================================================
print("\n2️⃣  PRODUCTION DP-SGD (Truncated Poisson)")
print("-" * 70)

# CIFAR-10 example: n=50k, batch=250, 10 epochs
n = 50_000
batch = 250
epochs = 10
steps = epochs * (n // batch)

step = acc.truncated_poisson(
    acc.gaussian(0.8),
    sample_rate=batch / n,
    batch_size_cap=batch,
    dataset_size=n,
)
training = step * steps
eps = training.epsilon_at(1e-5)
print(f"CIFAR-10 (n={n:,}, batch={batch}, {epochs} epochs):")
print(f"  Steps: {steps}")
print(f"  Privacy: (ε={eps:.6f}, δ=1e-5)")

# =============================================================================
# 3. All privacy metrics from a single PLD
# =============================================================================
print("\n3️⃣  PRIVACY METRICS")
print("-" * 70)

proc = acc.poisson(acc.gaussian(1.0), 0.01) * 500

print("Same process, different metrics:")
print(f"  (ε, δ)-DP:     ε={proc.epsilon_at(1e-5):.6f} at δ=1e-5")
print(f"  (ε, δ)-DP:     δ={proc.delta_at(3.0):.2e} at ε=3.0")
print(f"  f-DP:          advantage={proc.advantage():.6f}")
print(f"  (α, β) errors: β={proc.beta_at(0.05):.6f} at α=0.05")
print(f"  Bayes risk:    risk={proc.risk_at(0.5):.6f} at prior=0.5")

# =============================================================================
# 4. Calibration: Find noise for target privacy
# =============================================================================
print("\n4️⃣  CALIBRATION")
print("-" * 70)


def training_process(nm):
    """A 1000-step DP-SGD training run with given noise multiplier."""
    return acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000


# Find noise for (ε=3.0, δ=1e-5)
budget = cal.epsilon_budget(3.0, delta=1e-5)
result = cal.calibrate(budget, training_process, param_min=0.7, param_max=1.0, tolerance=0.01)

print(f"Target:   (ε=3.0, δ=1e-5)")
print(f"Solution: nm={result.param:.4f}")
print(f"Achieved: ε={result.achieved:.6f}")
print(f"Converged: {result.converged} ({result.iterations} iterations)")

# Calibrate for f-DP advantage
budget_adv = cal.advantage_budget(0.1)
result_adv = cal.calibrate(budget_adv, training_process, 0.7, 1.0, tolerance=0.001)
print(f"\nTarget:   f-DP advantage=0.1")
print(f"Solution: nm={result_adv.param:.4f}, achieved={result_adv.achieved:.6f}")

# =============================================================================
# 5. Multi-phase training (heterogeneous composition)
# =============================================================================
print("\n5️⃣  MULTI-PHASE TRAINING")
print("-" * 70)

# Three-phase curriculum: warm-up → main training → fine-tuning
warmup = acc.poisson(acc.gaussian(0.9), 0.01) * 200
main = acc.poisson(acc.gaussian(0.7), 0.01) * 600
finetune = acc.poisson(acc.gaussian(0.5), 0.01) * 200

# Compose with | operator
total = warmup | main | finetune
# Or: total = acc.compose(acc.compose(warmup, main), finetune)

eps_total = total.epsilon_at(1e-5)
print("Phase 1 (warm-up):   σ=0.9, 200 steps")
print("Phase 2 (main):      σ=0.7, 600 steps")
print("Phase 3 (fine-tune): σ=0.5, 200 steps")
print(f"Total privacy: (ε={eps_total:.6f}, δ=1e-5)")

# =============================================================================
# 6. All available mechanisms
# =============================================================================
print("\n6️⃣  ALL MECHANISMS")
print("-" * 70)

# Base Gaussian (no subsampling)
gauss = acc.gaussian(1.0)
print(f"Gaussian:           {gauss}")

# Poisson-subsampled Gaussian (standard DP-SGD)
poiss = acc.poisson(acc.gaussian(1.0), 0.01)
print(f"Poisson:            {poiss}")

# Truncated Poisson (production DP-SGD)
trunc = acc.truncated_poisson(acc.gaussian(1.0), 0.01, batch_size_cap=256, dataset_size=10000)
print(f"Truncated Poisson:  {trunc}")

# Parallel Poisson (multi-worker sampling)
parallel = acc.parallel_poisson(acc.poisson(acc.gaussian(1.0), 0.01), num_workers=4)
print(f"Parallel Poisson:   {parallel}")

# Adaptive clipping (Andrew et al. 2021)
adaclip = acc.adaclip(acc.gaussian(1.0), quantile_noise_std=50.0)
print(f"AdaClip:            {adaclip}")

# Fixed (ε, δ) guarantee (for composition with external mechanisms)
fixed = acc.eps_delta(epsilon=2.0, delta=1e-5)
print(f"Fixed (ε, δ):       {fixed}")

# Identity (zero privacy loss, useful for composition)
ident = acc.identity()
print(f"Identity:           {ident}")

# =============================================================================
# 7. Discretization configuration
# =============================================================================
print("\n7️⃣  DISCRETIZATION CONTROL")
print("-" * 70)

# Default precision (1e-4, high accuracy)
default = acc.poisson(acc.gaussian(1.0), 0.01)
eps_default = default.epsilon_at(1e-5)
print(f"Default (disc=1e-4): eps={eps_default:.6f}")

# Coarse precision (1e-3, faster)
coarse = acc.poisson(acc.gaussian(1.0, discretization=1e-3), 0.01)
eps_coarse = coarse.epsilon_at(1e-5)
print(f"Coarse  (disc=1e-3): eps={eps_coarse:.6f}")

# Fine precision (1e-5, maximum accuracy)
cfg_fine = DiscretizationConfig(discretization=1e-5, max_grid_size=1_000_000)
fine = acc.poisson(acc.gaussian(1.0, discretization=cfg_fine), 0.01)
eps_fine = fine.epsilon_at(1e-5)
print(f"Fine    (disc=1e-5): eps={eps_fine:.6f}")

# Module-level defaults
acc.set_discretization(discretization=1e-3)
module_default = acc.poisson(acc.gaussian(1.0), 0.01)
print(f"\nModule default set to 1e-3: eps={module_default.epsilon_at(1e-5):.6f}")

# =============================================================================
# 8. Quick inspection
# =============================================================================
print("\n8️⃣  QUICK INSPECTION")
print("-" * 70)

step = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
print(f"Training: {step}")
print(f"  epsilon(1e-5) = {step.epsilon_at(1e-5):.4f}")
print(f"  advantage     = {step.advantage():.4e}")

print("=" * 70)
print("✅ END OF DEMONSTRATION")
print("=" * 70)
