# Privacy Auditing

This guide explains how to empirically validate differential privacy guarantees using membership inference attacks on canary examples.

## Overview

While privacy accounting provides theoretical guarantees, **privacy auditing** empirically validates them:

| Approach | What it tells you | Guarantees |
|----------|-------------------|------------|
| **Accounting** | Maximum possible privacy loss | Upper bound (theoretical) |
| **Auditing** | Actual observed privacy loss | Lower bound (empirical) |

A well-implemented DP system should have:
```
audited_epsilon <= theoretical_epsilon
```

If the audited epsilon exceeds the theoretical epsilon, there's likely a bug in your implementation.

## How Privacy Auditing Works

### The Canary Approach

1. **Create canary examples**: Special training examples designed to be memorable
2. **Train with/without canaries**: Run training multiple times including/excluding each canary
3. **Run membership inference**: Use an attack to distinguish trained-on vs not-trained-on canaries
4. **Estimate epsilon**: Convert attack success to a privacy bound

### Intuition

If a model is (ε, δ)-DP, then:
- An attacker cannot reliably tell if any single example was in the training set
- The attack success rate is bounded by the privacy parameters

By measuring actual attack success, we get an empirical lower bound on epsilon.

## Basic Usage

### Minimal Example

```python
import numpy as np
from opaque.auditing import audit

# Attack scores: higher = more likely to be a training member
in_scores = np.array([0.8, 0.9, 0.7, 0.85, 0.75])   # Held-in canaries
out_scores = np.array([0.3, 0.4, 0.2, 0.35, 0.25])  # Held-out canaries

# Run comprehensive audit
result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)

print(f"Epsilon lower bound: {result.epsilon:.2f}")
print(f"Attack AUROC: {result.auroc:.3f}")
print(f"TPR at 1% FPR: {result.tpr_at_low_fpr:.3f}")
print(f"Max accuracy: {result.max_accuracy:.3f}")
```

### Understanding the Scores

**Attack scores** come from a membership inference attack. Common sources:

| Score Type | Description | Higher means |
|------------|-------------|--------------|
| Loss | Training loss on the example | More likely OUT |
| Negative loss | `-loss` | More likely IN |
| Confidence | Model's confidence on correct label | More likely IN |
| LiRA score | Likelihood ratio statistic | More likely IN |

Opaque's auditing functions expect **higher scores = more likely IN** (training member).

## Epsilon Estimation Methods

Opaque provides three methods for estimating epsilon:

### 1. Clopper-Pearson (Recommended)

Conservative statistical bounds using binomial confidence intervals.

```python
from opaque.auditing import epsilon_clopper_pearson

eps = epsilon_clopper_pearson(
    in_scores, out_scores,
    significance=0.05,  # 95% confidence
    delta=1e-5,         # DP delta parameter
)
print(f"Epsilon >= {eps:.2f} (95% confidence)")
```

**When to use**: Default choice. Provides formal guarantees that hold with high probability.

**Trade-off**: Conservative (may underestimate true epsilon with small samples).

### 2. One-Run Method (Nasr et al. 2023)

Likelihood-ratio test that's less conservative with smaller samples.

```python
from opaque.auditing import epsilon_one_run

eps = epsilon_one_run(
    in_scores, out_scores,
    significance=0.05,
    delta=1e-5,
)
print(f"Epsilon >= {eps:.2f} (one-run method)")
```

**When to use**: When you have fewer canaries but want tighter bounds.

**Reference**: [Tight Auditing of Differentially Private Machine Learning](https://arxiv.org/abs/2305.08846)

### 3. Raw Counts

Direct computation without confidence intervals.

```python
from opaque.auditing import epsilon_raw_counts

eps = epsilon_raw_counts(
    in_scores, out_scores,
    min_count=50,  # Minimum FP count for stability
    delta=1e-5,
)
print(f"Epsilon estimate: {eps:.2f}")
```

**When to use**: Quick sanity checks. Not recommended for final results.

**Trade-off**: Higher variance, no formal guarantees.

## Attack Metrics

Beyond epsilon, these metrics help understand attack strength:

### AUROC (Area Under ROC Curve)

```python
from opaque.auditing import attack_auroc

auroc = attack_auroc(in_scores, out_scores)
print(f"Attack AUROC: {auroc:.3f}")
```

| AUROC | Interpretation |
|-------|----------------|
| 0.50 | Random guessing (no privacy leakage) |
| 0.60 | Weak attack |
| 0.80 | Strong attack |
| 1.00 | Perfect attack (complete privacy breach) |

### TPR at Low FPR

True positive rate at a given false positive rate. Important for real-world attacks where false accusations are costly.

```python
from opaque.auditing import tpr_at_fpr

# TPR at 1% FPR (common benchmark)
tpr_001 = tpr_at_fpr(in_scores, out_scores, fpr=0.01)
print(f"TPR at 1% FPR: {tpr_001:.3f}")

# Multiple FPRs
tprs = tpr_at_fpr(in_scores, out_scores, fpr=[0.001, 0.01, 0.1])
for fpr, tpr in zip([0.001, 0.01, 0.1], tprs):
    print(f"  FPR={fpr:.1%}: TPR={tpr:.3f}")
```

### Maximum Accuracy

Best-case classification accuracy achievable by the attack.

```python
from opaque.auditing import max_accuracy

acc = max_accuracy(in_scores, out_scores)
print(f"Max accuracy: {acc:.1%}")

# With imbalanced prevalence (e.g., 10% of population was in training)
acc_imbalanced = max_accuracy(in_scores, out_scores, prevalence=0.1)
```

## Confidence Intervals with Bootstrap

For uncertainty quantification, use bootstrap resampling:

```python
from opaque.auditing import bootstrap, attack_auroc, epsilon_clopper_pearson, BootstrapParams

# Configure bootstrap
params = BootstrapParams.confidence_interval(
    confidence=0.95,
    num_samples=2000,
    bias_correction=True,  # BCa bootstrap
    acceleration=True,
    seed=42,
)

# Bootstrap AUROC
auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)
print(f"AUROC 95% CI: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]")

# Bootstrap epsilon (wrap to pass fixed parameters)
def eps_fn(in_s, out_s):
    return epsilon_clopper_pearson(in_s, out_s, significance=0.05, delta=1e-5)

eps_ci = bootstrap(eps_fn, in_scores, out_scores, params)
print(f"Epsilon 95% CI: [{eps_ci[0]:.2f}, {eps_ci[1]:.2f}]")
```

## Complete Auditing Workflow

### Step 1: Design Canaries

Choose canary examples that are:
- **Distinct**: Different from typical training examples
- **Memorable**: Easy for the model to memorize if included
- **Representative**: Cover the data distribution

```python
# Example: Random canaries
def create_canaries(dataset, num_canaries=1000, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_canaries, replace=False)
    return indices
```

### Step 2: Train with Canary Splitting

```python
def train_and_score(model_fn, dataset, canary_indices, seed):
    """Train model and return membership scores for canaries."""
    rng = np.random.default_rng(seed)

    # Random split: half in, half out
    in_mask = rng.random(len(canary_indices)) < 0.5
    in_indices = canary_indices[in_mask]
    out_indices = canary_indices[~in_mask]

    # Train on full dataset MINUS out canaries
    train_indices = set(range(len(dataset))) - set(out_indices)
    model = model_fn(dataset, list(train_indices))

    # Score all canaries
    in_scores = compute_membership_scores(model, dataset, in_indices)
    out_scores = compute_membership_scores(model, dataset, out_indices)

    return in_scores, out_scores
```

### Step 3: Compute Membership Scores

Common scoring functions:

```python
def loss_based_scores(model, dataset, indices):
    """Higher loss = less likely to be training member."""
    model.eval()
    scores = []
    for idx in indices:
        x, y = dataset[idx]
        with torch.no_grad():
            loss = F.cross_entropy(model(x.unsqueeze(0)), y.unsqueeze(0))
        scores.append(-loss.item())  # Negate so higher = more likely IN
    return np.array(scores)

def confidence_based_scores(model, dataset, indices):
    """Higher confidence = more likely training member."""
    model.eval()
    scores = []
    for idx in indices:
        x, y = dataset[idx]
        with torch.no_grad():
            probs = F.softmax(model(x.unsqueeze(0)), dim=1)
            confidence = probs[0, y].item()
        scores.append(confidence)
    return np.array(scores)
```

### Step 4: Run Audit

```python
from opaque.auditing import audit, bootstrap, BootstrapParams

# Collect scores from training run
in_scores, out_scores = train_and_score(
    model_fn, dataset, canary_indices, seed=42
)

# Run comprehensive audit
result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)

print("=== Privacy Audit Results ===")
print(f"Epsilon lower bound: {result.epsilon:.2f}")
print(f"Attack AUROC: {result.auroc:.3f}")
print(f"TPR at 1% FPR: {result.tpr_at_low_fpr:.3f}")
print(f"Max accuracy: {result.max_accuracy:.1%}")

# Compare to theoretical epsilon
theoretical_eps = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"\nTheoretical epsilon: {theoretical_eps:.2f}")
print(f"Gap: {theoretical_eps - result.epsilon:.2f}")

if result.epsilon > theoretical_eps:
    print("WARNING: Audited epsilon exceeds theoretical bound!")
```

## Interpreting Results

### Healthy DP Implementation

```
Theoretical epsilon: 3.00
Audited epsilon:     1.50
Gap: 1.50
```

The audited epsilon is less than theoretical. The gap exists because:
- Statistical estimation is imperfect
- The attack may not be optimal
- The bound is conservative

### Potential Issues

```
Theoretical epsilon: 3.00
Audited epsilon:     5.00  # PROBLEM!
```

If audited > theoretical, investigate:
- Bug in gradient clipping
- Bug in noise injection
- Privacy accounting error
- Data leakage in preprocessing

## Best Practices

### 1. Use Enough Canaries

| Canaries | Reliability |
|----------|-------------|
| 100 | Low (high variance) |
| 1,000 | Moderate |
| 10,000+ | High |

More canaries = tighter confidence intervals.

### 2. Use Conservative Methods

Start with `epsilon_clopper_pearson()` for formal guarantees. Use `epsilon_one_run()` only when sample size is limited.

### 3. Report Confidence Intervals

```python
params = BootstrapParams.confidence_interval(confidence=0.95, num_samples=2000)
eps_ci = bootstrap(eps_fn, in_scores, out_scores, params)
print(f"Epsilon: {result.epsilon:.2f} (95% CI: [{eps_ci[0]:.2f}, {eps_ci[1]:.2f}])")
```

### 4. Audit Multiple Metrics

Don't rely on epsilon alone:
- AUROC captures overall attack strength
- TPR@FPR captures worst-case scenarios
- Max accuracy is easy to interpret

### 5. Run Multiple Seeds

```python
# Aggregate over multiple random splits
all_results = []
for seed in range(10):
    in_s, out_s = train_and_score(model_fn, dataset, canaries, seed=seed)
    all_results.append(audit(in_s, out_s))

epsilons = [r.epsilon for r in all_results]
print(f"Epsilon: {np.mean(epsilons):.2f} +/- {np.std(epsilons):.2f}")
```

## Common Pitfalls

### 1. Using Test Set as Canaries

**Wrong**: Using the model's test set as "out" canaries.

**Problem**: Test examples may have been seen during hyperparameter tuning.

**Fix**: Reserve a separate set of canaries never used for anything else.

### 2. Score Direction

**Wrong**: Using loss directly (higher loss = out).

**Problem**: Opaque expects higher scores = more likely IN.

**Fix**: Negate loss scores: `scores = -losses`

### 3. Too Few Canaries

**Wrong**: Auditing with 10 canaries.

**Problem**: Confidence intervals will be too wide to be useful.

**Fix**: Use at least 1000 canaries for reliable results.

## API Reference

See [Privacy Auditing API Reference](../api/auditing.md) for detailed function documentation.

## References

- Nasr et al. (2023). [Tight Auditing of Differentially Private Machine Learning](https://arxiv.org/abs/2305.08846)
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570)
- Jagielski et al. (2020). [Auditing Differentially Private Machine Learning](https://arxiv.org/abs/2006.07709)
