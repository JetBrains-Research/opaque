# Differential Privacy Concepts

This page covers the theory behind differential privacy (DP) and DP-SGD at the
level needed to use Opaque effectively. For the API itself, see the topic-specific
guides linked throughout.

## What differential privacy guarantees

Differential privacy is a mathematical property of a *mechanism* (an algorithm
that takes a dataset and produces an output). A mechanism M is
(epsilon, delta)-differentially private if, for any two datasets D and D' that
differ in a single record, and for any set of outputs S:

    P[M(D) in S] <= exp(epsilon) * P[M(D') in S] + delta

Informally: the output distribution barely changes when one person's data is
added or removed. An adversary observing the output cannot confidently determine
whether any specific individual was in the training set.

**epsilon** (privacy loss) controls how much the distributions can differ. Smaller
epsilon means stronger privacy. At epsilon=0, the mechanism reveals nothing
about individuals; at epsilon=infinity, there is no privacy guarantee.

**delta** bounds the probability that the epsilon guarantee fails. It should be
cryptographically small, typically smaller than 1/n where n is the dataset size.
Common values: 1e-5 to 1e-6.

### Practical interpretation

| epsilon | Interpretation |
|---------|----------------|
| 0.1     | Strong privacy. Significant accuracy loss expected. |
| 1.0     | Moderate privacy. Reasonable accuracy on large datasets. |
| 3.0     | Pragmatic privacy. Common target for ML training. |
| 10.0    | Weak privacy. Some protection against memorization. |

These are rough guidelines. The actual privacy-utility trade-off depends on the
dataset size, model capacity, and training procedure.

## The DP-SGD algorithm

DP-SGD ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)) modifies
standard stochastic gradient descent to provide differential privacy. The key
insight is that if we bound the influence of each training example on the
gradient update and add enough noise, the resulting model satisfies DP.

### Standard SGD vs DP-SGD

Standard SGD computes the average gradient over a batch:

```
gradient = mean([grad(loss, x_i) for x_i in batch])
params = params - lr * gradient
```

One outlier example can produce a large gradient that dominates the update.
Removing that person's data would noticeably change the trained model,
violating privacy.

DP-SGD adds two operations: **clipping** and **noise**.

```
per_example_grads = [grad(loss, x_i) for x_i in batch]
clipped_grads = [clip(g, max_norm=C) for g in per_example_grads]
noisy_gradient = sum(clipped_grads) + N(0, sigma^2 * I)
params = params - lr * noisy_gradient
```

### Step 1: Per-example gradients

Each training example produces its own gradient vector. This is more expensive
than standard batch-gradient computation but necessary for bounding per-example
influence.

Opaque computes per-example gradients efficiently using `torch.func.vmap` and
`torch.func.grad`. See [Gradient Clipping](clipping.md) for details.

### Step 2: Clipping

Each per-example gradient is clipped to a maximum L2 norm C (the *clip norm*).
If the gradient's norm exceeds C, it is scaled down to have norm exactly C.
Gradients with norm below C are left unchanged.

Clipping bounds the *sensitivity* of the gradient query: changing one training
example changes the sum of clipped gradients by at most C (under add-or-remove
neighboring relation) or 2C (under replace-one).

### Step 3: Noise addition

Gaussian noise with standard deviation sigma is added to the sum of clipped
gradients. The noise magnitude is proportional to the sensitivity (the clip
norm) and the desired privacy level.

The ratio sigma/C is the *noise multiplier*. Larger noise multiplier means
stronger privacy (smaller epsilon) but more gradient corruption.

### Step 4: Parameter update

The noisy gradient is applied to update model parameters, exactly as in
standard SGD. Any optimizer (SGD, Adam, AdamW) can be used.

## Privacy budget and composition

### The budget is finite

Each DP-SGD step consumes some privacy budget. Over T training steps, the total
privacy cost composes. More steps means larger total epsilon for the same noise
level, or more noise needed to maintain the same epsilon.

This creates a fundamental trade-off: training longer improves model quality
but costs more privacy. The **calibration** step (see
[Privacy Accounting](accounting.md)) finds the noise level that achieves a
target epsilon for a given number of steps.

### Composition

When the same dataset is used for multiple DP mechanisms (e.g., T steps of
DP-SGD), the total privacy loss is bounded by *composition theorems*.

**Basic composition**: epsilon values add. T steps of epsilon_0-DP gives
T * epsilon_0 total privacy. This is loose.

**Advanced composition** (Kairouz et al. 2015): total epsilon grows as
roughly sqrt(T) * epsilon_0 for small epsilon_0. Much tighter.

**PLD composition**: Opaque uses *Privacy Loss Distributions* (PLD), which
track the full distribution of privacy loss rather than just summary
statistics. PLD composition is numerically tight and dominates both basic and
advanced composition.

The `*` and `|` operators on `DpProcess` objects compute PLD composition.
See [Privacy Accounting](accounting.md).

### Subsampling amplification

If each training step samples a random subset of the dataset (rather than using
the full dataset), the privacy cost per step is reduced. This is *privacy
amplification by subsampling*.

With Poisson sampling at rate q (each example included independently with
probability q), the effective noise multiplier is amplified by approximately
1/q. For q=0.01 (1% sample rate), this is a 100x amplification.

Opaque's `PoissonSampler` implements Poisson subsampling. The accounting module
accounts for this amplification via `acc.poisson(mechanism, sample_rate)`.
See [Sampling & Microbatching](sampling.md).

## Privacy metrics

Opaque supports three families of privacy metrics, all derived from the same
underlying PLD:

### (epsilon, delta)-DP

The standard DP definition. Given a target delta, compute the smallest epsilon
such that the mechanism is (epsilon, delta)-DP.

```python
eps = training.epsilon_at(delta=1e-5)
```

### f-DP advantage

The total-variation advantage measures how well an adversary can distinguish
between the output distributions on neighboring datasets. An advantage of 0
means the adversary cannot distinguish at all (perfect privacy); an advantage
of 1 means the adversary can always distinguish.

```python
adv = training.advantage()
```

This is related to the trade-off function in f-DP (Dong et al. 2019).

### (alpha, beta) error rates

The hypothesis-testing interpretation. Given a Type-I error rate alpha (false
positive rate), compute the Type-II error rate beta (false negative rate) for
the optimal distinguishing test.

Higher beta means stronger privacy: the adversary cannot reject the null
hypothesis (that the data was not in the training set) without high false
negative rates.

```python
beta = training.beta_at(alpha=0.01)
```

## Key trade-offs in DP training

### Privacy vs accuracy

More noise means stronger privacy but degrades gradient signal. The model may
converge to a worse solution or not converge at all.

Strategies to improve accuracy at fixed privacy:
- Increase batch size (amplification reduces per-step cost)
- Use adaptive clipping to avoid over-clipping
- Use LoRA or other parameter-efficient methods to reduce gradient dimensionality
- Use matrix-factorization noise for correlated noise (DP-FTRL) instead of
  independent Gaussian noise

### Privacy vs compute

Per-example gradients are more expensive than batch gradients. Opaque uses
`torch.func.vmap` for efficient vectorized computation. Microbatching trades
compute time for memory by processing the batch in smaller chunks.

### Clip norm

The clip norm C controls the sensitivity-noise trade-off:
- Higher C: less clipping distortion, but more noise needed for the same privacy
- Lower C: more clipping distortion, but less noise needed

A common heuristic is to set C to the median gradient norm observed during
non-private training. Opaque's `adaptive_clipped_grad` automates this by
adjusting C to maintain a target fraction of clipped gradients (typically the
median, i.e., 50%).

### Number of steps

More training steps consume more privacy budget. For a fixed budget, longer
training requires more noise per step, which can reduce accuracy. There is
often an optimal number of steps that balances convergence with noise level.

## Neighboring relations

The privacy guarantee depends on what "differ in one record" means. Opaque
supports three neighboring relations:

| Relation | Meaning | Sensitivity |
|----------|---------|-------------|
| `ADD_OR_REMOVE_ONE` | D' = D +/- one record | C |
| `REPLACE_ONE` | D' = D with one record swapped | 2C |
| `REPLACE_SPECIAL` | D' = D with one record replaced by a no-op | C |

`REPLACE_SPECIAL` is the default and most common in DP-SGD. It models the
case where the adversary knows the dataset differs by replacing one real
example with a "zero" example that contributes nothing.

The choice of neighboring relation affects the sensitivity (and therefore the
noise calibration). Opaque's `ClipState.sensitivity()` method computes the
correct sensitivity for each relation.

## References

- [Abadi et al. 2016 - Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Dong et al. 2019 - Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383)
- [Kairouz et al. 2015 - The Composition Theorem for Differential Privacy](https://arxiv.org/abs/1311.0776)
- [Balle et al. 2020 - Hypothesis Testing Interpretations of DP](https://arxiv.org/abs/1905.02383)
- [Andrew et al. 2021 - Differentially Private Learning with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
