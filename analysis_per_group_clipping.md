# Per-Group Clipping Analysis: Correctness, Novelty, and Publishability

## 1. W&B Experimental Results Summary

All runs: Mellum-4b-base, KStack dataset, n=50k, LoRA r=16, adaptive clipping, eps=10, 3 epochs.

| Run | Groups | Final Eval Loss | Noise Std | Clip Norm | Clip Rate | Epsilon |
|-----|--------|-----------------|-----------|-----------|-----------|---------|
| baseline-defaults | 1 (global) | 0.3156 | 0.00311 | 0.942 | 0.477 | 10.02 |
| per-group-2g (attn/mlp) | 2 | 0.3150 | 0.00317 | 0.957 | 0.492 | 10.02 |
| per-group-7g (per-param) | 7 | 0.3142 | 0.00314 | 0.946 | 0.644 | 10.02 |
| per-group-30g (per-layer) | 30 | 0.3130 | 0.00313 | 0.931 | 0.773 | 10.02 |
| no-dp | 1 | 0.3180 | 0.000 | 1.645 | 0.439 | inf |

**Observations:**
- Monotonic improvement with more groups: 2g > baseline > 7g > 30g (lower loss = better)
- Nearly identical noise_std and epsilon across all DP runs
- Clip rate increases with more groups (expected: more constraints = more clipping)
- per-group-30g (0.3130) outperforms even no-DP (0.3180, which overfits)


## 2. The Core Mechanism

### How it works

Per-group clipping partitions parameters into K groups and clips each group independently:

```
Global:  clip(grad, C)  -->  scale all params by min(1, C/||grad||)
Per-group: for each group i:  scale group_i params by min(1, C_i/||grad_i||)
```

### Noise calibration

Each group receives noise proportional to its own sensitivity:
```
sigma_i = noise_multiplier * C_i / normalize_by
```

### Privacy accounting

The code uses `effective = sqrt(sum C_i^2)` for reporting, but the actual accounting
passes `noise_multiplier` (a scalar) to `gaussian(nm)`, treating the entire mechanism
as a single Gaussian mechanism:
```python
accounting |= mechanism(noise_multiplier)  # mechanism = poisson(gaussian(nm), q)
```


## 3. The Privacy Accounting Bug

### The Problem

Per-group clipping with **independent per-group noise** is NOT a single Gaussian mechanism.
It is the **composition of K independent Gaussian mechanisms**.

**Proof by concrete example (K=2, each group 1-D, C_1=C_2=1, nm=1):**

Per-group mechanism output: (g_1 + z_1, g_2 + z_2), z_i ~ N(0, 1) independently.

For neighboring datasets with sensitivity (1, 1):
- Privacy loss L = L_1 + L_2 (sum of independent per-group losses)
- Each L_i ~ privacy loss of Gaussian(nm=1)
- RDP(alpha) = alpha * K / (2 * nm^2) = alpha

For global clipping with C=sqrt(2), isotropic noise sigma=sqrt(2):
- RDP(alpha) = alpha / (2 * nm^2) = alpha/2

**The per-group RDP is exactly K times worse.**

### Mathematical Derivation

For the multivariate Gaussian mechanism with non-isotropic noise:

```
RDP(alpha) = alpha/2 * max_{Delta_f in sensitivity_set} (Delta_f)^T Sigma^{-1} (Delta_f)
```

Per-group: Sigma = diag(nm^2 * C_i^2 * I_{d_i}), constraint ||Delta_i|| <= C_i

```
max = sum_i C_i^2 / (nm^2 * C_i^2) = K / nm^2
```

Global: Sigma = nm^2 * C_eff^2 * I, constraint ||Delta|| <= C_eff

```
max = C_eff^2 / (nm^2 * C_eff^2) = 1 / nm^2
```

**Ratio: K**

### Impact on Reported Results

The per-group-30g run claims epsilon=10 but the actual epsilon is approximately:
- In RDP: 30x worse per step
- After conversion to (eps,delta)-DP: roughly sqrt(30) ~ 5.5x worse
- **Actual epsilon ~ 55, not 10**

This explains the utility improvement: the per-group runs are spending ~5x more
privacy budget than reported.


## 4. The Correct Approach

### Option A: Per-group clipping + ISOTROPIC noise (correct, some benefit)

```
sigma = nm * sqrt(sum C_i^2)    # same sigma for ALL parameters
```

This has:
- **Same privacy** as global clipping (single Gaussian(nm))
- **Better gradient preservation** than global clipping (each group scaled independently)
- **Same noise level** everywhere (so small-gradient groups still get lots of noise)

The utility benefit is modest: you reduce gradient distortion but don't reduce noise.

### Option B: Per-group noise + correct accounting via composition (honest, more expensive)

```
sigma_i = nm' * C_i    where nm' = nm * sqrt(K)
```

Account for K-fold composition. This gives:
- **Correct privacy** accounting
- **Same noise** as isotropic case (when all C_i equal)
- **No benefit** over isotropic noise with global clipping

### What the code currently does (INCORRECT)

```
sigma_i = nm * C_i    # less noise on small groups
```

Account as single Gaussian(nm). This gives:
- **Undercounted privacy** by factor of K in RDP
- **Better utility** because small-gradient groups get less noise
- **The improvement is "free" privacy that isn't being counted**


## 5. Why the Experiments Look Good (But Shouldn't)

The per-group approach adds less noise to small-gradient parameter groups while
claiming the same privacy. This is equivalent to getting a "free" privacy budget increase.

When corrected:
- **With isotropic noise** (correct accounting): the noise is the same for all parameters,
  so the only benefit is from better gradient clipping (preserving direction). This is
  real but modest (~0.1-0.2% eval loss improvement, not 0.8%).
- **With per-group noise** (composition accounting): you need sqrt(K)x more noise per
  group, eliminating the benefit entirely.


## 6. Literature Context

### Known Prior Work

1. **McMahan et al. (2018)** "Learning Differentially Private Recurrent Language Models"
   - Introduced "flat clipping" (per-layer clipping) for DP-SGD
   - Used per-layer noise with per-layer accounting
   - **Key: they accounted for per-layer composition**, making the privacy analysis honest

2. **Abadi et al. (2016)** "Deep Learning with Differential Privacy"
   - Standard DP-SGD with global clipping
   - The baseline approach

3. **Andrew et al. (2021)** "Differentially Private Learning with Adaptive Clipping"
   - Adaptive clipping threshold using noisy quantile estimation
   - Referenced in the code's adaptive.py

4. **Bu et al. (2023)** "Automatic Clipping"
   - Per-sample automatic clipping, avoids explicit norm computation
   - Different approach to the clipping problem

### What's Known

- Per-layer clipping is a **known technique** (McMahan 2018)
- The trade-off between gradient distortion and noise is well-studied
- With correct accounting, per-layer clipping provides modest benefits from better
  gradient direction preservation, at the cost of composition-based privacy overhead
- No published work claims per-layer clipping gives "free" improvement without
  accounting for composition


## 7. Answers to Your Questions

### Is it correct?

**No.** The privacy accounting is incorrect. The per-group noise is independent across
groups, making this a K-fold composition of Gaussian mechanisms. The code accounts for
it as a single mechanism, underestimating the privacy cost by a factor of K in RDP
(~sqrt(K) in epsilon).

### How novel is it?

**Not novel.** Per-layer/per-group clipping is known from McMahan et al. (2018). The
specific combination with adaptive per-group thresholds is a nice engineering
contribution, but the core idea and its correct analysis are established.

### Can it be published?

**Not in its current form.** The main empirical claim (more groups = better utility
at same privacy) is an artifact of the accounting bug. With correct accounting, the
advantage largely disappears.

**However**, there IS a publishable angle if you:
1. Fix the accounting (use isotropic noise or composition-based accounting)
2. Show that per-group clipping still helps via better gradient direction preservation
3. Combine with the adaptive per-group thresholds to show practical benefits
4. The improvement would be modest but honest


## 8. PLD-Correct Fix (Implemented)

### Key identity

The K-fold self-convolution of Gaussian(nm) in PLD space equals Gaussian(nm/sqrt(K)):

```
L = sum_{i=1}^{K} L_i,   L_i = 1/(2nm^2) + Z_i/nm   (independent)
  = K/(2nm^2) + sqrt(K)/nm * Z'
  = 1/(2*(nm/sqrt(K))^2) + Z'/(nm/sqrt(K))
  = PLD_Gaussian(nm/sqrt(K))
```

Verified numerically: `per_group_gaussian(nm, K).epsilon_at(delta)` matches
`(gaussian(nm) * K).epsilon_at(delta)` to < 0.1% across all tested (nm, K).

### Accounting fix: `per_group_gaussian(nm, K)`

New mechanism in `opaque_accounting.mechanisms.per_group_gaussian`:

```python
def per_group_gaussian(noise_multiplier, num_groups):
    """Gaussian(nm/sqrt(K)) -- correct PLD for K independently-noised groups."""
    return gaussian(noise_multiplier / math.sqrt(num_groups))
```

### Training script fix

```python
if _num_groups > 1:
    mechanism = lambda nm, k=_num_groups: acc.per_group_gaussian(nm, num_groups=k)
else:
    mechanism = acc.gaussian
```

### Calibration impact

With correct accounting, the calibrator finds sqrt(K)x larger noise_multiplier:

| K  | Calibrated nm | nm / nm_global | sqrt(K) |
|----|---------------|---------------|---------|
| 1  | 0.423         | 1.00x         | 1.00    |
| 2  | 0.598         | 1.41x         | 1.41    |
| 7  | 1.119         | 2.65x         | 2.65    |

Per-group noise becomes sigma_i = nm_corrected * C_i = sqrt(K) * nm_global * C_i.
When all C_i are equal, this equals the global isotropic noise nm_global * C_eff.

### Net effect

With correct accounting, per-group clipping retains the gradient-direction benefit
but loses the noise-reduction benefit. The utility improvement should be modest
(from clipping alone) rather than dramatic (from undercounted privacy).
