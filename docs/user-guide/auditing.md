# Privacy Auditing

Privacy accounting gives theoretical upper bounds on privacy loss.
Privacy auditing gives empirical lower bounds by running a membership
inference attack. If the audited epsilon exceeds the theoretical epsilon,
there is likely a bug in the implementation.

## How it works

Opaque implements one-run auditing
([Steinke, Nasr, Jagielski 2023](https://arxiv.org/abs/2305.08846))
with the tight order-statistics tests from
[Xiang, Chen, Kerkouche (2025)](https://arxiv.org/abs/2509.08704):

1. **Designate canaries.** Randomly select `m` examples as canaries.
2. **Flip coins.** For each canary, include or exclude it from training
   with probability 0.5.
3. **Train once.** Train on the resulting subset. Non-canary data is always
   included.
4. **Score.** Compute a membership score for each canary (negative loss by
   default — lower loss means more likely a member).
5. **Test.** Convert scores + coin flips into an ε lower bound via the
   audit-method surface on `OneRunEstimate`.

Only one training run is needed, unlike shadow-model approaches.

## Quick start

```python
import opaque.auditing as auditing
from opaque.random import key
from torch.utils.data import DataLoader, Subset

# 1. Partition: designate canaries and flip coins
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))

# 2. Train with DP-SGD on train_data ...

# 3. Score: compute membership scores for canaries
def canary_collate(examples):
    batch = data_collator(examples)
    return (batch["input_ids"].to(device),)

canary_loader = DataLoader(
    Subset(dataset, cf.canary_indices.tolist()),
    batch_size=32, collate_fn=canary_collate,
)
scores = auditing.loss_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)

# 4. Estimate: build the one-run estimate and query ε
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (audit):  {estimate.epsilon_at(delta=1e-5):.4f}")
print(f"AUC (attack): {estimate.attack_auc():.4f}")
```

## Default audit method: μ-GDP

`OneRunEstimate.epsilon_at(delta=)`, `.delta_at(epsilon=)`, `.beta_at(alpha=)`,
and `.advantage()` all dispatch to the **μ-GDP order-statistics test** from
Xiang et al. (2025).

Use μ-GDP only when it models the audited mechanism; otherwise use the
mechanism-agnostic method.

The dispatch requires `delta > 0`: μ-GDP is incompatible with pure ε-DP.

```python
estimate.epsilon_at(delta=1e-5)                       # default — μ-GDP
estimate.epsilon_at(delta=1e-5, significance=0.01)    # stricter confidence
estimate.epsilon_at(delta=1e-5, threshold=4.0)        # specific threshold
estimate.delta_at(epsilon=3.0)                        # inverse along the boundary
estimate.beta_at(alpha=0.01)                          # theoretical β under μ̂-GDP
estimate.advantage()                                  # TV(μ̂) = 2·Φ(μ̂/2) − 1
```

## Choosing a different method

`OneRunEstimate.gdp(grid_size=)` returns the underlying μ-GDP method
object — useful if you want to tune the integration grid or reuse a
single μ̂ across multiple queries:

```python
gdp = estimate.gdp(grid_size=20_000)   # tighter numerics
gdp.epsilon_at(delta=1e-5)
gdp.beta_at(alpha=0.01)
```

`OneRunEstimate.eps_delta()` returns the mechanism-agnostic (ε,δ)-DP
method object. Use it when:

- You audit a non-Gaussian-DP mechanism (Laplace, discrete Gaussian,
  heavy-tailed noise, …).
- You need pure ε-DP (δ=0) auditing.

```python
estimate.eps_delta().epsilon_at(delta=0.0)   # pure ε-DP
estimate.eps_delta().epsilon_at(delta=1e-5)  # general (ε, δ)-DP fallback
estimate.eps_delta().delta_at(epsilon=3.0)
```

`EpsDeltaMethod` deliberately does *not* expose `beta_at` / `advantage`:
the (ε,δ)-DP trade-off function is a family envelope, so those metrics
would be worst-case across the family rather than instance-specific. The
μ-GDP method's inferred μ̂ pins down a single curve, so its `beta_at` /
`advantage` are sharp.

## Integration with training

The four-step API separates concerns: **partition** (before training),
**train** (unchanged), **score** + **estimate** (after training).

### Step 1: Partition before training

Call `auditing.coin_flip` after preparing the dataset:

```python
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))

# Remove held-out canaries from training set
train_dataset = dataset.select(cf.train_indices(len(dataset)))
```

### Step 2: Train normally

No changes to the training loop. The dataset is already filtered.

### Step 3: Score and estimate after training

```python
scores = auditing.loss_scores(
    per_example_loss_fn, trained_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (audit): {estimate.epsilon_at(delta=1e-5):.4f}")
```

See [examples/train_dpsgd.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd.py)
and [examples/train_dpftrl.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpftrl.py)
for complete working examples with the `--audit` flag.

### Parameter reference

- **`batch_argnums`**: Which positional arguments of `loss_fn` come from the
  dataloader. `(1,)` means arg 1 is batched; `(1, 2)` means args 1 and 2 are
  batched. Same convention as `clipped_grad`.
- **`dataloader`**: Any iterable yielding batches. Each batch should be a
  tensor (single `batch_argnums`) or a tuple of tensors (multiple
  `batch_argnums`). Use a custom `collate_fn` on the DataLoader to handle
  dict-style batches (e.g., HuggingFace).
- **`reference_scores`**: Baseline scores from an untrained model. When
  provided, returned scores are `scores - reference_scores` (loss reduction).

## Attack-side metrics

Independent of the audit method, you can read attack-side empirical
metrics directly from the estimate:

```python
estimate.attack_auc()                  # ROC AUC (0.5 = random, 1.0 = perfect)
estimate.attack_beta_at(alpha=0.01)    # empirical attack β = 1 − TPR at FPR=α
```

AUC confidence intervals:

```python
ci = estimate.attack_auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

The `attack_` prefix distinguishes these from the (audit-side) `beta_at`
that reads the theoretical f-DP β at the inferred μ̂-GDP.

## Number of canaries

For the mechanism-agnostic $(\varepsilon,\delta)$ method, a perfect attack
has an approximate ceiling `ε ≲ ln(m / -ln(α))`. This does not apply to the
μ-GDP estimator:

| Canaries m | Hard ceiling at α=0.05 |
|---|---|
| 1 000 | ≈ 5.8 |
| 10 000 | ≈ 8.1 |
| 100 000 | ≈ 10.4 |
| 1 000 000 | ≈ 12.7 |

Imperfect attacks lower the mechanism-agnostic estimate further.

In practice:

| Canaries | Use case |
|---|---|
| 100 | Debugging only, very noisy |
| 500 | Detecting large implementation bugs |
| 1000+ | Recommended for reliable bounds |
| 5000+ | Needed for small-epsilon regimes |

## Interpreting results

The audited epsilon should be **below** the theoretical epsilon:

```
  ε (one-run):          1.50
  ε (theoretical):      3.00    ← expected: gap exists
```

**Gap is expected.** The audit runs a suboptimal attack and is capped by
`ln(m)`; the theoretical bound is a worst-case upper bound. A large gap is
normal, especially in high-noise regimes.

**Audited > theoretical is a red flag.** Investigate:

- Gradient clipping applied incorrectly
- Noise scaled to wrong sensitivity
- Privacy accounting doesn't match actual training procedure
- Data leakage outside the private training loop

## References

- Xiang, Chen, Kerkouche (2025). [Tight Privacy Auditing in One Run](https://arxiv.org/abs/2509.08704).
- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

## API reference

See [Auditing API Reference](../reference/auditing.md) for complete function
signatures.
