# Privacy Auditing

Privacy accounting gives theoretical upper bounds on privacy loss.
Privacy auditing gives empirical lower bounds by running a membership
inference attack. If the audited epsilon exceeds the theoretical epsilon,
there is likely a bug in the implementation.

## How it works

Opaque implements one-run auditing
([Steinke, Nasr, Jagielski 2023](https://arxiv.org/abs/2305.08846))
with the tighter order-statistics tests from
[Xiang, Chen, Kerkouche (2025)](https://arxiv.org/abs/2509.08704):

1. **Designate canaries.** Randomly select `m` examples as canaries.
2. **Flip coins.** For each canary, include or exclude it from training
   with probability 0.5.
3. **Train once.** Train on the resulting subset. Non-canary data is always
   included.
4. **Score.** Compute a membership score for each canary (negative loss by
   default — lower loss means more likely a member).
5. **Test.** Pick an audit method (`eps_delta` or `gdp`) and convert
   scores + coin flips into an ε lower bound.

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

# 4. Estimate: build the one-run estimate
estimate = auditing.one_run(scores, coin_flip=cf)

# 5. Pick an audit method and query ε
print(f"ε (eps_delta): {estimate.eps_delta().epsilon_at(delta=1e-5):.4f}")
print(f"ε (gdp):       {estimate.gdp().epsilon_at(delta=1e-5):.4f}")
print(f"AUC:           {estimate.auc():.4f}")
```

## Integration with training

The three-step API separates concerns: **partition** (before training),
**score** (after training), **estimate** (compute metrics).

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
print(f"ε (eps_delta): {estimate.eps_delta().epsilon_at(delta=1e-5):.4f}")
```

See [examples/train_causal_lm.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py)
and [examples/train_dp_ftrl.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dp_ftrl.py)
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

## Audit methods

`OneRunEstimate` exposes two factory methods, one per f-DP family. Pick one
explicitly — neither is the silent default — and chain `.epsilon_at(...)`:

| Method | f-DP family | Mechanism scope | `delta` |
|---|---|---|---|
| `estimate.eps_delta()` | (ε, δ)-DP | Any DP mechanism (Laplace, discrete Gaussian, …) | `delta >= 0` (pure ε-DP OK) |
| `estimate.gdp()` | μ-GDP | Gaussian-DP mechanisms (DP-SGD, MF DP-FTRL) | `delta > 0` (required) |

`gdp()` is strictly tighter than `eps_delta()` when the audited mechanism
satisfies Gaussian DP. For arbitrary mechanisms — or pure ε-DP — use
`eps_delta()`.

```python
estimate.eps_delta().epsilon_at(delta=1e-5)
estimate.eps_delta().epsilon_at(delta=1e-5, significance=0.01)
estimate.eps_delta().epsilon_at(delta=1e-5, threshold=4.0)

estimate.gdp().epsilon_at(delta=1e-5)
estimate.gdp(grid_size=20_000).epsilon_at(delta=1e-5)  # tighter numerical
```

Both methods also expose `delta_at(epsilon=)` — the inverse query along the
audit's (ε, δ) boundary:

```python
estimate.eps_delta().delta_at(epsilon=3.0)   # largest δ for which ε ≥ 3.0 is certified
estimate.gdp().delta_at(epsilon=3.0)
```

`gdp()` additionally mirrors the full
[`Pld`](accounting.md#privacy-metrics) surface — the inferred μ̂ pins down
a single f-DP curve so the values below are sharp:

```python
estimate.gdp().beta_at(alpha=0.01)   # theoretical β under inferred μ̂-GDP
estimate.gdp().advantage()           # 2·Φ(μ̂/2) − 1
```

`eps_delta()` deliberately does *not* expose `beta_at` / `advantage`:
the (ε, δ)-DP trade-off function is a family envelope rather than a single
mechanism's curve, and the resulting values would be worst-case across the
family rather than instance-specific.

## Attack metrics

Independent of which audit method you chose, you can read attack-side
metrics from the estimate directly:

```python
estimate.auc()                  # ROC AUC (0.5 = random, 1.0 = perfect)
estimate.beta_at(alpha=0.01)    # empirical attack β at 1% FPR (1 − TPR)
```

AUC confidence intervals:

```python
ci = estimate.auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Number of canaries

The one-run audit has a hard ceiling at `ε ≲ ln(m / -ln(α))` for a perfect
attack with `m` canaries at significance `α`. Anything above that is
unreachable regardless of which method you pick:

| Canaries m | Hard ceiling at α=0.05 |
|---|---|
| 1 000 | ≈ 5.8 |
| 10 000 | ≈ 8.1 |
| 100 000 | ≈ 10.4 |
| 1 000 000 | ≈ 12.7 |

Imperfect attacks lower this further. So if you train at `ε = 60`, the
audit cannot certify anywhere near it — that's a property of one-run
auditing, not an Opaque limitation.

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
