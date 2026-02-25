# Plan: Redesign `opaque.auditing` API for HuggingFace Integration

## Context

The `examples/train_causal_lm.py` example defines the target HuggingFace integration
pattern we want to support: HF datasets, `DataCollatorForLanguageModeling`, `PoissonSampler`,
functional model via `make_functional`, and `clipped_grad`-style loss functions. The current
`opaque.auditing` API does not fit this pattern.

This plan redesigns the auditing API to integrate cleanly with the HF training loop,
keeps only the "Privacy Auditing with One (1) Training Run" (Steinke et al. 2023) approach,
and fixes implementation gaps relative to the paper.

---

## 1. API Mismatch Analysis (Current Problems)

### Problem 1: `score_by_loss` hardcodes `(params, x, y)` tuple pattern

Current code in `scoring.py`:
```python
per_example_fn = torch.func.vmap(loss_fn, in_dims=(None, 0, 0))
for batch in loader:
    x, y = batch[0], batch[1]
    losses = per_example_fn(params, x, y)
```

The example uses `loss_fn(trainable, tokens_batch)` with a single batched tensor
from an HF `DataCollator`, not `(x, y)` tuples. The `in_dims` and batch unpacking
are both wrong for the HF pattern.

### Problem 2: `evaluate()` requires `loss_fn(params, x, y)` signature

The example's per-example loss function:
```python
def per_example_loss_fn(trainable, tokens_batch):
    output = fmodel(merged_params(trainable), tokens_batch, labels=tokens_batch)
    return output.loss
```

This takes `(params, tokens)` with a **single batched argument** — the same signature
used by `clipped_grad(batch_argnums=(1,))`. The auditing API should accept the
same `batch_argnums` convention.

### Problem 3: `CoinFlipExperiment.subset()` breaks HF datasets

`subset()` returns `torch.utils.data.Subset`, which wraps the HF `Dataset` and
loses all HF methods (`.map()`, `.select()`, column manipulation). The example
needs to tokenize, remove columns, etc. — all HF operations.

### Problem 4: DataLoader incompatibility

`score_by_loss` creates its own `DataLoader` without a `collate_fn`. The example
requires `DataCollatorForLanguageModeling` for proper padding/batching. Without it,
dict-style HF batches fail.

### Problem 5: Auditing not integrated in the example

The example has `--audit` and `--audit_canaries` CLI flags but the training loop
never actually uses them. The auditing workflow needs to be wired in.

---

## 2. Paper vs. Implementation Gap Analysis

### Gap 1: Scoring function is weaker than recommended

**Paper recommends (black-box):** `Score(x_i) = loss(w^0, x_i) - loss(w^l, x_i)`
(loss *decrease* from initialization to final model).

**Current implementation:** `Score(x_i) = -loss(w^l, x_i)` (just negative final loss).

The loss-decrease score is a stronger membership signal because it measures how much
the model improved on each example. The paper explicitly uses this.

### Gap 2: Only positive guesses, no negative guesses

**Paper (Algorithm 1):** Uses both k+ positive guesses ("in") and k- negative
guesses ("out"). The test statistic `W = Σ max{0, T_i * S_i}` counts correct
guesses on **both sides**. Total guesses `r = k+ + k-`, correct `v = TP + TN`.

**Current implementation:** Only searches over positive guesses:
`n_guess = TP + FP`, `n_correct = TP`. This ignores correct negative guesses,
which can reduce statistical power.

### Gap 3: No support for fixed-size datasets (Appendix A)

The paper discusses a variant where the training set size stays constant
(by using replacement pairs instead of inclusion/exclusion). This is important
when the training set size materially affects training dynamics.

*Lower priority — can be added later.*

---

## 3. New API Design

### 3.1 `auditing.setup()` — Unchanged

```python
def setup(dataset, *, num_canaries: int, key: RngKey) -> CoinFlipExperiment:
    """Set up a one-run privacy audit experiment."""
```

No changes needed. Works with anything that has `len()`.

### 3.2 `CoinFlipExperiment` — Changes

```python
class CoinFlipExperiment:
    num_canaries: int
    canary_indices: np.ndarray   # RENAMED from _canary_indices (now public)
    in_indices: np.ndarray       # Canaries included (coin = heads)
    out_indices: np.ndarray      # Canaries excluded (coin = tails)

    def train_indices(self, dataset_size: int) -> list[int]:
        """Indices to use for training (all except held-out canaries).
        Returns a list for direct use with HF dataset.select()."""

    def subset(self, dataset) -> Subset:
        """Return torch Subset for training (for torch-style datasets)."""

    def audit(self, scores: np.ndarray) -> AuditResult:
        """Split scores by coin flip and return AuditResult."""
```

Changes:
- **`canary_indices` made public** (rename `_canary_indices` → `canary_indices`)
- **`train_indices()` returns `list[int]`** instead of `np.ndarray` for HF `.select()` compat
- **Remove `canary_subset()`** — trivially replaced by `dataset.select(exp.canary_indices.tolist())`

### 3.3 `score()` — New flexible scoring function (replaces `score_by_loss`)

```python
def score(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    indices: np.ndarray | None = None,
    collate_fn: Callable | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-example membership scores (negative loss) via vmap.

    Follows the same convention as clipped_grad: loss_fn receives
    positional args, and batch_argnums identifies which args come
    from the dataset and are vmapped over.

    Args:
        loss_fn: Per-example loss function, same signature as clipped_grad.
        *args: Non-batched arguments to loss_fn (e.g., model parameters).
        batch_argnums: Indices of loss_fn args that come from dataset batches
            (same convention as clipped_grad).
        dataset: Dataset to score.
        indices: If provided, only score these dataset indices.
        collate_fn: Collate function for DataLoader (e.g., DataCollatorForLanguageModeling).
        batch_size: Batch size for scoring.

    Returns:
        Array of scores, shape (n,). Higher = more likely member.

    Example (HF pattern)::

        scores = auditing.score(
            per_example_loss_fn,
            trainable_params,        # arg 0: not batched
            batch_argnums=(1,),      # arg 1: tokens from dataset
            dataset=train_dataset,
            collate_fn=data_collator,
            batch_size=32,
        )

    Example (torch (x, y) pattern)::

        scores = auditing.score(
            loss_fn,
            params,                  # arg 0: not batched
            batch_argnums=(1, 2),    # args 1,2: x,y from dataset
            dataset=dataset,
            batch_size=256,
        )
    """
```

Implementation details:
- Creates `DataLoader(subset, batch_size=..., collate_fn=..., shuffle=False)`
- For each batch, extracts batched args from the batch (dict → values or tuple → elements)
- Constructs `in_dims` from `batch_argnums`: `None` for non-batch, `0` for batch
- Calls `torch.func.vmap(loss_fn, in_dims=in_dims)` on full args
- Returns `-losses` as scores

Batch unpacking logic:
- If `collate_fn` produces a dict: pass `batch["input_ids"]` (or appropriate key) as the batched arg
- If `collate_fn` produces a tuple/list: pass `batch[i]` for each batch argnum
- **Simpler approach**: Accept a `batch_unpack` callable or auto-detect

Actually, the simpler design: the dataset + collate_fn produce batches, and batch_argnums
specifies which positions in `loss_fn(arg0, arg1, ...)` are batched. We need a way to map
batch outputs to positional args. Two approaches:

**Option A (recommended): Use `keep_batch_dim=True` pattern from `clipped_grad`**

The existing `clipped_grad` already handles this:
```python
grad_fn, clip_state = clipped_grad(
    per_example_loss_fn,
    argnums=0,
    batch_argnums=(1,),
    keep_batch_dim=True,
)
(grads, aux), clip_state = grad_fn(trainable_params, tokens, state=clip_state)
```

For scoring, the user calls:
```python
scores = auditing.score(
    per_example_loss_fn,
    trainable_params,
    batch_argnums=(1,),
    dataset=canary_dataset,
    collate_fn=data_collator,
    batch_key="input_ids",     # Which key from collated dict to use as batch arg
    batch_size=32,
)
```

**Option B (even simpler): Let users pass batches directly**

Provide a lower-level function:
```python
def score_batches(
    loss_fn: Callable,
    *args: Any,
    batches: Iterable,
    batch_argnums: tuple[int, ...],
) -> np.ndarray:
    """Score pre-formed batches."""
```

And users handle their own DataLoader:
```python
canary_loader = DataLoader(canary_data, batch_size=32, collate_fn=data_collator)
scores = auditing.score_batches(
    per_example_loss_fn,
    trainable_params,
    batches=((b["input_ids"].to(device),) for b in canary_loader),
    batch_argnums=(1,),
)
```

**Decision: Go with Option A (with batch_key) as the primary API, expose the lower-level
function for advanced users.**

### 3.4 `evaluate()` — Updated

```python
def evaluate(
    experiment: CoinFlipExperiment,
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    collate_fn: Callable | None = None,
    batch_key: str | None = None,
    batch_size: int = 256,
) -> AuditResult:
    """Score canaries and produce audit results in one call.

    Combines score() + experiment.audit() for convenience.
    """
```

### 3.5 `AuditResult` — Minor changes

Keep all existing methods. Add:

1. **Two-sided threshold search in `epsilon_one_run()`**: For each threshold,
   compute both the positive-only version (current) AND the two-sided version
   (`r = m`, `v = TP + TN`). Take the best.

2. **Summary includes theoretical epsilon comparison** when provided:
   ```python
   def summary(self, *, significance=0.05, delta=0.0, theoretical_epsilon=None) -> str:
   ```

### 3.6 Remove `score_by_loss` (replaced by `score`)

The old `score_by_loss` with its hardcoded `(params, x, y)` signature is replaced
by the new `score()` function.

---

## 4. Implementation Steps

### Step 1: Update `CoinFlipExperiment` (audit.py)

- Rename `_canary_indices` → `canary_indices` (public attribute)
- Change `train_indices()` to return `list[int]`
- Remove `canary_subset()` method
- Update all internal references

### Step 2: Rewrite `scoring.py` with new `score()` function

- New `score()` function with `batch_argnums` and `collate_fn` support
- Handle dict-style batches (HF) and tuple-style batches (torch)
- Use `batch_key` parameter for dict batches
- Maintain vmap-based per-example scoring

### Step 3: Update `__init__.py` with new exports

- Export `score` instead of `score_by_loss`
- Update `evaluate()` signature
- Update module docstring with new HF-compatible example

### Step 4: Add two-sided threshold search to `epsilon_one_run()`

- For each Pareto-optimal threshold, also compute r=m, v=TP+TN version
- Take best epsilon across both one-sided and two-sided at each threshold
- This is a correctness improvement from the paper

### Step 5: Integrate auditing into `examples/train_causal_lm.py`

Wire in the full auditing workflow:
```python
# Before training:
if args.audit:
    experiment = auditing.setup(train_dataset, num_canaries=args.audit_canaries, key=key(args.seed))
    audit_train_indices = experiment.train_indices(len(train_dataset))
    train_dataset = train_dataset.select(audit_train_indices)
    # Recompute sample_rate with new dataset size

# After training:
if args.audit:
    audit = auditing.evaluate(
        experiment,
        per_example_loss_fn,
        trainable_params,
        batch_argnums=(1,),
        dataset=full_train_dataset,   # The original full dataset
        collate_fn=data_collator,
        batch_key="input_ids",
        batch_size=args.audit_batch_size,
    )
    print(audit.summary(delta=args.target_delta, theoretical_epsilon=args.target_epsilon))
```

### Step 6: Update tests

- Update `test_scoring.py` for new `score()` signature
- Add tests for dict-style batches (HF pattern)
- Add tests for two-sided epsilon_one_run
- Update `test_integration.py`

### Step 7: Update documentation

- Update `docs/user-guide/auditing.md` with HF example
- Update `docs/api/auditing.md` with new function signatures
- Update `docs/tutorials/privacy_auditing.ipynb`

---

## 5. Detailed Scoring Function Design

The key challenge is mapping DataLoader outputs to loss_fn arguments.

**For HF dict batches** (from `DataCollatorForLanguageModeling`):
```python
batch = {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}
```
The user's loss_fn takes `(params, tokens)` where tokens = `batch["input_ids"]`.
We need `batch_key="input_ids"` to extract the right tensor.

**For torch tuple datasets** (from `TensorDataset`):
```python
batch = (x_tensor, y_tensor)
```
The user's loss_fn takes `(params, x, y)` where x = `batch[0]`, y = `batch[1]`.
With `batch_argnums=(1, 2)`, we know args 1 and 2 come from the batch.

**Implementation approach:**

```python
def score(loss_fn, *args, batch_argnums, dataset, indices=None,
          collate_fn=None, batch_key=None, batch_size=256, device=None):

    # Create subset if indices provided
    if indices is not None:
        dataset = Subset(dataset, indices.tolist()) if hasattr(dataset, '__getitem__') else dataset.select(indices.tolist())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Build in_dims: None for non-batch args, 0 for batch args
    total_args = len(args) + len(batch_argnums)
    in_dims = tuple(0 if i in batch_argnums else None for i in range(total_args))
    per_example_fn = torch.func.vmap(loss_fn, in_dims=in_dims)

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            # Extract batch tensors
            if isinstance(batch, dict):
                if batch_key is not None:
                    batch_tensors = [batch[batch_key].to(device)]
                else:
                    # Use first key as default
                    batch_tensors = [next(iter(batch.values())).to(device)]
            elif isinstance(batch, (list, tuple)):
                batch_tensors = [batch[i].to(device) for i in range(len(batch_argnums))]
            else:
                batch_tensors = [batch.to(device)]

            # Merge args and batch_tensors in correct positions
            full_args = list(args)
            for idx, bt in zip(sorted(batch_argnums), batch_tensors):
                full_args.insert(idx, bt)

            losses = per_example_fn(*full_args)
            all_scores.append(-losses.detach().cpu().numpy())

    return np.concatenate(all_scores)
```

Wait, this arg-merging approach is fragile. Let me think about a cleaner design.

Actually, the cleanest approach mirrors `clipped_grad` exactly. In `clipped_grad`, the user calls:
```python
(grads, aux), clip_state = grad_fn(trainable_params, tokens, state=clip_state)
```

The user passes ALL args (both non-batched and batched) positionally. `batch_argnums`
tells which are batched.

For scoring, the user should also pass all args. But the batched args come from the
DataLoader. So we need to know: which args does the user pass, and which come from
the DataLoader?

**Simpler design: separate non-batch args from batch extraction.**

```python
def score(
    loss_fn,
    *args,               # Non-batched args (e.g., params)
    dataset,
    batch_argnums,       # Positions in loss_fn signature for batched args
    collate_fn=None,
    batch_unpack=None,   # Callable: batch -> tuple of tensors for batch_argnums
    batch_size=256,
):
```

Where `batch_unpack` maps a DataLoader batch to the tuple of batched tensors:
```python
# HF pattern:
batch_unpack = lambda b: (b["input_ids"].to(device),)

# Torch pattern:
batch_unpack = lambda b: (b[0].to(device), b[1].to(device))
```

If `batch_unpack` is None, use a default that handles common cases.

This is explicit and avoids magic. Let me incorporate this into the plan.

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `packages/opaque/src/opaque/auditing/audit.py` | Rename `_canary_indices`, remove `canary_subset`, update `epsilon_one_run` for two-sided search, add `theoretical_epsilon` to summary |
| `packages/opaque/src/opaque/auditing/scoring.py` | Complete rewrite: new `score()` with `batch_argnums`, `collate_fn`, `batch_unpack` |
| `packages/opaque/src/opaque/auditing/__init__.py` | Update exports (`score` replaces `score_by_loss`), update `evaluate()` |
| `packages/opaque/tests/auditing/test_scoring.py` | Update for new `score()` signature, add HF-style tests |
| `packages/opaque/tests/auditing/test_audit.py` | Update for public `canary_indices`, removed `canary_subset` |
| `packages/opaque/tests/auditing/test_integration.py` | Update integration tests |
| `examples/train_causal_lm.py` | Wire in auditing workflow |
| `docs/user-guide/auditing.md` | HF example, updated API |
| `docs/api/auditing.md` | Updated function signatures |
| `docs/tutorials/privacy_auditing.ipynb` | Updated tutorial |
