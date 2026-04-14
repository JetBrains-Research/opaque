# JME (Joint Moment Estimation) → DP-Adam via MF: Implementation Plan

**Paper**: Kalinin, Upadhyay, Lampert — "Continual Release Moment Estimation with Differential Privacy" ([arXiv:2502.06597](https://arxiv.org/abs/2502.06597), NeurIPS 2025)

**Goal**: Enable DP-Adam and DP-AdaGrad with MF correlated noise by implementing JME within Opaque's existing matrix factorization framework.

**Design**: JME is NOT a new noise mechanism type — it is a **calibration result**
that allows two standard `mf_noise` streams to share a privacy budget. The math
lives in `noise/mf/jme.py` (three pure functions), and the integration lives
directly in `train_dp_ftrl.py --optimizer adam`. No `JmeStrategy`, no `jme_noise`
entry point — just two `mf_noise` calls with the right stddevs.

---

## 1. Background & Motivation

### The Current Limitation

`train_dp_ftrl.py` explicitly restricts DP-FTRL to SGD with Polyak momentum:

> "SGD with Polyak momentum ONLY — Adam/AdaGrad are nonlinear operators on the
> gradient stream and destroy the noise correlation structure that MF depends on
> for utility gains."

Adam requires estimating **two** running statistics from the gradient stream:
- **First moment** (mean): `m_t = β₁·m_{t-1} + (1-β₁)·g_t`
- **Second moment** (uncentered variance): `v_t = β₂·v_{t-1} + (1-β₂)·g_t²`

Since `v_t` is a **nonlinear** function of gradients (`g_t²`), naively adding MF correlated noise only to `g_t` doesn't help — the second moment still needs independent noise, and the noise budget must be split between the two estimates.

### What JME Does

JME solves this by jointly analyzing the sensitivity of estimating **both** moments simultaneously. The key insight (Theorem 3.2 of the paper):

> When λ is set to `‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})`, the **joint sensitivity** of the combined (first moment, second moment) estimation equals the sensitivity of the first moment **alone**:
>
> `s_joint = 2ζ · ‖C₁‖_{1→2}`

This means: **the second moment estimation is "free" from a privacy perspective** — no additional noise budget is required. Both moments get their own MF correlated noise stream, with the second moment's noise scaled by `λ^{-1/2}`.

### Why This Matters for Opaque

- **Adam is the default optimizer** for LLM fine-tuning. Enabling DP-Adam with correlated noise would be a major differentiator.
- The Toeplitz/streaming matrix factorization infrastructure already exists in Opaque.
- JME works with **any** existing strategy (BandMF, BLT, BISR, λ-CGD, identity) — it wraps the strategy to produce two noise streams instead of one.

---

## 2. Mathematical Framework

### 2.1 JME Algorithm (from Algorithm 1 of the paper)

**Input**: Stream `x₁, ..., xₙ ∈ ℝ^d` with `‖xᵢ‖₂ ≤ ζ`; workload matrices `A₁, A₂`; privacy `(ε, δ)`; noise-shaping matrices `C₁, C₂`.

**Setup**:
1. `λ ← ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})`
   - where `c₁ = 8/(11 + 5√5) ≈ 0.339` for `d=1`, and `c_d = 2` for `d ≥ 2`
2. `s ← 2ζ · ‖C₁‖_{1→2}` (joint sensitivity — same as first-moment-only sensitivity!)
3. `Z₁ ~ N(0, σ²_{ε,δ} · s²)^{n×d}` (noise for first moment)
4. `Z₂ ~ N(0, σ²_{ε,δ} · s²)^{n×d²}` (noise for second moment)

**Per step t**:
1. `x̂_t ← x_t + [C₁⁻¹ · Z₁]_{[t,·]}` (noisy gradient via existing MF mechanism)
2. `x̂_t⊗x̂_t ← x_t⊗x_t + λ^{-1/2} · [C₂⁻¹ · Z₂]_{[t,·,·]}` (noisy squared gradient via MF)
3. Yield: `Ŷ_t = Σᵢ [A₁]_{t,i} · x̂ᵢ` and `Ŝ_t = Σᵢ [A₂]_{t,i} · (x̂ᵢ⊗x̂ᵢ)`

### 2.2 Mapping to Adam

For DP-Adam via JME:
- `x_t = g_t` (clipped gradient at step t)
- `ζ = clipping_norm / normalize_by` (the sensitivity from clipping)
- First moment workload `A₁`: exponential moving average with decay `β₁`
- Second moment workload `A₂`: exponential moving average with decay `β₂`
- `C₁`: any MF strategy (BandMF, BLT, BISR, λ-CGD) for first moment noise shaping
- `C₂`: any MF strategy for second moment noise shaping (can use same or different)
- Adam update: `θ_{t+1} = θ_t - lr · m̂_t / (√v̂_t + ε_adam)`

### 2.3 Sensitivity: `‖C‖_{1→2}` Computation

The `1→2` norm `‖C‖_{1→2} = max_j ‖C[:,j]‖₂` is the **maximum column L2 norm** of the strategy matrix `C`. This is exactly the **single-participation sensitivity** already computed by every Opaque strategy.

For each existing strategy type:
| Strategy | `‖C‖_{1→2}` already available as |
|----------|-------------------------------------|
| BandMF | `strategy.sensitivity` |
| BLT | `strategy.sensitivity` |
| BISR | `strategy.sensitivity` (when `max_participations=1`) |
| λ-CGD | `_native.lambda_cgd_max_column_norm(λ, n)` |
| Identity | `1.0` |

For multi-participation settings, the **joint** sensitivity formula in the paper still uses `‖C‖_{1→2}` (max column norm), **not** the multi-participation sensitivity. The multi-participation sensitivity is used for accounting, while the joint sensitivity formula uses max column norm for noise calibration.

### 2.4 λ Computation

```python
# c_d constant from the paper
c_d = 8.0 / (11.0 + 5.0 * math.sqrt(5.0)) if d == 1 else 2.0

# ‖C‖_{1→2} = max column norm = single-participation sensitivity
c1_norm = strategy_first_moment.max_column_norm   # or sensitivity for single-participation
c2_norm = strategy_second_moment.max_column_norm

# λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})
lambda_jme = c1_norm**2 / (c_d * zeta**2 * c2_norm**2)
```

### 2.5 Noise Scaling for Second Moment

The second moment noise is scaled by `λ^{-1/2}` compared to the first moment noise. Since `λ = ‖C₁‖²/(c_d·ζ²·‖C₂‖²)`:

```
λ^{-1/2} = ζ · √c_d · ‖C₂‖_{1→2} / ‖C₁‖_{1→2}
```

So the effective stddev for the second moment stream is:
```
stddev_second = stddev_first * λ^{-1/2}
              = σ_{ε,δ} · s · λ^{-1/2}
              = σ_{ε,δ} · 2ζ · ‖C₁‖_{1→2} · ζ·√c_d·‖C₂‖_{1→2}/‖C₁‖_{1→2}
              = σ_{ε,δ} · 2ζ²·√c_d · ‖C₂‖_{1→2}
```

This is consistent with the utility bound (eq 4 in the paper): `E‖S - Ŝ‖_F = 2ζ²·√c_d·d·σ·‖C₂‖_{1→2}·‖A₂C₂⁻¹‖_F`.

---

## 3. Architecture Overview

### 3.1 New Components (by layer)

```
packages/opaque/src/opaque/noise/mf/
├── jme.py                    # NEW: JmeStrategy dataclass + jme_strategy() factory
│                             #      + _make_jme_noise() internal builder
├── dispatcher.py             # MODIFIED: Add JmeStrategy to MfStrategy union + dispatch
├── __init__.py               # MODIFIED: Export JmeStrategy, jme_strategy
└── _engine.py                # UNCHANGED (reused by JME for both noise streams)

packages/opaque-accounting/src/
├── matrix_factorization/
│   ├── jme.rs                # NEW: JME joint sensitivity, λ computation
│   └── mod.rs                # MODIFIED: Export jme module
├── python/
│   ├── matrix_factorization.rs  # MODIFIED: Add PyO3 bindings for JME functions
│   └── mod.rs                   # MODIFIED: Register new functions

packages/opaque-accounting/opaque_accounting/
├── __init__.py               # MODIFIED: Export jme accounting function
└── mechanisms.py             # MODIFIED: Add jme() mechanism factory
└── _accounting.pyi           # MODIFIED: Add type stubs for new Rust functions

examples/
└── train_dp_ftrl_adam.py     # NEW: DP-Adam training script using JME
```

### 3.2 Component Interaction Diagram

```
                         User Code (train_dp_ftrl_adam.py)
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                   jme_strategy()              acc.jme()
                   (noise/mf/jme.py)         (accounting)
                        │                           │
                ┌───────┴───────┐           ┌───────┴───────┐
                │               │           │               │
         inner_strategy₁  inner_strategy₂   │    Rust: jme_joint_sensitivity()
         (first moment)   (second moment)   │    Rust: jme_lambda()
                │               │           │
                └───────┬───────┘           │
                        │                   │
                   mf_noise()               │
                   (dispatcher.py)          │
                        │                   │
              ┌─────────┴─────────┐        │
              │                   │        │
         noise_fn₁          noise_fn₂     │
         (grad stream)    (grad² stream)   │
              │                   │        │
              └─────────┬─────────┘        │
                        │                  │
                   DP-Adam optimizer   calibrate()
                   (training loop)    (privacy budget)
```

---

## 4. Detailed Implementation Plan

### Phase 1: Rust — JME Sensitivity Functions

**File**: `packages/opaque-accounting/src/matrix_factorization/jme.rs`

#### 4.1.1 `jme_lambda()`
Compute the optimal λ that makes second moment estimation "free":

```rust
pub fn jme_lambda(
    c1_max_col_norm: f64,   // ‖C₁‖_{1→2}
    c2_max_col_norm: f64,   // ‖C₂‖_{1→2}
    zeta: f64,              // clipping bound
    d: usize,               // dimension (for c_d constant)
) -> Result<f64>
```

Returns `λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})`.

#### 4.1.2 `jme_joint_sensitivity()`
Compute joint sensitivity for both moments:

```rust
pub fn jme_joint_sensitivity(
    c1_max_col_norm: f64,   // ‖C₁‖_{1→2}
    zeta: f64,              // clipping bound
) -> f64
```

Returns `s = 2ζ · ‖C₁‖_{1→2}` (Theorem 3.2 of the paper).

#### 4.1.3 `jme_second_moment_noise_scale()`
Compute the noise scaling factor for the second moment stream:

```rust
pub fn jme_second_moment_noise_scale(
    lambda_jme: f64,
) -> f64
```

Returns `λ^{-1/2}`.

#### 4.1.4 Testing
- Verify `jme_lambda` produces correct values for known inputs.
- Verify that with `λ` set optimally, the joint sensitivity equals `2ζ·‖C₁‖_{1→2}`.
- Verify the "privacy for free" property: the noise variance for the first moment remains unchanged.
- Cross-check with the paper's examples.

### Phase 2: Rust — PyO3 Bindings

**File**: `packages/opaque-accounting/src/python/matrix_factorization.rs`

Add Python-exposed functions:
- `jme_lambda(c1_norm, c2_norm, zeta, d) -> float`
- `jme_joint_sensitivity(c1_norm, zeta) -> float`
- `jme_second_moment_noise_scale(lambda_jme) -> float`

**File**: `packages/opaque-accounting/opaque_accounting/_accounting.pyi`

Add type stubs for the new functions.

### Phase 3: Python — JmeStrategy

**File**: `packages/opaque/src/opaque/noise/mf/jme.py`

#### 4.3.1 `JmeStrategy` Dataclass

```python
@dataclass(frozen=True, slots=True)
class JmeStrategy:
    """Joint Moment Estimation strategy for DP-Adam/AdaGrad.

    Wraps two inner MF strategies (for first and second moments) and
    computes the joint sensitivity that makes second moment estimation
    "free" from a privacy perspective.
    """
    sensitivity: float                    # joint sensitivity s = 2ζ·‖C₁‖_{1→2}
    coefficients: tuple[float, ...]       # first moment strategy coefficients
    gram_matrix: tuple[float, ...] | None # first moment Gram matrix (for BnB accounting)

    _first_moment_strategy: MfStrategy    # inner strategy for gradient noise
    _second_moment_strategy: MfStrategy   # inner strategy for squared-gradient noise
    _lambda_jme: float                    # scaling parameter λ
    _zeta: float                          # clipping bound (sensitivity per sample)
    _d: int                               # parameter dimension (for c_d constant)
```

#### 4.3.2 `jme_strategy()` Factory

```python
def jme_strategy(
    first_moment_strategy: MfStrategy,
    second_moment_strategy: MfStrategy | None = None,
    *,
    zeta: float,
    d: int = 2,
) -> JmeStrategy:
    """Create a JME strategy for DP-Adam/AdaGrad.

    Args:
        first_moment_strategy: MF strategy for the gradient (first moment) noise.
            Any existing strategy: band_mf_strategy(), blt_strategy(),
            lambda_cgd_strategy(), bisr_strategy(), identity_strategy().
        second_moment_strategy: MF strategy for the squared-gradient (second
            moment) noise. If None, reuses first_moment_strategy.
        zeta: Clipping bound per sample (= clipping_norm / normalize_by).
            This is clip_state.sensitivity from the clipping module.
        d: Parameter dimension. For d>=2, c_d=2 (default). For d=1,
            c_d = 8/(11+5√5). In practice, always d>=2 for neural networks.

    Returns:
        A JmeStrategy wrapping both inner strategies.
    """
```

Key implementation details:
- If `second_moment_strategy is None`, use the same strategy as `first_moment_strategy`.
- Compute `‖C₁‖_{1→2}` and `‖C₂‖_{1→2}` from the inner strategies' `sensitivity` field (which is max column norm for single participation).
- Call Rust `jme_lambda()` and `jme_joint_sensitivity()`.
- Forward the inner strategies' accounting data (coefficients, gram_matrix) for first moment.

#### 4.3.3 `_make_jme_noise()` Internal Noise Builder

```python
def _make_jme_noise(
    grad_template: Any,
    strategy: JmeStrategy,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, JmeNoiseState], tuple[tuple[Any, Any], JmeNoiseState]],
    JmeNoiseState,
]:
    """Create paired noise functions for JME (first + second moment).

    Returns a noise function that, given clipped gradients and state,
    returns ((noisy_grads, noisy_squared_grads), new_state).
    """
```

Implementation approach:
1. Create two independent MF noise functions using the existing `_matrix_factorization_noise` or `_make_lambda_cgd_noise` engine (dispatching on inner strategy type).
2. The first moment noise function uses `stddev` as-is.
3. The second moment noise function uses `stddev * λ^{-1/2}` as its stddev.
4. Both use independent RNG keys (derived from the provided key via `fold_in`).

**State**: A new `JmeNoiseState` that wraps two `MFNoiseState` instances:

```python
@dataclass(frozen=True)
class JmeNoiseState(NoiseState):
    _first_moment_state: MFNoiseState
    _second_moment_state: MFNoiseState
```

**Noise function signature change**: The JME noise function returns `(noisy_grads, noisy_squared_grads)` rather than just `noisy_grads`. The training loop uses `noisy_grads` for Adam's first moment and `noisy_squared_grads` for Adam's second moment.

### Phase 4: Python — Dispatcher Integration

**File**: `packages/opaque/src/opaque/noise/mf/dispatcher.py`

#### 4.4.1 Update `MfStrategy` Union

```python
MfStrategy = (
    BandMfStrategy | BltStrategy | LambdaCgdStrategy | BisrStrategy
    | IdentityStrategy | JmeStrategy
)
```

#### 4.4.2 Update `mf_noise()` Dispatch

Add a new case for `JmeStrategy` in the `match` statement. Since JME returns a **dual** noise function (both first and second moment), we need a separate entry point.

Two options:
- **Option A**: `mf_noise()` returns the same `(noise_fn, state)` signature, but when called with `JmeStrategy`, `noise_fn` returns `(noisy_grads, noisy_squared_grads)` as the "noisy_grads" output. The training loop destructures.
- **Option B** (recommended): Add a new `jme_noise()` entry point with an explicit dual-output signature, keeping `mf_noise()` backward-compatible for single-stream strategies.

Recommendation: **Option B** — add `jme_noise()` as a separate function:

```python
def jme_noise(
    grad_template: Any,
    strategy: JmeStrategy,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, JmeNoiseState], tuple[tuple[Any, Any], JmeNoiseState]],
    JmeNoiseState,
]:
```

This keeps the existing `mf_noise()` API unchanged and clean. The new `jme_noise()` is used specifically for Adam/AdaGrad training loops.

`mf_noise()` should still accept `JmeStrategy` but raise a helpful error pointing to `jme_noise()`.

### Phase 5: Python — Accounting Integration

#### 4.5.1 New Mechanism Wrapper

Since JME uses the same privacy budget as the first-moment-only strategy (that's the whole point), the accounting for JME simply reuses the existing accounting mechanism for the first moment strategy. No new Rust accounting code is needed beyond the sensitivity/lambda functions.

**Approach**: In `train_dp_ftrl_adam.py`, use the same `acct_mechanism` as the inner first-moment strategy. The joint sensitivity proves this is valid (Theorem 3.2).

### Phase 6: Example Training Script

**File**: `examples/train_dp_ftrl_adam.py`

Fork from `train_dp_ftrl.py` with the following modifications:

#### 4.6.1 Optimizer Change
Replace `torchopt.sgd` with a manual Adam implementation (or `torchopt.adam` with modifications to accept separate noisy first/second moments).

Since standard Adam implementations compute moments internally from gradients, we need a **modified Adam** that accepts pre-computed noisy moments:

```python
class DPAdamState:
    """State for DP-Adam with externally-provided noisy moments."""
    m: dict   # first moment (EMA of noisy gradients)
    v: dict   # second moment (EMA of noisy squared gradients)
    step: int

def dp_adam_update(
    noisy_grads,        # from JME first moment stream
    noisy_sq_grads,     # from JME second moment stream
    state: DPAdamState,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict, DPAdamState]:
```

The key insight: JME provides **noisy** versions of `g_t` and `g_t²` at each step. The Adam EMA is a **linear** operation applied to these noisy estimates. The workload matrices `A₁, A₂` in the JME framework correspond to these EMA operations.

**Two approaches for the training loop**:

**Approach A (Simpler, recommended for initial implementation)**: JME provides correlated noise for both the gradient and squared-gradient streams. The training loop applies Adam's EMA externally:

```python
# Per step:
noisy_grads, noisy_sq_grads, noise_state = jme_noise_fn(clipped_grads, noise_state)

# Adam EMA (these are linear operations on the noisy estimates)
m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, m_state, noisy_grads)
v = tree_map(lambda v, g2: beta2 * v + (1 - beta2) * g2, v_state, noisy_sq_grads)

# Bias correction
m_hat = tree_map(lambda m: m / (1 - beta1**(step+1)), m)
v_hat = tree_map(lambda v: v / (1 - beta2**(step+1)), v)

# Adam update
updates = tree_map(lambda m, v: -lr * m / (torch.sqrt(v) + eps), m_hat, v_hat)
```

**Approach B (Paper-native)**: Build the EMA workloads into the MF mechanism by setting `A₁` and `A₂` as the Adam EMA workload matrices. This is more mathematically elegant but requires deeper integration. Save for a follow-up optimization.

#### 4.6.2 Squared Gradients
Need to compute `g_t ⊗ g_t` (element-wise squaring) of clipped gradients as input to the second moment noise stream:

```python
squared_grads = tree_map(lambda g: g * g, clipped_grads)
```

#### 4.6.3 CLI Arguments
Add:
- `--optimizer adam|sgd` (default: adam)
- `--beta1` (default: 0.9)
- `--beta2` (default: 0.999)
- `--adam-eps` (default: 1e-8)

#### 4.6.4 Presets
- Smoke test: GPT-2 on ag_news with DP-Adam
- Mellum preset: DP-Adam on KStack

---

## 5. Key Design Decisions

### 5.1 Same vs. Different Inner Strategies

The paper allows `C₁ ≠ C₂` (different noise shaping for moments). Our default should be **same strategy for both** (`C₁ = C₂`), which simplifies the API and is the common case. Power users can pass different strategies.

When `C₁ = C₂`: `λ = 1/(c_d · ζ²)`, and `λ^{-1/2} = ζ · √c_d`.

### 5.2 RNG Key Management

The two noise streams must use **independent** RNG keys. Derive them from the user's key:
```python
key_first = fold_in(key, 0)
key_second = fold_in(key, 1)
```

### 5.3 Memory Considerations

For the second moment, `g_t ⊗ g_t` in the paper means the **outer product** `vec(g_t · g_t^T)`, which is `d²`-dimensional. For neural network parameters, this would be prohibitively large.

**Critical optimization**: For Adam, we only need the **diagonal** of the second moment matrix (element-wise squared gradients, not the full outer product). This reduces the second moment from `d²` to `d` dimensions — same as the first moment.

This is valid because Adam only uses `E[g²]` (element-wise), not `E[g·gᵀ]` (full covariance). The JME framework supports this: the workload for Adam's `v_t` is element-wise, so we can apply the noise element-wise.

The sensitivity analysis still holds because `‖g_t ⊙ g_t‖₂ ≤ ‖g_t‖₂² ≤ ζ²` (by Cauchy-Schwarz and the clipping bound). Actually, this bound needs careful analysis — see Section 5.4.

### 5.4 Sensitivity of Squared Gradients

The squared gradient `g_t²` (element-wise) has sensitivity bounded by:
- `‖g_t² - g_t'²‖₂ ≤ ‖g_t‖₂² = ζ²` when one of `g_t, g_t'` is zero (substitute/add-or-remove DP).

The JME paper handles this via the `c_d` constant and the λ parameter. Specifically, for d ≥ 2:
- First moment sensitivity: `‖C₁ · e_i‖₂ · ζ` (contribution of one data point to gradient)
- Second moment sensitivity: `‖C₂ · e_i‖₂ · ζ²` (contribution to squared gradient — scales as `ζ²` not `ζ`)

The joint sensitivity bound `s = 2ζ · ‖C₁‖_{1→2}` is derived by the paper to account for **both** contributions simultaneously.

### 5.5 Noise Correlation Structure

Both noise streams are independently correlated (each has its own `C⁻¹ · Z_i`), but the two streams `Z₁` and `Z₂` are **independent** of each other. This is important for the privacy proof.

### 5.6 Accounting

JME accounting is identical to the first-moment-only accounting — this is the "privacy for free" result. The `acct_mechanism` from the training script remains unchanged; only `noise_multiplier` is calibrated using the **joint sensitivity** instead of the first-moment-only sensitivity.

Since `s_joint = 2ζ · ‖C₁‖_{1→2}` and the first-moment-only sensitivity is `ζ · ‖C₁‖_{1→2}`, the joint sensitivity is exactly 2× the first-moment sensitivity. This means the noise multiplier will be 2× higher to achieve the same ε, but this is fully compensated by the fact that the second moment is now private without any additional budget.

**Important**: The factor of 2 in `s = 2ζ · ‖C₁‖_{1→2}` comes from the substitute-one DP model (replacing one data point). In the add/remove model used by most Opaque accounting, the sensitivity might differ. This needs careful verification during implementation.

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Rust tests** (`packages/opaque-accounting/src/matrix_factorization/jme.rs`):
- `jme_lambda` correctness for known inputs
- `jme_joint_sensitivity` correctness
- Edge cases: `d=1` vs `d≥2`, `C₁ = C₂`, extreme ζ values

**Python tests** (`packages/opaque/tests/matrix_factorization/test_jme.py`):
- `JmeStrategy` creation with all inner strategy types
- Noise function produces two streams of correct shapes
- Noise streams are independent (low correlation between `Z₁` and `Z₂`)
- First moment noise matches what the inner strategy alone would produce
- Second moment noise is scaled by `λ^{-1/2}`
- `JmeNoiseState` updates correctly across steps

### 6.2 Integration Tests

- End-to-end: create `JmeStrategy` → generate noise for 100 steps → verify statistical properties
- Accounting: verify that privacy budget with JME matches first-moment-only budget
- Cross-validation: compare noise variance with naive approach (splitting budget 50/50)

### 6.3 Training Validation

- Smoke test: `train_dp_ftrl_adam.py --preset smoke` runs without errors
- Baseline comparison: DP-Adam via JME vs. DP-SGD on CIFAR-10 (or GPT-2 on ag_news)
- Verify that DP-Adam converges and achieves meaningful loss reduction

---

## 7. Implementation Order

### Step 1: Rust Core (jme.rs)
- Implement `jme_lambda()`, `jme_joint_sensitivity()`, `jme_second_moment_noise_scale()`
- Add tests
- Add PyO3 bindings

### Step 2: Python Strategy (jme.py)
- Implement `JmeStrategy` dataclass
- Implement `jme_strategy()` factory
- Implement `_make_jme_noise()` noise builder
- Implement `JmeNoiseState`

### Step 3: Dispatcher Integration
- Add `JmeStrategy` to `MfStrategy` union
- Add `jme_noise()` entry point
- Update `__init__.py` exports

### Step 4: Training Script
- Fork `train_dp_ftrl.py` → `train_dp_ftrl_adam.py`
- Implement manual DP-Adam optimizer
- Wire up JME noise for both moment streams
- Add CLI arguments and presets

### Step 5: Tests
- Rust unit tests
- Python unit tests
- Integration tests
- Smoke test of training script

### Step 6: Documentation
- Update `docs/mechanisms/` with JME mechanism page
- Update `docs/user-guide/noise.md` and `optimizers.md`
- Add JME to the mechanism list in docs

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Factor-of-2 sensitivity under add/remove DP vs. substitute-one DP | Privacy guarantee could be wrong | Carefully check which DP model Opaque uses; adjust sensitivity accordingly |
| Memory: storing two noise states | 2× memory for noise state buffers | Both streams share the same `StreamingMatrix` structure; overhead is minimal (just buffer state) |
| Second moment noise variance too high for small λ | Utility of second moment estimate could be poor | λ is set optimally by the paper; utility bound is proven. If d is large, this is fine |
| Adam's division by `√v` amplifies noise in second moment | Could hurt convergence | Paper shows improvements on CIFAR-10; bias correction helps. May need learning rate tuning |
| Performance: two noise streams per step | ~2× noise generation time | Noise generation is typically <5% of step time; acceptable overhead |
| Compatibility with microbatching | Microbatch clipping + JME interaction | JME operates on already-clipped gradients; no interaction with microbatching |

---

## 9. Future Extensions

1. **EMA-aware workloads**: Build Adam's EMA directly into the MF workload matrices `A₁, A₂` for optimal noise structure (Approach B from Section 4.6.2).
2. **DP-AdaGrad**: Replace Adam's EMA with cumulative sum for the second moment — maps directly to prefix-sum workload.
3. **Per-group JME**: Extend per-group noise scaling (from `per_group.py`) to JME's dual streams.
4. **DP-LAMB**: Large-batch Adam variant with layer-wise adaptive learning rates.
5. **Benchmark suite**: Systematic comparisons on CIFAR-10, WikiText, and code datasets.
