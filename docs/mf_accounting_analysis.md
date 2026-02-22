# Matrix Factorization Privacy Accounting: Literature vs Implementation Analysis

## 1. Literature Survey (2021-2026)

### 1.1 Foundational Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Kairouz et al. "Practical and Private (Deep) Learning without Sampling or Shuffling" | 2021 | Introduced DP-FTRL with matrix factorization for correlated noise |
| Denisov et al. (arXiv:2202.08312) "Improved DP for SGD via Optimal Private Linear Operators" | 2022 | Parameter-free fixed-point algorithm for optimal dense factorizations |
| Choquette-Choo et al. (arXiv:2211.06530) "Multi-Epoch MF Mechanisms" | 2022 | Fixed-epoch participation pattern (k,b)-epoch-order; multi-epoch sensitivity |
| Choquette-Choo et al. (arXiv:2306.08153) "(Amplified) BandMF" | 2023 | Banded matrices, VecSens algorithm, cyclic Poisson amplification |
| Dvijotham et al. (arXiv:2404.16706) "Efficient and Near-Optimal Noise for Streaming DP" | 2024 | Buffered Linear Toeplitz (BLT) mechanism with O(d) memory |
| Choquette-Choo et al. (arXiv:2405.13763) "Banded Square Root MF" | 2024 | Closed-form sensitivity for decreasing Toeplitz (Theorem 2) |
| McMahan et al. (arXiv:2408.08868) "Don't Use Tree Aggregation, Use BLTs" | 2024 | Multi-participation BLT-DP-FTRL |
| Choquette-Choo et al. (arXiv:2410.06266) "Near Exact Privacy Amplification for Matrix Mechanisms" | 2024 | Monte Carlo accounting + balls-in-bins batching for near-exact privacy params |
| McMahan & Pillutla (arXiv:2504.21413) "BLT Inversion Theorem" | 2025 | BLT family closed under inversion; O(d^3) inverse algorithm; Pillutla score |
| Kalinin et al. (arXiv:2505.12128) "Back to Square Roots" | 2025 | Optimal MF error bound for multi-epoch DP-SGD (BISR) |
| (arXiv:2601.21636) "Sampling-Free Privacy Accounting for Matrix Mechanisms" | 2026 | Deterministic Renyi + conditional composition accountants; no Monte Carlo |
| (arXiv:2602.09338) "Privacy Amplification for BandMF via b-Min-Sep Subsampling" | 2026 | Generalizes Poisson and balls-in-bins for BandMF |

### 1.2 Core Theoretical Framework

**Matrix Mechanism**: Given gradient stream G in R^{n x d}, workload A = tril(ones(n,n)) (prefix sum),
factorize A = BC and release M(G) = B(CG + Z) where Z ~ N(0, sigma^2 I). By post-processing:

- **Sensitivity**: S = ||C||_{1,2} = max_j sqrt(sum_i C_{i,j}^2)  (max L2 column norm)
- **Error**: E[||AG - M(G)||_F^2] = sigma^2 * ||AC^{-1}||_F^2
- **Combined objective**: minimize sens(C)^2 * ||AC^{-1}||_F^2

**Sensitivity under participation patterns**:
- **Single participation**: S = max column norm of C
- **Min-sep participation**: S = sqrt(max_u <diag(X), u>) for banded X = C^T C, via VecSens DP
- **Fixed-epoch (k,b)**: S = sqrt(max_group sum(|X[indices, indices]|))
- **General (Algorithm 4)**: Two-stage upper bound on general Gram matrices

**Privacy accounting**:
- Single Gaussian PLD with effective noise multiplier sigma_eff = sigma / S
- For cyclic/Poisson amplification: compose per-group amplified PLDs

**Key result — BandMF amplification** (Proposition 2.1 of arXiv:2306.08153):
For b-banded C with ||C||_{1,2} <= 1, the mechanism satisfies (eps,delta)-DP when
sigma_BandMF = sigma_SGD(epsilon, delta, k, n/b) — i.e., privacy cost scales with n/b, not n.

**Key result — b-min-sep subsampling** (arXiv:2602.09338):
b-min-sep subsampling Pareto dominates cyclic Poisson for BandMF, providing
approximately b times more examples available for sampling per iteration.

### 1.3 Paper Genealogy

```
Dwork et al. 2010 (continual observation)
    |
Chan et al. 2011 (binary tree mechanism)
    |
Kairouz et al. 2021 (DP-FTRL with tree aggregation)
    |
Denisov et al. 2022 (optimal dense MF, adaptive streams)
    |
    +--- Choquette-Choo et al. 2023a (multi-epoch MF, VecSens)
    |       |
    |       +--- Choquette-Choo et al. 2023b (BandMF + amplification)
    |       |       |
    |       |       +--- McKenna 2024 (scaling BandMF)
    |       |       +--- Dong et al. 2025 (b-min-sep subsampling)
    |       |
    |       +--- Kalinin & Lampert 2024 (BSR)
    |               +--- Kalinin et al. 2025 (BISR, optimal bounds)
    |
    +--- Dvijotham et al. 2024 (BLT mechanism)
            |
            +--- McMahan et al. 2024 (BLT-DP-FTRL for practice)
            +--- McMahan & Pillutla 2025 (BLT inversion theorem)
```

---

## 2. Implementation Inventory

### 2.1 Python Noise Layer (`src/opaque/noise/matrix_factorization/`)

| Module | What It Does | Theory Reference |
|--------|-------------|-----------------|
| `noise.py` | Core noise addition via dense or streaming C^{-1} | Kairouz 2021 |
| `toeplitz.py` | Toeplitz strategy: materialize, inverse, error, optimize | BandMF (2306.08153) |
| `buffered_toeplitz.py` | BLT: parameterized Toeplitz with buffers | BLT (2404.16706) |
| `banded.py` | Column-normalized banded matrices | BandMF (2306.08153) |
| `dense.py` | General dense strategy optimization | Denisov 2022 |
| `sensitivity.py` | VecSens, banded, fixed-epoch, general UB | BandMF Algorithms 3-4 |
| `streaming_matrix.py` | Memory-efficient matrix multiplication | BLT (2404.16706) |
| `optimization.py` | L-BFGS wrapper for strategy optimization | Multiple papers |

### 2.2 Rust Accounting (`crates/dp-accounting/src/matrix_factorization/`)

| Module | What It Does | Theory Reference |
|--------|-------------|-----------------|
| `sensitivity.rs` | VecSens, banded, general UB, fixed-epoch sensitivity | BandMF (2306.08153) |
| `mf_gaussian.rs` | MF Gaussian PLD: single Gaussian with sigma_eff = sigma/S | Core MF privacy reduction |

### 2.3 Python Accounting (`src/opaque/accounting/`)

| Module | What It Does |
|--------|-------------|
| `mechanisms/mf_gaussian.py` | `MfGaussian` DpProcess wrapping Rust `mf_gaussian_pld` |

---

## 3. Correctness Assessment

### 3.1 CORRECT: Noise Generation

| Component | Status | Notes |
|-----------|--------|-------|
| Dense MF noise via C^{-1} rows | CORRECT | Linear combination of Gaussians correctly produces correlated noise |
| Streaming MF noise via StreamingMatrix | CORRECT | Correctly applies C^{-1} in streaming fashion |
| Toeplitz inverse (Algorithm 9) | CORRECT | Recurrence xi = (yi - coef[1:] @ state) / coef[0] matches paper |
| BLT streaming inverse | CORRECT | xi = yi - read(state), state = update(state, xi) matches Algorithm 3 of 2408.08868 |
| RNG determinism in distributed mode | CORRECT | fold_in(base_key, step_counter) ensures reproducible per-step noise |

### 3.2 CORRECT: Sensitivity Computation (Python)

| Function | Status | Notes |
|----------|--------|-------|
| `single_participation_sensitivity` | CORRECT | max column norm of C |
| `max_participation_for_linear_fn` (VecSens) | CORRECT | DP matches Algorithm 3, O(n * max_participations) |
| `get_sensitivity_banded_for_X` | CORRECT | sqrt(VecSens on diag(X)) for min_sep-banded X |
| `get_min_sep_sensitivity_upper_bound_for_X` (Algorithm 4) | CORRECT | Two-stage optimization |
| `fixed_epoch_sensitivity_for_X` | CORRECT | Submatrix absolute sum approach |
| `toeplitz.minsep_sensitivity_squared` | CORRECT | Matches Theorem 2 of BSR paper (2405.13763) |
| `buffered_toeplitz.sensitivity_squared` | CORRECT | Matches Lemma 5.3 of BLT paper (2404.16706) |
| `banded.minsep_sensitivity_squared` | CORRECT | For column-normalized: sensitivity^2 = max_participations when min_sep >= bands |

### 3.3 CORRECT: Sensitivity Computation (Rust)

| Function | Status | Notes |
|----------|--------|-------|
| `max_participation_for_linear_fn` | CORRECT | Matches Python VecSens; tested against known optima |
| `single_participation_sensitivity` | CORRECT | max of pre-computed column norms |
| `banded_sensitivity` | CORRECT | sqrt(VecSens on gram_diag) |
| `general_sensitivity_upper_bound` | CORRECT | Algorithm 4 two-stage |
| `fixed_epoch_sensitivity` | CORRECT | Submatrix sum; group indices match Python |

### 3.4 CORRECT: Privacy Accounting

| Component | Status | Notes |
|-----------|--------|-------|
| `mf_gaussian_pld` (Rust) | CORRECT | Single Gaussian with sigma_eff = sigma/S; validated against analytical delta formula |
| `MfGaussian.pld()` (Python) | CORRECT | Wraps Rust correctly |
| Consistency: mf_gaussian at S=1 matches standard gaussian | VERIFIED | Test confirms < 1e-6 difference |
| Monotonicity: higher S -> higher epsilon | VERIFIED | Tested |
| Equivalence: same sigma/S -> same PLD | VERIFIED | Tested |

---

## 4. Discrepancies, Wrong Assumptions, and Missing Features

### 4.1 DISCREPANCY: BandMF Cyclic Amplification Not Encapsulated

**Theory** (BandMF paper, Section 5): When BandMF uses cyclic Poisson batching with band width b:
- Divide n rounds into k = ceil(n/b) groups
- Each group is an independent Poisson-subsampled Gaussian mechanism
- Per-group: noise_multiplier = sigma/kappa, sampling_rate = b*batch_size/dataset_size
- Total privacy: compose k group PLDs

**Implementation**: No single function encapsulates this. Users must manually:
1. Compute sensitivity kappa from their strategy
2. Call `poisson_gaussian_pld(sigma/kappa, rate).self_compose(k)`

**Impact**: Medium. Works correctly when done manually, but error-prone and undocumented.

### 4.2 DISCREPANCY: Column Normalization Creates Two Sensitivity Regimes

**Theory**: BandMF paper distinguishes:
- Raw banded Toeplitz: sensitivity from VecSens on Gram diagonal
- Column-normalized banded: sensitivity^2 = max_participations (constant, simpler)

**Implementation**:
- Python `banded.py` uses column-normalized representation -> `minsep_sensitivity_squared = max_participations`
- Rust `banded_sensitivity()` takes raw Gram diagonal -> VecSens approach
- Python `toeplitz.py` uses raw Toeplitz coefficients

**Impact**: Low. Both are correct for their respective representations, but users must know which representation they're using.

### 4.3 MISSING: BLT Sensitivity in Rust

**Theory** (Lemma 5.3, BLT paper): BLT sensitivity has a closed-form involving geometric sums:
```
sensitivity^2(C) = 1 + sum_ij omega_i * omega_j * geometric_sum(1, theta_i * theta_j, n-1)
```

**Implementation**: Only in Python (`buffered_toeplitz.sensitivity_squared`). Not in Rust.

**Impact**: Medium. Users must compute BLT sensitivity in Python and pass to Rust for PLD. Prevents fully Rust-side BLT accounting or calibration.

### 4.4 MISSING: Toeplitz Min-Sep Sensitivity in Rust

**Theory** (Theorem 2, BSR paper): Closed-form sensitivity for decreasing Toeplitz:
```
sens^2 = ||sum_{j=0}^{k-1} M[., 1+jb]||^2
```

**Implementation**: Only in Python (`toeplitz.minsep_sensitivity_squared`). Not in Rust.

**Impact**: Low-Medium. Same issue as 4.3.

### 4.5 MISSING: Privacy Amplification for Matrix Mechanisms

**Theory** (arXiv:2410.06266, arXiv:2601.21636, arXiv:2602.09338): Recent papers provide:
- Monte Carlo accounting for near-exact privacy params under batching
- Sampling-free Renyi + conditional composition accountants
- b-min-sep subsampling for tighter BandMF amplification

**Implementation**: None of these are implemented. The closest is composing `poisson_gaussian_pld` manually.

**Impact**: High. These are the state-of-the-art for BandMF privacy analysis. The current approach of treating the entire MF run as a single Gaussian mechanism is correct but conservative for the amplified case.

### 4.6 MISSING: Workload-Dependent Privacy Analysis

**Theory**: The matrix mechanism's privacy depends on the workload A:
- Prefix sum: A = tril(ones(n,n))
- Momentum SGD: A encodes learning rate schedule + momentum
- General: arbitrary linear workload

**Implementation**: Python optimization supports workload-dependent error computation (e.g., `momentum_sgd_matrix`), but privacy accounting always assumes prefix sum sensitivity. The Rust side has no workload awareness.

**Impact**: Low. Prefix sum is the standard workload and the most common in practice. But using a non-prefix workload could theoretically change the sensitivity analysis.

### 4.7 WRONG ASSUMPTION: Effective Noise Multiplier Range

**Implementation** (Rust `mf_gaussian.rs`):
```rust
const MF_MIN_EFFECTIVE_NOISE_MULTIPLIER: f64 = 0.01;
const MF_MAX_EFFECTIVE_NOISE_MULTIPLIER: f64 = 1000.0;
```

**Issue**: While these are practical bounds, they are hardcoded constants rather than derived from theory. For extreme MF configurations:
- Very high sensitivity (many participations, long training) could push sigma_eff below 0.01
- Users get a confusing error rather than a (valid but imprecise) PLD

**Impact**: Low. These edge cases are rare in practice. The error message is informative.

### 4.8 GAP: No End-to-End MF Accounting API

**Theory**: A complete MF accounting pipeline should be:
1. Choose strategy (BandMF/BLT/Dense) + participation pattern
2. Compute sensitivity
3. Compute PLD
4. Query epsilon/delta

**Implementation**: Steps 1-2 are in Python, step 3 is in Rust (via Python binding), step 4 is in Python. There is no single function like:
```python
acc.band_mf_accounting(n=1000, bands=5, min_sep=5, noise_multiplier=1.0)
```

**Impact**: Medium. Users must understand the full pipeline and manually connect pieces.

### 4.9 GAP: No Calibration Support for MF Mechanisms

**Theory**: Calibration = "find noise_multiplier for target (epsilon, delta) budget given a strategy."

**Implementation**: The generic `calibrate()` function works with any `DpProcess`, so `MfGaussian` can be calibrated. However:
- Calibration requires a fixed sensitivity, which depends on the strategy
- There's no `mf_calibrate(strategy, target_budget)` that jointly considers strategy and noise

**Impact**: Low. Users can use `calibrate(lambda sigma: acc.mf_gaussian(sigma, precomputed_sensitivity))`.

### 4.10 POTENTIAL ISSUE: Numerical Stability of BLT Inverse

**Theory** (arXiv:2504.21413, Theorem 1): BLT^{-1} is itself a BLT with explicit inverse formula.
The inverse decay parameters hat_lambda depend on sum(alpha_i / lambda_i) (Pillutla score):
- If Pillutla score < 1: all hat_lambda_i in (0,1) — well-behaved
- If Pillutla score > 1: one hat_lambda_j in (-1,0) — oscillatory inverse
- If Pillutla score = 1: degenerate case

**Implementation** (`buffered_toeplitz.py:inverse()`):
- Uses `torch.linalg.eigvals` for eigenvalues (Lemma 5.2 of BLT paper)
- Closed-form eigenvectors from `omega / (evals - buf_decay)`
- Gap check: `min_buf_decay_gap >= 1e-9`
- Verification: reconstructed Theta2 matches within `atol=1e-7`
- Pillutla score constraint enforced during optimization (penalty for score >= 1)

**Issue**: For buf_decay values very close together (but > 1e-9 gap), the eigenvector computation
involves dividing by small differences, amplifying floating-point errors. The paper (arXiv:2504.21413)
provides the more stable explicit formula (Theorem 2) using interlaced decay parameters, which the
implementation does NOT use — instead using the eigendecomposition approach.

**Impact**: Low-Medium. The verification check catches failures, but the newer O(d^3) algorithm from
arXiv:2504.21413 could be more numerically stable for difficult configurations.

### 4.11 MISSING: Recent Advances Not Incorporated

| Paper | Feature | Status |
|-------|---------|--------|
| arXiv:2505.12128 (BISR) | Optimal MF error bound for multi-epoch | NOT IMPLEMENTED |
| arXiv:2601.21636 | Sampling-free Renyi accounting for banded matrices | NOT IMPLEMENTED |
| arXiv:2602.09338 | b-min-sep subsampling amplification | NOT IMPLEMENTED |
| arXiv:2511.17994 | Learning rate scheduling with MF | NOT IMPLEMENTED |
| arXiv:2601.22334 | DP-lambda-CGD efficient noise correlation | NOT IMPLEMENTED |

---

## 5. Improvement and Correctness Plan

### Priority 1: High Impact, Correctness-Critical

#### P1.1: Add BandMF Amplified Accounting (Cyclic Poisson)

**Goal**: Provide a dedicated function for BandMF with cyclic/Poisson amplification.

**Implementation**:
```python
# Python-side (composition is Python's responsibility):
def band_mf_amplified(
    noise_multiplier: float,
    sensitivity: float,
    sample_rate: float,
    num_groups: int,
) -> DpProcess:
    """BandMF with cyclic Poisson amplification.

    Composes num_groups Poisson-subsampled Gaussian mechanisms.
    """
    step = poisson(gaussian(noise_multiplier / sensitivity), sample_rate=sample_rate)
    return step * num_groups
```

**Files to modify**:
- `src/opaque/accounting/mechanisms/__init__.py`: add export
- New file or addition to `mf_gaussian.py`: implement `band_mf_amplified()`
- `src/opaque/accounting/__init__.py`: add to public API

**Tests**: Verify that for bands=1 (diagonal strategy), the result matches standard `poisson_gaussian * n`.

#### P1.2: Document the End-to-End MF Accounting Pipeline

**Goal**: Add comprehensive documentation connecting noise strategies to their accounting.

**Implementation**: Add a `docs/mf_accounting_guide.md` or docstring-level examples showing:
- BandMF single-participation: `mf_gaussian(sigma, sensitivity)`
- BandMF min-sep: `mf_gaussian(sigma, banded_sensitivity(gram_diag, min_sep))`
- BandMF amplified: `band_mf_amplified(sigma, sensitivity, rate, k)`
- BLT: `mf_gaussian(sigma, blt_sensitivity_squared(blt, n).sqrt())`
- Dense: `mf_gaussian(sigma, single_participation_sensitivity(column_norms))`

### Priority 2: Medium Impact, Feature Gaps

#### P2.1: Add BLT Sensitivity to Rust

**Goal**: Port the closed-form BLT sensitivity (Lemma 5.3) to Rust.

**Implementation**:
- Add `blt_sensitivity()` function to `matrix_factorization/sensitivity.rs`
- Takes buf_decay, output_scale, n parameters
- Implements geometric_sum with Taylor series near r=1
- Add PyO3 binding

**Tests**: Compare against Python implementation for various BLT configurations.

#### P2.2: Add Toeplitz Min-Sep Sensitivity to Rust

**Goal**: Port BSR Theorem 2 closed-form to Rust.

**Implementation**:
- Add `toeplitz_minsep_sensitivity()` to `matrix_factorization/sensitivity.rs`
- Takes Toeplitz coefficients, min_sep, max_participations
- Uses cumsum-based restructuring
- Add PyO3 binding

#### P2.3: Add Sampling-Free Privacy Accounting for Banded Matrices

**Goal**: Implement the deterministic Renyi accountant from arXiv:2601.21636.

**Scope**: This is a larger effort. Start with the conditional composition approach for banded matrices, which provides tighter bounds in high-privacy regimes.

**Implementation**:
- New Rust module `matrix_factorization/amplified_accounting.rs`
- Renyi divergence computation for banded Gram structures
- Integration with existing PLD framework

### Priority 3: Lower Impact, Nice-to-Have

#### P3.1: Add End-to-End MF Accounting Convenience Functions

```python
def band_mf_accounting(
    n: int,
    bands: int,
    min_sep: int,
    noise_multiplier: float,
    strategy_coef: torch.Tensor | None = None,
) -> DpProcess:
    """Complete BandMF accounting from strategy parameters."""
    if strategy_coef is None:
        strategy_coef = toeplitz.optimal_max_error_strategy_coefs(bands)
    gram_diag = toeplitz.sensitivity_squared(strategy_coef)  # diagonal elements
    sensitivity = banded_sensitivity(gram_diag, min_sep)
    return mf_gaussian(noise_multiplier, sensitivity)
```

#### P3.2: Add MF-Specific Calibration Helper

```python
def calibrate_mf(
    strategy_sensitivity: float,
    budget: Budget,
    **calibration_kwargs,
) -> float:
    """Find noise_multiplier for target privacy budget given MF sensitivity."""
    return calibrate(
        lambda sigma: mf_gaussian(sigma, strategy_sensitivity),
        budget,
        **calibration_kwargs,
    )
```

#### P3.3: Widen Effective Noise Multiplier Range or Make Configurable

Replace hardcoded `[0.01, 1000]` range with configurable bounds or automatically adapt based on discretization config.

#### P3.4: Add BISR Strategy Support

Port the Banded Inverse Square Root (BISR) strategy from arXiv:2505.12128, which achieves optimal MF error for multi-epoch training. This would be a new strategy type alongside Toeplitz, BLT, and Dense.

---

## 6. Summary

### What's Working Well
- **Noise generation pipeline** (Python): Comprehensive, correct, numerically stable
- **Sensitivity computation** (Python): Complete coverage of all published algorithms
- **Sensitivity computation** (Rust): Correctly ports core algorithms (VecSens, banded, general UB, fixed-epoch)
- **Basic MF accounting** (Rust+Python): `mf_gaussian_pld` correctly reduces MF privacy to single Gaussian
- **BLT infrastructure**: Full implementation with inverse, streaming, optimization, numerical stability

### What Needs Improvement
- **Amplified accounting**: BandMF with batching is the most common production use case but has no dedicated accounting function
- **End-to-end API**: Users must manually connect strategy -> sensitivity -> PLD
- **Recent advances**: 2025-2026 papers (sampling-free accounting, b-min-sep subsampling, BISR) are not yet incorporated
- **Rust completeness**: BLT/Toeplitz-specific sensitivity functions only in Python

### No Correctness Bugs Found
All implemented algorithms match their paper references. The only issues are:
- Missing features (not wrong, just incomplete)
- Conservative bounds (single Gaussian instead of amplified accounting)
- Interface gaps (manual sensitivity computation required)
