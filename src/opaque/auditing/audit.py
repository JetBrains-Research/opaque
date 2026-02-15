"""Privacy audit results from canary membership inference scores.

Construct an :class:`AuditResult` from held-in and held-out canary scores,
then query metrics as methods. Shared state (TN/FN counts on the Pareto
frontier) is precomputed once at construction.

For end-to-end auditing with a single training run, use
:class:`CoinFlipExperiment` to manage canary inclusion/exclusion, then
call :meth:`CoinFlipExperiment.audit` to produce an :class:`AuditResult`.

Example — post-hoc (scores already computed):
    >>> result = AuditResult(in_scores, out_scores)
    >>> print(f"Epsilon: {result.epsilon_clopper_pearson():.2f}")

Example — one-run auditing (Steinke et al. 2023):
    >>> experiment = CoinFlipExperiment(num_canaries=1000, seed=42)
    >>> # train model on data that includes experiment.in_indices canaries
    >>> # and excludes experiment.out_indices canaries ...
    >>> scores = compute_membership_scores(model, canary_data)
    >>> result = experiment.audit(scores)
    >>> print(f"Epsilon: {result.epsilon_one_run():.2f}")

References:
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.stats

from opaque.auditing.bootstrap import BootstrapParams
from opaque.auditing.helpers import (
    _clopper_pearson_upper,
    _epsilon_raw_counts_helper,
    _get_tn_fn_counts,
    _one_run_p_value,
    _tpr_at_given_fpr,
)

__all__ = ["AuditResult", "CoinFlipExperiment"]


class AuditResult:
    """Privacy audit results computed from canary scores.

    Construct from held-in (training) and held-out (test) canary scores.
    Precomputes shared TN/FN counts on the Pareto frontier once, then
    all metric queries reuse this state.

    This is read-only: scores go in at construction, methods only read.

    Attributes:
        n_in: Number of held-in canaries.
        n_out: Number of held-out canaries.

    Example:
        >>> result = AuditResult(in_scores, out_scores)
        >>> result.epsilon_clopper_pearson(significance=0.05, delta=1e-5)
        >>> result.auroc()
        >>> result.tpr_at_fpr(fpr=0.01)
    """

    def __init__(self, in_scores: np.ndarray, out_scores: np.ndarray) -> None:
        in_arr = np.asarray(in_scores, dtype=float)
        out_arr = np.asarray(out_scores, dtype=float)

        if in_arr.size == 0 or out_arr.size == 0:
            raise ValueError("Both in_scores and out_scores must be non-empty")

        self._in_arr = in_arr
        self._out_arr = out_arr
        self.n_in = len(in_arr)
        self.n_out = len(out_arr)

        # Precompute Pareto-optimal thresholds and counts
        thresholds, tn_counts, fn_counts = _get_tn_fn_counts(in_arr, out_arr)
        self._thresholds = thresholds
        self._tn_counts = tn_counts
        self._fn_counts = fn_counts

        # TP/FP counts (reversed for ROC perspective)
        self._tp_counts = (fn_counts[-1] - fn_counts)[::-1]
        self._fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    # ------------------------------------------------------------------
    # Epsilon estimation
    # ------------------------------------------------------------------

    def epsilon_clopper_pearson(
        self,
        *,
        significance: float = 0.05,
        delta: float = 0.0,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound using Clopper-Pearson confidence intervals.

        Constructs conservative binomial confidence intervals for TPR/FPR.
        Uses Bonferroni correction over Pareto-optimal thresholds unless an
        explicit threshold is provided.

        Args:
            significance: Allowed failure probability (1 - confidence).
            delta: DP delta parameter. Default: 0 (pure DP).
            threshold: If provided, use this specific threshold instead of
                searching with Bonferroni correction.

        Returns:
            Epsilon lower bound at the specified confidence level.
        """
        _validate_significance(significance)
        _validate_delta(delta)

        if threshold is not None:
            fn = int(np.sum(self._in_arr < threshold))
            fp = int(np.sum(self._out_arr >= threshold))
            return _epsilon_cp(fn, fp, self.n_in, self.n_out, significance, delta)

        # Bonferroni correction over all Pareto-optimal thresholds
        sig_corrected = significance / len(self._thresholds)
        best = 0.0
        for i in range(len(self._thresholds)):
            fn_i = self._fn_counts[i]
            fp_i = self.n_out - self._tn_counts[i]
            eps_i = _epsilon_cp(
                fn_i, fp_i, self.n_in, self.n_out, sig_corrected, delta
            )
            best = max(best, eps_i)
        return best

    def epsilon_one_run(
        self,
        *,
        significance: float = 0.05,
        delta: float = 0.0,
        threshold: float | None = None,
        eps_max: float = 20.0,
        tol: float = 1e-4,
    ) -> float:
        """Epsilon lower bound using the one-run method from Nasr et al. (2023).

        Uses a likelihood-ratio test tailored for DP auditing. Generally less
        conservative than Clopper-Pearson for the same sample size.

        Args:
            significance: Allowed failure probability (1 - confidence).
            delta: DP delta parameter. Default: 0 (pure DP).
            threshold: If provided, use this specific threshold.
            eps_max: Maximum epsilon to search. Default: 20.0.
            tol: Binary search tolerance. Default: 1e-4.

        Returns:
            Epsilon lower bound at the specified confidence level.

        Reference:
            Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
        """
        _validate_significance(significance)
        _validate_delta(delta)

        m = self.n_in + self.n_out

        if threshold is not None:
            tp = int(np.sum(self._in_arr >= threshold))
            fp = int(np.sum(self._out_arr >= threshold))
            return _epsilon_one_run_search(
                tp + fp, tp, m, significance, delta, eps_max, tol
            )

        # Bonferroni correction over all Pareto-optimal thresholds
        sig_corrected = significance / len(self._thresholds)
        best = 0.0
        for i in range(len(self._thresholds)):
            tp_i = self.n_in - self._fn_counts[i]
            fp_i = self.n_out - self._tn_counts[i]
            eps_i = _epsilon_one_run_search(
                tp_i + fp_i, tp_i, m, sig_corrected, delta, eps_max, tol
            )
            best = max(best, eps_i)
        return best

    def epsilon_raw_counts(
        self,
        *,
        min_count: int = 50,
        delta: float = 0.0,
    ) -> float:
        """Epsilon estimate from raw TPR/FPR counts.

        Direct computation without confidence intervals. Less conservative
        but higher variance than Clopper-Pearson.

        Args:
            min_count: Minimum FP count to consider a threshold. Default: 50.
            delta: DP delta parameter. Default: 0 (pure DP).

        Returns:
            Epsilon estimate.
        """
        if min_count < 1:
            raise ValueError(f"min_count must be positive, got {min_count}")
        _validate_delta(delta)

        return max(
            0.0,
            _epsilon_raw_counts_helper(
                self._tp_counts, self._fp_counts, min_count, delta
            ),
        )

    # ------------------------------------------------------------------
    # Attack utility metrics
    # ------------------------------------------------------------------

    def auroc(self) -> float:
        """Area under the ROC curve for the membership inference attack.

        AUROC = 0.5 means random guessing, AUROC = 1.0 means perfect attack.

        Returns:
            AUROC value in [0, 1].
        """
        tnr = self._tn_counts / self._tn_counts[-1]
        fnr = self._fn_counts / self._fn_counts[-1]
        return float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))

    def tpr_at_fpr(self, *, fpr: float | np.ndarray) -> float | np.ndarray:
        """True positive rate at a given false positive rate.

        Args:
            fpr: Target false positive rate(s) in [0, 1].

        Returns:
            TPR value(s) at the specified FPR(s).
        """
        return _tpr_at_given_fpr(fpr, self._tp_counts, self._fp_counts)

    def max_accuracy(self, *, prevalence: float | None = None) -> float:
        """Maximum classification accuracy achievable.

        Args:
            prevalence: Fraction of positives in population.
                Default: use sample ratio.

        Returns:
            Maximum accuracy across all thresholds.
        """
        if prevalence is not None and not 0.0 <= prevalence <= 1.0:
            raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")

        n_pos = self._fn_counts[-1]
        n_neg = self._tn_counts[-1]

        if prevalence is None:
            prevalence = n_pos / (n_pos + n_neg)

        tp_counts = n_pos - self._fn_counts
        tnr = self._tn_counts / n_neg
        tpr = tp_counts / n_pos

        return float(np.max(tpr * prevalence + tnr * (1 - prevalence)))

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        metric: Callable[[AuditResult], float],
        params: BootstrapParams,
    ) -> np.ndarray:
        """Bootstrap confidence intervals for any metric.

        Resamples scores with replacement and computes the metric on each
        resample. Supports bias-corrected and accelerated (BCa) intervals.

        Args:
            metric: Function that takes an AuditResult and returns a float.
                Can be an unbound method (e.g. ``AuditResult.auroc``).
            params: Bootstrap parameters (num_samples, quantiles, etc.).

        Returns:
            Array of quantiles specified in params.quantiles.

        Example:
            >>> result = AuditResult(in_scores, out_scores)
            >>> params = BootstrapParams(num_samples=1000, seed=42)
            >>> auroc_ci = result.bootstrap(AuditResult.auroc, params)
            >>> eps_ci = result.bootstrap(
            ...     lambda r: r.epsilon_clopper_pearson(significance=0.05),
            ...     params,
            ... )
        """
        rng = np.random.default_rng(seed=params.seed)

        values = np.empty(params.num_samples)
        for i in range(params.num_samples):
            in_sample = rng.choice(self._in_arr, size=self.n_in)
            out_sample = rng.choice(self._out_arr, size=self.n_out)
            values[i] = metric(AuditResult(in_sample, out_sample))

        if not params.bias_correction:
            return np.quantile(values, params.quantiles, method="linear")

        # Bias-corrected bootstrap (BCa)
        full_estimate = metric(self)
        prop_less = (np.sum(values < full_estimate) + 1) / (params.num_samples + 2)
        z0 = scipy.stats.norm.ppf(prop_less)

        if params.acceleration:
            # Jackknife for acceleration
            jk = np.empty(self.n_in + self.n_out)
            for i in range(self.n_in):
                jk[i] = metric(
                    AuditResult(np.delete(self._in_arr, i), self._out_arr)
                )
            for i in range(self.n_out):
                jk[self.n_in + i] = metric(
                    AuditResult(self._in_arr, np.delete(self._out_arr, i))
                )

            jk_mean = np.mean(jk)
            num = np.sum((jk_mean - jk) ** 3)
            denom = 6 * np.sum((jk_mean - jk) ** 2) ** 1.5
            accel = 0.0 if denom == 0 else num / denom
        else:
            accel = 0.0

        z_q = scipy.stats.norm.ppf(params.quantiles)
        corrected = scipy.stats.norm.cdf(
            z0 + (z0 + z_q) / (1 - accel * (z0 + z_q))
        )

        return np.quantile(values, corrected, method="linear")


class CoinFlipExperiment:
    """One-run privacy audit: manages canary coin flips and score splitting.

    Implements the canary setup from Steinke, Nasr, Jagielski (2023):
    each canary is independently included or excluded from training with
    probability 0.5 (a fair coin flip). After training, the user computes
    a membership score for each canary and calls :meth:`audit` to get an
    :class:`AuditResult`.

    This bridges training and auditing — the two pieces that were previously
    disconnected. The user provides canary indices into their dataset; the
    experiment decides which are in/out and provides index arrays for
    constructing the training set.

    Attributes:
        num_canaries: Total number of canary examples.
        in_indices: Canary indices included in training (coin = heads).
        out_indices: Canary indices excluded from training (coin = tails).

    Example:
        >>> # 1. Setup: pick 1000 canaries from a 50k dataset
        >>> canary_idx = rng.choice(50000, size=1000, replace=False)
        >>> experiment = CoinFlipExperiment(canary_idx, seed=42)
        >>>
        >>> # 2. Build training set: full dataset minus excluded canaries
        >>> train_idx = sorted(set(range(50000)) - set(experiment.out_indices))
        >>> model = train(dataset, train_idx, ...)
        >>>
        >>> # 3. Score all canaries (higher = more likely member)
        >>> scores = -np.array([loss(model, dataset[i]) for i in canary_idx])
        >>>
        >>> # 4. Audit
        >>> result = experiment.audit(scores)
        >>> print(f"Epsilon: {result.epsilon_one_run(significance=0.05):.2f}")
        >>> print(f"AUROC: {result.auroc():.3f}")

    Reference:
        Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training
        Run." NeurIPS 2023. https://arxiv.org/abs/2305.08846
    """

    def __init__(
        self,
        canary_indices: np.ndarray,
        *,
        seed: int | None = None,
    ) -> None:
        """Flip coins for each canary to decide inclusion/exclusion.

        Args:
            canary_indices: Array of dataset indices designated as canaries.
            seed: Random seed for reproducible coin flips.
        """
        canary_indices = np.asarray(canary_indices)
        if canary_indices.ndim != 1 or canary_indices.size == 0:
            raise ValueError("canary_indices must be a non-empty 1-D array")

        rng = np.random.default_rng(seed)
        in_mask = rng.random(len(canary_indices)) < 0.5

        self.num_canaries = len(canary_indices)
        self._canary_indices = canary_indices
        self._in_mask = in_mask
        self.in_indices = canary_indices[in_mask]
        self.out_indices = canary_indices[~in_mask]

    def train_indices(self, dataset_size: int) -> np.ndarray:
        """Dataset indices to use for training.

        Returns all indices in ``range(dataset_size)`` except the excluded
        canaries. Convenience method so users don't have to do set arithmetic.

        Args:
            dataset_size: Total number of examples in the full dataset.

        Returns:
            Sorted array of training indices.
        """
        excluded = set(self.out_indices.tolist())
        return np.array([i for i in range(dataset_size) if i not in excluded])

    def audit(self, scores: np.ndarray) -> AuditResult:
        """Split scores by coin flip and return an AuditResult.

        Args:
            scores: Membership score for each canary, in the same order as
                ``canary_indices`` passed to the constructor. Shape
                ``(num_canaries,)``. Higher scores should indicate higher
                likelihood of being a training member.

        Returns:
            :class:`AuditResult` with in/out scores split by coin flip.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.num_canaries,):
            raise ValueError(
                f"Expected {self.num_canaries} scores, got shape {scores.shape}"
            )
        return AuditResult(scores[self._in_mask], scores[~self._in_mask])


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _validate_significance(significance: float) -> None:
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")


def _validate_delta(delta: float) -> None:
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")


def _epsilon_cp(
    fn: int,
    fp: int,
    n_in: int,
    n_out: int,
    significance: float,
    delta: float,
) -> float:
    """Clopper-Pearson epsilon at given FN/FP counts."""
    fnr_ub = _clopper_pearson_upper(fn, n_in, significance / 2)
    fpr_ub = _clopper_pearson_upper(fp, n_out, significance / 2)

    tpr_lb = 1 - fnr_ub
    if tpr_lb <= delta:
        return 0.0

    return max(0.0, float(np.log(tpr_lb - delta) - np.log(fpr_ub)))


def _epsilon_one_run_search(
    n_guess: int,
    n_correct: int,
    m: int,
    significance: float,
    delta: float,
    eps_max: float,
    tol: float,
) -> float:
    """One-run epsilon via binary search."""
    if n_guess == 0 or n_correct == 0:
        return 0.0

    eps_lo, eps_hi = 0.0, eps_max
    while eps_hi - eps_lo > tol:
        eps_mid = (eps_lo + eps_hi) / 2
        p_val = _one_run_p_value(m, n_guess, n_correct, eps_mid, delta)
        if p_val < significance:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_lo
