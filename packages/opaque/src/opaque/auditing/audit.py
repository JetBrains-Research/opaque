"""Privacy audit results from canary membership inference scores.

Construct an :class:`AuditResult` from held-in and held-out canary scores,
then query metrics as methods. Shared state (TN/FN counts on the Pareto
frontier) is precomputed once at construction.

For end-to-end auditing with a single training run, use
:func:`opaque.auditing.setup` and :func:`opaque.auditing.evaluate`::

    import opaque.auditing as auditing
    from opaque.random import key

    audit_state = auditing.setup(
        dataset, num_canaries=1000, key=key(42),
        batch_argnums=(1,),
    )
    train_data = dataset.select(audit_state.train_indices)
    # ... train model with DP-SGD ...
    audit = auditing.evaluate(loss_fn, params, state=audit_state)
    print(audit.summary(delta=1e-5))

Or use the two-step API for more control::

    coin_flip = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
    estimator = auditing.one_run(coin_flip, dataset=dataset, batch_argnums=(1,))
    train_data = dataset.select(estimator.train_indices)
    # ... train model with DP-SGD ...
    audit = auditing.evaluate(loss_fn, params, state=estimator)

References:
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

import numpy as np
import scipy.stats

from opaque.auditing.helpers import (
    _get_tn_fn_counts,
    _one_run_p_value,
    _tpr_at_given_fpr,
)
from opaque.random import RngKey

__all__ = ["AuditResult", "CoinFlip", "OneRunEstimator"]


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
        >>> result.epsilon_at(delta=1e-5)
        >>> result.auc()
        >>> result.beta_at(alpha=0.01)
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

    def __repr__(self) -> str:
        return (
            f"AuditResult(n_in={self.n_in}, n_out={self.n_out}, auc={self.auc():.4f})"
        )

    # ------------------------------------------------------------------
    # Epsilon estimation
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float = 0.0,
        significance: float = 0.05,
    ) -> float:
        """Epsilon lower bound at the given delta.

        Matches the accounting API (``DpProcess.epsilon_at(delta=)``).
        Uses the one-run likelihood-ratio test from Steinke et al. (2023).

        Args:
            delta: DP delta parameter. Default: 0 (pure DP).
            significance: Allowed failure probability (1 - confidence).

        Returns:
            Epsilon lower bound at the specified confidence level.
        """
        return self.epsilon_one_run(significance=significance, delta=delta)

    def epsilon_one_run(
        self,
        *,
        significance: float = 0.05,
        delta: float = 0.0,
        threshold: float | None = None,
        eps_max: float = 20.0,
        tol: float = 1e-4,
    ) -> float:
        """Epsilon lower bound using the one-run method (Steinke et al. 2023).

        Uses a likelihood-ratio test tailored for DP auditing. For each
        Pareto-optimal threshold, tries both positive-only guesses
        (k+ = TP + FP, k- = 0) and two-sided guesses (k+ = TP + FP,
        k- = TN + FN) per Algorithm 1 of the paper, and takes the best.

        Args:
            significance: Allowed failure probability (1 - confidence).
            delta: DP delta parameter. Default: 0 (pure DP).
            threshold: If provided, use this specific threshold.
            eps_max: Maximum epsilon to search. Default: 20.0.
            tol: Binary search tolerance. Default: 1e-4.

        Returns:
            Epsilon lower bound at the specified confidence level.

        Reference:
            Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
        """
        _validate_significance(significance)
        _validate_delta(delta)

        m = self.n_in + self.n_out

        if threshold is not None:
            tp = int(np.sum(self._in_arr >= threshold))
            fp = int(np.sum(self._out_arr >= threshold))
            tn = int(np.sum(self._out_arr < threshold))

            # Positive-only: r = TP + FP, v = TP
            eps_pos = _epsilon_one_run_search(
                tp + fp, tp, m, significance, delta, eps_max, tol
            )
            # Two-sided: r = m, v = TP + TN
            eps_both = _epsilon_one_run_search(
                m, tp + tn, m, significance, delta, eps_max, tol
            )
            return max(eps_pos, eps_both)

        # Bonferroni correction: divide significance by the number of
        # hypotheses tested.  We test 2 variants per threshold (positive-only
        # and two-sided), so the total count is 2 * num_thresholds.
        n_thresholds = len(self._thresholds)
        sig_corrected = significance / (2 * n_thresholds)
        best = 0.0
        for i in range(n_thresholds):
            tp_i = self.n_in - self._fn_counts[i]
            fp_i = self.n_out - self._tn_counts[i]
            tn_i = self._tn_counts[i]

            # Positive-only: r = TP + FP, v = TP
            eps_pos = _epsilon_one_run_search(
                tp_i + fp_i, tp_i, m, sig_corrected, delta, eps_max, tol
            )
            # Two-sided: r = m, v = TP + TN
            eps_both = _epsilon_one_run_search(
                m, tp_i + tn_i, m, sig_corrected, delta, eps_max, tol
            )
            best = max(best, eps_pos, eps_both)
        return best

    # ------------------------------------------------------------------
    # Attack utility metrics
    # ------------------------------------------------------------------

    def auc(
        self,
        *,
        confidence: float | None = None,
        num_samples: int = 1000,
        key: RngKey | None = None,
    ) -> float | tuple[float, float]:
        """Area under the ROC curve for the membership inference attack.

        AUC = 0.5 means random guessing, AUC = 1.0 means perfect attack.

        When ``confidence`` is provided, returns a confidence interval
        as a ``(lower, upper)`` tuple instead of a point estimate.

        Args:
            confidence: If provided, return a symmetric CI at this level
                (e.g. 0.95 for 95% CI). Must be in (0, 1).
            num_samples: Number of resamples for CI. Default: 1000.
            key: RNG key for reproducible resampling.

        Returns:
            Float AUC if ``confidence`` is None, otherwise
            ``(lower, upper)`` tuple.

        Example:
            >>> result.auc()                              # point estimate
            >>> result.auc(confidence=0.95, key=key(42))  # 95% CI
        """
        tnr = self._tn_counts / self._tn_counts[-1]
        fnr = self._fn_counts / self._fn_counts[-1]
        point = float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))

        if confidence is None:
            return point

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        significance = 1 - confidence
        quantiles = (significance / 2, 1 - significance / 2)

        rng = np.random.default_rng(seed=key.seed if key else None)
        values = np.empty(num_samples)
        for i in range(num_samples):
            in_sample = rng.choice(self._in_arr, size=self.n_in)
            out_sample = rng.choice(self._out_arr, size=self.n_out)
            values[i] = AuditResult(in_sample, out_sample).auc()

        # Bias-corrected bootstrap
        prop_less = (np.sum(values < point) + 1) / (num_samples + 2)
        z0 = scipy.stats.norm.ppf(prop_less)
        z_q = scipy.stats.norm.ppf(quantiles)
        corrected = scipy.stats.norm.cdf(z0 + (z0 + z_q) / (1 - 0.0 * (z0 + z_q)))

        ci = np.quantile(values, corrected, method="linear")
        return (float(ci[0]), float(ci[1]))

    def beta_at(self, *, alpha: float | np.ndarray) -> float | np.ndarray:
        """Type-II error rate at a given Type-I error rate.

        Consistent with ``DpProcess.beta_at(alpha=)`` in the accounting
        module.  Higher beta means the attack is weaker (more private).

        Relationship to TPR/FPR: ``beta = 1 - TPR`` at ``alpha = FPR``.

        Args:
            alpha: Type-I error rate(s) (false positive rate) in [0, 1].

        Returns:
            Type-II error rate(s) at the specified alpha(s).
        """
        tpr = _tpr_at_given_fpr(alpha, self._tp_counts, self._fp_counts)
        return 1.0 - tpr

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
    # Display
    # ------------------------------------------------------------------

    def summary(
        self,
        *,
        significance: float = 0.05,
        delta: float = 0.0,
        theoretical_epsilon: float | None = None,
    ) -> str:
        """Multi-line summary of all metrics.

        Matches the accounting API (``DpProcess.summary()``).

        Args:
            significance: Allowed failure probability for epsilon bounds.
            delta: DP delta parameter.
            theoretical_epsilon: If provided, display alongside empirical
                bound for comparison.

        Returns:
            Formatted string with all metrics.
        """
        eps = self.epsilon_one_run(significance=significance, delta=delta)

        lines = [
            "Audit Summary",
            "\u2500" * 40,
            f"  Samples:              {self.n_in} in, {self.n_out} out",
            f"  AUC:                  {self.auc():.4f}",
            f"  \u03b5 (one-run):          {eps:.4f}",
        ]

        if theoretical_epsilon is not None:
            lines.append(f"  \u03b5 (theoretical):      {theoretical_epsilon:.4f}")

        lines.extend(
            [
                f"  \u03b2 @ \u03b1=0.01:           {self.beta_at(alpha=0.01):.4f}",
                f"  \u03b2 @ \u03b1=0.10:           {self.beta_at(alpha=0.1):.4f}",
                f"  Max accuracy:         {self.max_accuracy():.4f}",
                f"  (\u03b1={significance}, \u03b4={delta})",
            ]
        )
        return "\n".join(lines)


class CoinFlip:
    """Coin-flip partitioning for canary-based privacy auditing.

    Each canary is independently included or excluded from training
    with probability 0.5 (a fair coin flip). This class only handles
    the partition — it does not know about scoring or epsilon estimation.

    Attributes:
        num_canaries: Total number of canary examples.
        canary_indices: All canary dataset indices.
        in_indices: Canary indices included in training (coin = heads).
        out_indices: Canary indices excluded from training (coin = tails).

    Reference:
        Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training
        Run." NeurIPS 2023. https://arxiv.org/abs/2305.08846
    """

    def __init__(
        self,
        canary_indices: np.ndarray,
        *,
        key: RngKey,
    ) -> None:
        canary_indices = np.asarray(canary_indices)
        if canary_indices.ndim != 1 or canary_indices.size == 0:
            raise ValueError("canary_indices must be a non-empty 1-D array")

        rng = np.random.default_rng(key.seed)
        in_mask = rng.random(len(canary_indices)) < 0.5

        self.num_canaries = len(canary_indices)
        self.canary_indices = canary_indices
        self._in_mask = in_mask
        self.in_indices = canary_indices[in_mask]
        self.out_indices = canary_indices[~in_mask]

    def __repr__(self) -> str:
        return (
            f"CoinFlip(num_canaries={self.num_canaries}, "
            f"n_in={len(self.in_indices)}, n_out={len(self.out_indices)})"
        )

    def train_indices(self, dataset_size: int) -> list[int]:
        """Dataset indices to use for training.

        Returns all indices in ``range(dataset_size)`` except the excluded
        canaries (coin = tails).

        Args:
            dataset_size: Total number of examples in the full dataset.

        Returns:
            Sorted list of training indices.
        """
        excluded = set(self.out_indices.tolist())
        return [i for i in range(dataset_size) if i not in excluded]

    def split_scores(
        self, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split per-canary scores into in-group and out-group.

        Args:
            scores: Membership scores, shape ``(num_canaries,)``, in the
                same order as ``canary_indices``.

        Returns:
            ``(in_scores, out_scores)`` tuple.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.num_canaries,):
            raise ValueError(
                f"Expected {self.num_canaries} scores, got shape {scores.shape}"
            )
        return scores[self._in_mask], scores[~self._in_mask]


class OneRunEstimator:
    """One-run estimator: wraps a :class:`CoinFlip` with scoring config.

    Returned by :func:`~opaque.auditing.setup` (or
    :func:`~opaque.auditing.one_run`). Passed as ``state`` to
    :func:`~opaque.auditing.evaluate`.

    The public surface is ``train_indices`` and ``coin_flip``.
    Scoring config is internal, consumed by :func:`evaluate`.

    Attributes:
        coin_flip: The :class:`CoinFlip` partition.
        train_indices: Sorted list of dataset indices to use for training
            (all indices except held-out canaries). Pass directly to
            ``dataset.select()``::

                audit_state = auditing.setup(dataset, ...)
                train_data = dataset.select(audit_state.train_indices)
    """

    def __init__(
        self,
        coin_flip: CoinFlip,
        *,
        dataset,
        batch_argnums: tuple[int, ...] | None = None,
        collate_fn=None,
        batch_unpack=None,
        batch_size: int = 256,
    ) -> None:
        self.coin_flip = coin_flip
        self._dataset = dataset
        self._batch_argnums = batch_argnums
        self._collate_fn = collate_fn
        self._batch_unpack = batch_unpack
        self._batch_size = batch_size
        self.train_indices = coin_flip.train_indices(len(dataset))

    def __repr__(self) -> str:
        cf = self.coin_flip
        return (
            f"OneRunEstimator(num_canaries={cf.num_canaries}, "
            f"n_in={len(cf.in_indices)}, n_out={len(cf.out_indices)})"
        )

    def audit(self, scores: np.ndarray) -> AuditResult:
        """Split scores by coin flip and return an AuditResult.

        Args:
            scores: Membership scores, shape ``(num_canaries,)``.

        Returns:
            :class:`AuditResult` with in/out scores split by coin flip.
        """
        in_scores, out_scores = self.coin_flip.split_scores(scores)
        return AuditResult(in_scores, out_scores)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _validate_significance(significance: float) -> None:
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")


def _validate_delta(delta: float) -> None:
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")


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
