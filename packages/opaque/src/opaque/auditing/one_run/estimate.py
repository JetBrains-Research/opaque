"""One-run privacy audit estimator (Steinke et al. 2023).

The ``one_run()`` function precomputes the Pareto-optimal threshold
structure from canary scores and returns a frozen ``OneRunEstimate``
that holds all precomputed state. Query methods on the estimate
(``epsilon_at``, ``beta_at``, etc.) dispatch into that state.

References:
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.stats

from opaque.auditing.coin_flip import CoinFlip
from opaque.auditing.one_run.roc import get_tn_fn_counts, tpr_at_given_fpr
from opaque.auditing.one_run.stats import (
    epsilon_one_run_search,
    validate_delta,
    validate_significance,
)
from opaque.random import RngKey

__all__ = ["OneRunEstimate", "one_run"]


def one_run(scores: np.ndarray, *, coin_flip: CoinFlip) -> OneRunEstimate:
    """Build a one-run privacy estimate from canary scores.

    Splits scores by the coin-flip partition, precomputes the
    Pareto-optimal ROC frontier, and returns a frozen estimate.

    Args:
        scores: Per-canary membership scores, shape ``(num_canaries,)``.
            Higher score = more likely a training member.
        coin_flip: The :class:`~opaque.auditing.CoinFlip` partition.

    Returns:
        A :class:`OneRunEstimate` with precomputed threshold structure.

    Example::

        import opaque.auditing as auditing
        from opaque.random import key

        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        scores = auditing.loss_scores(loss_fn, params,
                                       batch_argnums=(1,),
                                       dataset=dataset,
                                       indices=cf.canary_indices)
        estimate = auditing.one_run(scores, coin_flip=cf)
        print(estimate.epsilon_at(delta=1e-5))
    """
    in_scores, out_scores = coin_flip.split_scores(scores)

    if in_scores.size == 0 or out_scores.size == 0:
        raise ValueError("Both in_scores and out_scores must be non-empty")

    thresholds, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return OneRunEstimate(
        n_in=len(in_scores),
        n_out=len(out_scores),
        thresholds=thresholds,
        tn_counts=tn_counts,
        fn_counts=fn_counts,
        tp_counts=tp_counts,
        fp_counts=fp_counts,
        in_scores=in_scores,
        out_scores=out_scores,
    )


@dataclasses.dataclass(frozen=True)
class OneRunEstimate:
    """Precomputed one-run audit estimate.

    Constructed by :func:`one_run`. Holds the Pareto-optimal threshold
    structure and exposes query methods for privacy metrics.

    This is a frozen dataclass — all heavy computation happens in
    :func:`one_run`, and query methods dispatch into precomputed fields.
    """

    n_in: int
    n_out: int
    thresholds: np.ndarray
    tn_counts: np.ndarray
    fn_counts: np.ndarray
    tp_counts: np.ndarray
    fp_counts: np.ndarray
    in_scores: np.ndarray
    out_scores: np.ndarray

    def __repr__(self) -> str:
        return (
            f"OneRunEstimate(n_in={self.n_in}, n_out={self.n_out}, "
            f"auc={self.auc():.4f})"
        )

    # ------------------------------------------------------------------
    # Epsilon estimation
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float = 0.0,
        significance: float = 0.05,
        threshold: float | None = None,
        eps_max: float = 20.0,
        tol: float = 1e-4,
    ) -> float:
        """Epsilon lower bound using the one-run method (Steinke et al. 2023).

        Uses a likelihood-ratio test tailored for DP auditing. For each
        Pareto-optimal threshold, tries positive-only, negative-only,
        and two-sided guesses per Algorithm 1 of the paper, with
        Bonferroni correction over all thresholds and variants.

        Args:
            delta: DP delta parameter. Default: 0 (pure DP).
            significance: Allowed failure probability (1 - confidence).
            threshold: If provided, use this specific threshold instead
                of searching over all Pareto-optimal thresholds.
            eps_max: Maximum epsilon to search. Default: 20.0.
            tol: Binary search tolerance. Default: 1e-4.

        Returns:
            Epsilon lower bound at the specified confidence level.
        """
        validate_significance(significance)
        validate_delta(delta)

        m = self.n_in + self.n_out

        if threshold is not None:
            tp = int(np.sum(self.in_scores >= threshold))
            fp = int(np.sum(self.out_scores >= threshold))
            tn = int(np.sum(self.out_scores < threshold))

            # Bonferroni over the three variants (positive-only, negative-only, two-sided)
            sig_corrected = significance / 3.0
            fn = int(np.sum(self.in_scores < threshold))

            eps_pos = epsilon_one_run_search(
                tp + fp, tp, m, sig_corrected, delta, eps_max, tol
            )
            eps_neg = epsilon_one_run_search(
                fn + tn, tn, m, sig_corrected, delta, eps_max, tol
            )
            eps_both = epsilon_one_run_search(
                m, tp + tn, m, sig_corrected, delta, eps_max, tol
            )
            return max(eps_pos, eps_neg, eps_both)

        n_thresholds = len(self.thresholds)
        sig_corrected = significance / (3 * n_thresholds)
        best = 0.0
        for i in range(n_thresholds):
            tp_i = self.n_in - self.fn_counts[i]
            fp_i = self.n_out - self.tn_counts[i]
            fn_i = self.fn_counts[i]
            tn_i = self.tn_counts[i]

            eps_pos = epsilon_one_run_search(
                tp_i + fp_i, tp_i, m, sig_corrected, delta, eps_max, tol
            )
            eps_neg = epsilon_one_run_search(
                fn_i + tn_i, tn_i, m, sig_corrected, delta, eps_max, tol
            )
            eps_both = epsilon_one_run_search(
                m, tp_i + tn_i, m, sig_corrected, delta, eps_max, tol
            )
            best = max(best, eps_pos, eps_neg, eps_both)
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
        """
        point = _auc_from_counts(self.tn_counts, self.fn_counts)

        if confidence is None:
            return point

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        significance = 1 - confidence
        quantiles = (significance / 2, 1 - significance / 2)

        rng = np.random.default_rng(seed=key.seed if key else None)
        values = np.empty(num_samples)
        for i in range(num_samples):
            in_sample = rng.choice(self.in_scores, size=self.n_in)
            out_sample = rng.choice(self.out_scores, size=self.n_out)
            _, tn, fn = get_tn_fn_counts(in_sample, out_sample)
            values[i] = _auc_from_counts(tn, fn)

        # Bias-corrected bootstrap
        prop_less = (np.sum(values < point) + 1) / (num_samples + 2)
        z0 = scipy.stats.norm.ppf(prop_less)
        z_q = scipy.stats.norm.ppf(quantiles)
        corrected = scipy.stats.norm.cdf(2 * z0 + z_q)

        ci = np.quantile(values, corrected, method="linear")
        return (float(ci[0]), float(ci[1]))

    def beta_at(self, *, alpha: float | np.ndarray) -> float | np.ndarray:
        """Type-II error rate at a given Type-I error rate.

        Higher beta means the attack is weaker (more private).

        Args:
            alpha: Type-I error rate(s) (false positive rate) in [0, 1].

        Returns:
            Type-II error rate(s) at the specified alpha(s).
        """
        tpr = tpr_at_given_fpr(alpha, self.tp_counts, self.fp_counts)
        return 1.0 - tpr

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

        Args:
            significance: Allowed failure probability for epsilon bounds.
            delta: DP delta parameter.
            theoretical_epsilon: If provided, display alongside empirical
                bound for comparison.

        Returns:
            Formatted string with all metrics.
        """
        eps = self.epsilon_at(significance=significance, delta=delta)

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
                f"  (\u03b1={significance}, \u03b4={delta:.2e})",
            ]
        )
        return "\n".join(lines)


def _auc_from_counts(tn_counts: np.ndarray, fn_counts: np.ndarray) -> float:
    """Compute AUC from precomputed TN/FN count arrays."""
    tnr = tn_counts / tn_counts[-1]
    fnr = fn_counts / fn_counts[-1]
    return float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))
