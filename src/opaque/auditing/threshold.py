"""Threshold selection strategies for privacy auditing.

In membership inference attacks, we classify examples as "member" (held-in) or
"non-member" (held-out) based on an attack score exceeding a threshold. Different
threshold strategies handle the bias-variance tradeoff differently:

- **Explicit**: Use a pre-specified threshold (no data-driven selection)
- **Split**: Split data once to select threshold, then compute bound on remainder
- **MultiSplit**: Split data multiple times and take median (more robust)
- **Bonferroni**: Use all thresholds with Bonferroni correction (most conservative)

See Section 4 of https://arxiv.org/pdf/2305.08846 for details.
"""

from dataclasses import dataclass


class ThresholdStrategy:
    """Base class for threshold selection strategies.

    Subclasses define different strategies for choosing the classification threshold
    in membership inference attacks. The choice affects the privacy bound's
    bias-variance tradeoff:
    - More data-driven selection → lower variance, higher bias risk
    - Less data-driven selection → higher variance, lower bias

    Subclasses: Explicit, Split, MultiSplit, Bonferroni
    """


class Bonferroni(ThresholdStrategy):
    """Use Bonferroni correction across all possible thresholds.

    This strategy considers all possible threshold values (all unique score values
    in the data) and applies Bonferroni correction to control the family-wise error
    rate. This is the most conservative approach but requires no data splitting.

    The significance level is divided by the number of possible thresholds, ensuring
    that the overall false positive rate is bounded.

    Pros:
    - No data splitting (uses all data for bound computation)
    - No risk of overfitting to threshold selection

    Cons:
    - Most conservative (widest confidence intervals)
    - Slower (tests all thresholds)

    Example:
        >>> auditor = CanaryScoreAuditor(in_scores, out_scores)
        >>> eps = auditor.epsilon_clopper_pearson(
        ...     threshold_strategy=Bonferroni(),
        ...     significance=0.05,
        ... )
    """


@dataclass(frozen=True)
class Explicit(ThresholdStrategy):
    """Use a specific pre-specified threshold value.

    This strategy uses a threshold chosen independently of the data (e.g., from
    prior knowledge or theory). No data is used for threshold selection, so all
    data is available for computing the privacy bound.

    This is ideal when you have a principled choice of threshold (e.g., from
    a theoretical analysis or separate validation set).

    Attributes:
        threshold: The threshold value to use for classification. Examples with
            attack score ≥ threshold are classified as "held-in".

    Example:
        >>> # Use threshold from theoretical analysis
        >>> auditor = CanaryScoreAuditor(in_scores, out_scores)
        >>> eps = auditor.epsilon_clopper_pearson(
        ...     threshold_strategy=Explicit(threshold=0.5),
        ...     significance=0.05,
        ... )
    """

    threshold: float


@dataclass(frozen=True)
class Split(ThresholdStrategy):
    """Split data once to choose threshold, then compute bound on remainder.

    This strategy splits the canary scores into two disjoint sets:
    1. Threshold selection set: Used to choose the threshold (e.g., by maximizing
       accuracy)
    2. Bound computation set: Used to compute the privacy bound with the chosen
       threshold

    The split avoids overfitting the threshold to the data used for the privacy
    bound, but reduces the effective sample size by half.

    Attributes:
        threshold_estimation_frac: Fraction of data (in [0, 1]) to use for threshold
            selection. Default is 0.5 (half for threshold, half for bound).
        seed: Random seed for reproducible splitting. If None, a non-deterministic
            seed is chosen.

    Example:
        >>> # Use 50% of data for threshold selection
        >>> auditor = CanaryScoreAuditor(in_scores, out_scores)
        >>> eps = auditor.epsilon_clopper_pearson(
        ...     threshold_strategy=Split(threshold_estimation_frac=0.5, seed=42),
        ...     significance=0.05,
        ... )
    """

    threshold_estimation_frac: float = 0.5
    seed: int | None = None

    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.threshold_estimation_frac < 1:
            raise ValueError(
                f"threshold_estimation_frac must be in (0, 1), "
                f"got {self.threshold_estimation_frac}"
            )


@dataclass(frozen=True)
class MultiSplit(ThresholdStrategy):
    """Split data multiple times and take median bound with significance correction.

    This strategy extends Split by performing multiple random splits and computing
    the privacy bound for each split. The final bound is the median of the per-split
    bounds, with significance level adjusted to maintain overall validity.

    Based on Meinshausen et al. (2009), "Stability selection": Theorem 3.1 shows that
    2 * median(p_i) is a valid p-value, which implies that the median of bounds
    computed at significance alpha/2 corresponds to a valid rejection at level alpha.

    This provides more stable bounds than a single split while still avoiding
    threshold overfitting.

    Attributes:
        num_samples: Number of random splits to perform. More splits give more
            stable results but increase computation. Default: 100.
        threshold_estimation_frac: Fraction of data to use for threshold selection
            in each split. Default: 0.5.
        seed: Random seed for reproducibility. If None, a non-deterministic seed
            is chosen.

    Reference:
        Meinshausen et al. (2009), https://arxiv.org/pdf/0811.2177

    Example:
        >>> # Use 100 splits for stable estimates
        >>> auditor = CanaryScoreAuditor(in_scores, out_scores)
        >>> eps = auditor.epsilon_clopper_pearson(
        ...     threshold_strategy=MultiSplit(num_samples=100, seed=42),
        ...     significance=0.05,
        ... )
    """

    num_samples: int = 100
    threshold_estimation_frac: float = 0.5
    seed: int | None = None

    def __post_init__(self):
        """Validate parameters."""
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if not 0 < self.threshold_estimation_frac < 1:
            raise ValueError(
                f"threshold_estimation_frac must be in (0, 1), "
                f"got {self.threshold_estimation_frac}"
            )


__all__ = ["ThresholdStrategy", "Bonferroni", "Explicit", "Split", "MultiSplit"]
