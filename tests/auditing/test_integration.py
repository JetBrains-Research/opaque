"""Integration tests for privacy auditing workflow."""

import numpy as np

from opaque.auditing import (
    BootstrapParams,
    attack_auroc,
    audit,
    bootstrap,
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
    max_accuracy,
    tpr_at_fpr,
)


def test_basic_audit_workflow():
    """Test basic privacy auditing workflow."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=100)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=100)

    # Test epsilon estimation with explicit threshold
    eps = epsilon_clopper_pearson(
        in_scores, out_scores, significance=0.05, delta=1e-5, threshold=4.0
    )
    assert eps > 0, "Should detect privacy leakage"

    # Test epsilon with Bonferroni (default)
    eps_bonf = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05, delta=1e-5)
    assert eps_bonf > 0, "Bonferroni should also detect leakage"

    # Test utility metrics
    auroc = attack_auroc(in_scores, out_scores)
    assert 0.5 < auroc < 1.0, "AUROC should be above random"

    tpr = tpr_at_fpr(in_scores, out_scores, fpr=0.05)
    assert tpr > 0.05, "TPR should exceed FPR for good attack"

    acc = max_accuracy(in_scores, out_scores)
    assert acc > 0.5, "Accuracy should exceed random"


def test_audit_with_bootstrap():
    """Test auditing with bootstrap confidence intervals."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=50)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=50)

    params = BootstrapParams(num_samples=20, seed=42)

    # Test AUROC with bootstrap
    auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)
    assert len(auroc_ci) == 2
    assert auroc_ci[0] <= auroc_ci[1]


def test_no_privacy_leakage():
    """Test auditing when there's no privacy leakage."""
    np.random.seed(42)
    scores = np.random.normal(loc=3.0, scale=1.0, size=100)

    in_scores = scores[:50]
    out_scores = scores[50:]

    eps = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05, delta=0)
    assert eps < 0.5, "Should detect minimal leakage"

    auroc = attack_auroc(in_scores, out_scores)
    assert 0.4 < auroc < 0.6, "AUROC should be near random"


def test_perfect_attack():
    """Test auditing with perfect attack separation."""
    in_scores = np.arange(50, 100, dtype=float)
    out_scores = np.arange(0, 50, dtype=float)

    auroc = attack_auroc(in_scores, out_scores)
    assert auroc > 0.99, "Perfect attack should have AUROC ≈ 1.0"

    eps = epsilon_clopper_pearson(
        in_scores, out_scores, significance=0.05, delta=0, threshold=50
    )
    assert eps > 2.5, "Perfect attack should give large epsilon"


def test_real_world_scenario():
    """Test realistic privacy auditing scenario."""
    np.random.seed(42)
    in_scores = np.random.exponential(scale=2.5, size=200)
    out_scores = np.random.exponential(scale=2.0, size=200)

    eps = epsilon_raw_counts(in_scores, out_scores, min_count=20, delta=1e-5)
    assert eps > 0, "Should detect some privacy leakage"

    auroc = attack_auroc(in_scores, out_scores)
    assert 0.5 < auroc < 0.75, "AUROC should show modest attack success"

    tpr_at_1pct = tpr_at_fpr(in_scores, out_scores, fpr=0.01)
    assert tpr_at_1pct < 0.3, "TPR should be limited at low FPR"


def test_one_run_audit():
    """Test one-run privacy auditing method."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=100)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=100)

    eps_one_run = epsilon_one_run(in_scores, out_scores, significance=0.05, delta=1e-5)
    assert eps_one_run > 0, "Should detect privacy leakage"

    eps_cp = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05, delta=1e-5)
    assert eps_cp > 0, "Clopper-Pearson should also detect leakage"


def test_audit_convenience_function():
    """Test the audit convenience function."""
    np.random.seed(42)
    in_scores = np.random.normal(loc=5.0, scale=1.0, size=100)
    out_scores = np.random.normal(loc=3.0, scale=1.0, size=100)

    result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)

    assert result.epsilon > 0
    assert 0.5 < result.auroc < 1.0
    assert result.tpr_at_low_fpr >= 0
    assert result.max_accuracy > 0.5
