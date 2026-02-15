//! Property tests — mathematical invariants and cross-cutting properties
//!
//! These tests verify that privacy metrics satisfy expected mathematical properties
//! (monotonicity, bounds, consistency) across different noise multipliers and parameters.

use super::*;

#[test]
fn test_pld_metric_properties() {
    // Build one PLD and check multiple properties on it.
    let pld = gaussian(0.8).unwrap().pld().unwrap();

    // 1. delta monotonically decreases with epsilon
    let epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut prev_delta = f64::INFINITY;
    for eps in epsilons {
        let delta = pld.delta_at(eps);
        assert!(
            delta < prev_delta,
            "delta not decreasing: d({})={} >= prev={}",
            eps,
            delta,
            prev_delta
        );
        assert!(
            delta >= 0.0 && delta <= 1.0,
            "delta out of range: {}",
            delta
        );
        prev_delta = delta;
    }

    // 2. epsilon monotonically decreases as delta increases
    let deltas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
    let mut prev_eps = f64::INFINITY;
    for d in deltas {
        let eps = pld.epsilon_at(d);
        assert!(
            eps <= prev_eps,
            "epsilon not decreasing: e({})={} > prev={}",
            d,
            eps,
            prev_eps
        );
        prev_eps = eps;
    }

    // 3. delta always finite for finite epsilon
    for &eps in &[0.0, 0.001, 0.1, 1.0, 10.0, 100.0] {
        let delta = pld.delta_at(eps);
        assert!(
            delta.is_finite(),
            "delta not finite at e={}: {}",
            eps,
            delta
        );
        assert!(delta >= 0.0 && delta <= 1.0);
    }

    // 4. epsilon always finite for delta > 0
    for &d in &[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1] {
        let eps = pld.epsilon_at(d);
        assert!(eps.is_finite(), "epsilon not finite at d={}: {}", d, eps);
        assert!(eps >= 0.0);
    }
}

#[test]
fn test_delta_decreases_with_noise() {
    // For fixed epsilon, delta should decrease as noise increases (better privacy)
    let epsilons = [1.0, 3.0, 5.0];
    let noise_multipliers = [0.3, 0.5, 0.8, 1.0, 1.2];
    for eps in epsilons {
        let mut prev_delta = f64::INFINITY;
        for noise in noise_multipliers {
            let delta = gaussian(noise).unwrap().pld().unwrap().delta_at(eps);
            assert!(
                delta < prev_delta,
                "d not decreasing at e={}: s={}, d={} >= prev={}",
                eps,
                noise,
                delta,
                prev_delta
            );
            prev_delta = delta;
        }
    }
}

#[test]
fn test_epsilon_decreases_with_noise() {
    // For fixed delta, epsilon should decrease as noise increases (better privacy)
    let deltas = [1e-6, 1e-5, 1e-4];
    let noise_multipliers = [0.3, 0.5, 0.8, 1.0, 1.2];
    for delta in deltas {
        let mut prev_eps = f64::INFINITY;
        for noise in noise_multipliers {
            let eps = gaussian(noise).unwrap().pld().unwrap().epsilon_at(delta);
            assert!(
                eps < prev_eps,
                "e not decreasing at d={}: s={}, e={} >= prev={}",
                delta,
                noise,
                eps,
                prev_eps
            );
            prev_eps = eps;
        }
    }
}

#[test]
fn test_boundary_epsilon_values() {
    // Epsilon with large noise multiplier
    let pld_large = gaussian(1.2).unwrap().pld().unwrap();
    let delta = pld_large.delta_at(0.001);
    assert!(delta.is_finite() && delta >= 0.0 && delta <= 1.0);

    // Large epsilon - delta should approach 0
    let pld_small = gaussian(0.5).unwrap().pld().unwrap();
    let delta = pld_small.delta_at(50.0);
    assert!(delta.is_finite() && delta >= 0.0);
    assert!(delta < 1e-10, "delta should be tiny for large e: {}", delta);

    // Zero epsilon: delta(0) = Phi(1) - Phi(-1) ~ 0.683
    let delta = pld_small.delta_at(0.0);
    assert!(
        delta > 0.6 && delta < 0.7,
        "delta at e=0 for s=0.5 should be ~0.68: {}",
        delta
    );

    // Negative epsilon
    let delta = pld_small.delta_at(-1.0);
    assert!(delta.is_finite() && delta >= 0.5 && delta <= 1.0);
}

#[test]
fn test_training_scenarios() {
    // CIFAR-10 fine-tuning: s=1.0, d=1e-5
    let eps = gaussian(1.0).unwrap().epsilon_at(1e-5).unwrap();
    assert!(eps > 0.0 && eps < 10.0, "cifar10: {}", eps);

    // ImageNet pre-training: s=0.5, d=1e-6
    let eps = gaussian(0.5).unwrap().epsilon_at(1e-6).unwrap();
    assert!(eps > 1.0 && eps < 20.0, "imagenet: {}", eps);

    // LLM fine-tuning: s=0.8, d=1e-7
    let eps = gaussian(0.8).unwrap().epsilon_at(1e-7).unwrap();
    assert!(eps > 0.0 && eps < 15.0, "llm: {}", eps);

    // Low noise / high epsilon
    let delta = gaussian(0.3).unwrap().pld().unwrap().delta_at(10.0);
    assert!(delta > 0.0 && delta < 0.1, "low noise: {}", delta);

    // High noise / low epsilon
    let delta = gaussian(1.2).unwrap().pld().unwrap().delta_at(0.1);
    assert!(delta > 0.1, "high noise: {}", delta);
}
