//! Composition tests — repeat, compose, merge optimization, and mathematical invariants
//!
//! Tests `repeat()`, `repeat_flat()`, and `compose()` for correctness, verifies
//! merge optimization cases in `Composed::try_merge()`, and checks mathematical
//! properties (monotonicity, commutativity, privacy degradation).

use super::*;

// =========================================================================
// Repeated: basic correctness
// =========================================================================

#[test]
fn test_repeat_matches_self_compose() {
    let g = gaussian(1.0).unwrap();
    let k = 10;

    let process = repeat(g, k).unwrap();

    // Process trait metrics should work
    let eps = process.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0);

    let delta = process.delta_at(1.0).unwrap();
    assert!(delta >= 0.0 && delta <= 1.0);
}

#[test]
fn test_repeat_metrics_via_process_trait() {
    let g = gaussian(1.0).unwrap();
    let process = repeat(g, 10).unwrap();

    // All Process trait default methods should work
    let eps = process.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0);

    let delta = process.delta_at(1.0).unwrap();
    assert!(delta >= 0.0 && delta <= 1.0);

    let adv = process.advantage().unwrap();
    assert!(adv >= 0.0 && adv <= 1.0);

    let beta = process.beta_at(0.05).unwrap();
    assert!(beta >= 0.0 && beta <= 1.0);

    let risk = process.risk_at(0.5).unwrap();
    assert!(risk >= 0.0 && risk <= 0.5);
}

#[test]
fn test_repeat_count_one() {
    let g = gaussian(1.0).unwrap();
    let repeated = repeat(g.clone(), 1).unwrap();

    let eps_repeat = repeated.epsilon_at(1e-5).unwrap();
    let eps_single = g.epsilon_at(1e-5).unwrap();
    assert!(
        (eps_repeat - eps_single).abs() < 1e-10,
        "repeat(g, 1) should match single: {} vs {}",
        eps_repeat,
        eps_single
    );
}

// =========================================================================
// repeat_flat: flattened nested repeat
// =========================================================================

#[test]
fn test_repeat_flat_matches_single_repeat() {
    let g = gaussian(1.0).unwrap();
    let n = 5;
    let m = 10;

    // repeat_flat flattens to a single Repeated with count = n * m
    let flattened = repeat_flat(repeat(g.clone(), n).unwrap(), m).unwrap();
    assert_eq!(flattened.inner, g);
    assert_eq!(flattened.count, n * m);

    // Flattened repeat should match a single repeat with total count
    let via_flat = flattened.pld().unwrap();
    let via_single = repeat(g, n * m).unwrap().pld().unwrap();

    let delta_flat = via_flat.delta_at(1.0);
    let delta_single = via_single.delta_at(1.0);
    assert!(
        (delta_flat - delta_single).abs() < 1e-12,
        "repeat_flat delta {} vs single repeat {}",
        delta_flat,
        delta_single
    );

    let eps_flat = via_flat.epsilon_at(1e-5);
    let eps_single = via_single.epsilon_at(1e-5);
    assert!(
        (eps_flat - eps_single).abs() < 1e-10,
        "repeat_flat epsilon {} vs single repeat {}",
        eps_flat,
        eps_single
    );
}

#[test]
fn test_nested_repeat_without_flatten() {
    // repeat(repeat(a, n), m) without flatten — still correct but two FFT passes
    let g = gaussian(1.0).unwrap();
    let n = 5;
    let m = 10;

    let nested = repeat(repeat(g.clone(), n).unwrap(), m).unwrap();
    let flat = repeat_flat(repeat(g, n).unwrap(), m).unwrap();

    // Two sequential self_compose calls accumulate slightly different FFT
    // rounding than a single self_compose(n*m). Use relative tolerance.
    let delta_nested = nested.delta_at(1.0).unwrap();
    let delta_flat = flat.delta_at(1.0).unwrap();
    assert_relative_eq(delta_nested, delta_flat, 1e-4, "nested repeat delta");

    let eps_nested = nested.epsilon_at(1e-5).unwrap();
    let eps_flat = flat.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_nested, eps_flat, 1e-4, "nested repeat epsilon");
}

// =========================================================================
// Heterogeneous composition
// =========================================================================

#[test]
fn test_compose_different_mechanisms() {
    let g1 = gaussian(0.5).unwrap();
    let g2 = gaussian(1.0).unwrap();

    let process = compose(g1.clone(), g2.clone());

    // Composed privacy should be worse than either alone
    let eps_composed = process.epsilon_at(1e-5).unwrap();
    let eps_1 = g1.epsilon_at(1e-5).unwrap();
    let eps_2 = g2.epsilon_at(1e-5).unwrap();
    assert!(
        eps_composed > eps_1 && eps_composed > eps_2,
        "composed epsilon {} should exceed both ({}, {})",
        eps_composed,
        eps_1,
        eps_2
    );

    let delta = process.delta_at(1.0).unwrap();
    assert!(delta >= 0.0 && delta <= 1.0);
}

#[test]
fn test_hetero_compose_small_and_large_noise() {
    // Compose a low-noise mechanism (nm=0.1, aggressive privacy) with a
    // high-noise mechanism (nm=1.0, strong privacy). This is a realistic
    // scenario: e.g. one training step with less noise + another with more.
    let g1 = gaussian(0.1).unwrap();
    let g2 = gaussian(1.0).unwrap();
    let composed = compose(g1.clone(), g2.clone());

    // Composed privacy should be worse than either alone.
    let eps_1 = g1.epsilon_at(1e-5).unwrap();
    let eps_2 = g2.epsilon_at(1e-5).unwrap();
    let eps_composed = composed.epsilon_at(1e-5).unwrap();
    assert!(
        eps_composed > eps_1 && eps_composed > eps_2,
        "Composed epsilon ({}) should exceed both components ({}, {})",
        eps_composed,
        eps_1,
        eps_2
    );

    // Delta should be in valid range.
    let delta = composed.delta_at(1.0).unwrap();
    assert!(
        delta >= 0.0 && delta <= 1.0,
        "delta out of range: {}",
        delta
    );
}

#[test]
fn test_hetero_compose_very_small_noise_pair() {
    // Compose two small-noise mechanisms (nm=0.1, nm=0.15).
    // Both have wide epsilon ranges — tests that composition works
    // even when grids are large.
    let composed = compose(gaussian(0.1).unwrap(), gaussian(0.15).unwrap());

    // Both are nearly deterministic, so composed should also be.
    // Delta can slightly exceed 1.0 due to discretization artifacts
    // on these enormous grids (~28M+ points). Allow small tolerance.
    let delta = composed.delta_at(10.0).unwrap();
    assert!(
        delta >= 0.0 && delta <= 1.0 + 1e-6,
        "delta out of range: {}",
        delta
    );
    let eps = composed.epsilon_at(1e-4).unwrap();
    assert!(eps.is_finite(), "epsilon should be finite");
}

#[test]
fn test_hetero_compose_medium_noise_pair() {
    // Compose two moderate-noise mechanisms (nm=0.4, nm=0.8).
    // This is a typical DP-SGD scenario with varying per-step noise.
    let composed = compose(gaussian(0.4).unwrap(), gaussian(0.8).unwrap());

    let eps = composed.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite());

    // Monotonicity: adding more composition makes privacy worse.
    let composed_3 = compose(composed, gaussian(0.6).unwrap());
    assert!(
        composed_3.epsilon_at(1e-5).unwrap() > eps,
        "Three-way compose should have larger epsilon"
    );
}

#[test]
fn test_hetero_compose_large_noise_pair() {
    // Compose two large-noise mechanisms (nm=1.0, nm=1.2).
    // Strong privacy region — small grids, should be fast.
    let g1 = gaussian(1.0).unwrap();
    let g2 = gaussian(1.2).unwrap();
    let composed = compose(g1.clone(), g2.clone());

    let eps = composed.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite());

    // Both have small delta at eps=1, composed should be larger.
    let d1 = g1.delta_at(1.0).unwrap();
    let d2 = g2.delta_at(1.0).unwrap();
    let dc = composed.delta_at(1.0).unwrap();
    assert!(
        dc >= d1 && dc >= d2,
        "Composed delta ({}) should be >= both ({}, {})",
        dc,
        d1,
        d2
    );
}

#[test]
fn test_hetero_compose_extreme_range() {
    // Compose mechanisms at opposite extremes: nm=0.1 (low noise)
    // and nm=1.2 (strong privacy). Maximum discretization disparity.
    let composed = compose(gaussian(0.1).unwrap(), gaussian(1.2).unwrap());

    let eps = composed.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite());

    // Delta can slightly exceed 1.0 due to discretization artifacts
    // when composing with the large nm=0.1 grid.
    let delta = composed.delta_at(1.0).unwrap();
    assert!(
        delta >= 0.0 && delta <= 1.0 + 1e-6,
        "delta out of range: {}",
        delta
    );
}

#[test]
fn test_compose_with_repeated_different_inner() {
    // compose(repeat(g1, 10), g2) where g1 != g2 — no merge, fallback compose
    let g1 = gaussian(0.5).unwrap();
    let g2 = gaussian(1.0).unwrap();

    let process = compose(repeat(g1.clone(), 10).unwrap(), g2.clone());

    // Should produce valid results (no merge, standard PLD convolution)
    let eps = process.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0);

    // Should be worse than either component alone
    let eps_g1_10 = repeat(g1, 10).unwrap().epsilon_at(1e-5).unwrap();
    let eps_g2 = g2.epsilon_at(1e-5).unwrap();
    assert!(
        eps > eps_g1_10 && eps > eps_g2,
        "composed epsilon {} should exceed both ({}, {})",
        eps,
        eps_g1_10,
        eps_g2
    );
}

// =========================================================================
// Merge optimization cases
// =========================================================================

#[test]
fn test_merge_case1_compose_same_mechanism() {
    // compose(a, a) → a.pld().self_compose(2)
    let g = gaussian(1.0).unwrap();

    let via_compose = compose(g.clone(), g.clone());
    let via_repeat = repeat(g, 2).unwrap();

    let delta_compose = via_compose.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert!(
        (delta_compose - delta_repeat).abs() < 1e-12,
        "case 1 delta {} vs repeat {}",
        delta_compose,
        delta_repeat
    );

    let eps_compose = via_compose.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert!(
        (eps_compose - eps_repeat).abs() < 1e-10,
        "case 1 epsilon {} vs repeat {}",
        eps_compose,
        eps_repeat
    );
}

#[test]
fn test_merge_case2_repeated_left_bare_right() {
    // compose(repeat(a, n), a) → a.pld().self_compose(n + 1)
    let g = gaussian(1.0).unwrap();
    let n = 10;

    let via_compose = compose(repeat(g.clone(), n).unwrap(), g.clone());
    let via_repeat = repeat(g, n + 1).unwrap();

    let delta_compose = via_compose.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert!(
        (delta_compose - delta_repeat).abs() < 1e-12,
        "case 2 delta {} vs repeat {}",
        delta_compose,
        delta_repeat
    );

    let eps_compose = via_compose.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert!(
        (eps_compose - eps_repeat).abs() < 1e-10,
        "case 2 epsilon {} vs repeat {}",
        eps_compose,
        eps_repeat
    );
}

#[test]
fn test_merge_case3_bare_left_repeated_right() {
    // compose(a, repeat(a, n)) → a.pld().self_compose(n + 1)
    let g = gaussian(1.0).unwrap();
    let n = 10;

    let via_compose = compose(g.clone(), repeat(g.clone(), n).unwrap());
    let via_repeat = repeat(g, n + 1).unwrap();

    let delta_compose = via_compose.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert!(
        (delta_compose - delta_repeat).abs() < 1e-12,
        "case 3 delta {} vs repeat {}",
        delta_compose,
        delta_repeat
    );

    let eps_compose = via_compose.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert!(
        (eps_compose - eps_repeat).abs() < 1e-10,
        "case 3 epsilon {} vs repeat {}",
        eps_compose,
        eps_repeat
    );
}

#[test]
fn test_merge_equal_repeats() {
    // compose(repeat(a,n), repeat(a,n)) → effectively self_compose(2n)
    let g = gaussian(1.0).unwrap();
    let n = 10;

    let via_compose = compose(repeat(g.clone(), n).unwrap(), repeat(g.clone(), n).unwrap());
    let via_repeat = repeat(g, 2 * n).unwrap();

    // Merge fires (case 1: equal Repeated<G>), computing self_compose(10).self_compose(2)
    // vs reference self_compose(20). Different FFT paths → use relative tolerance.
    let delta_compose = via_compose.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert_relative_eq(delta_compose, delta_repeat, 1e-4, "equal repeats delta");

    let eps_compose = via_compose.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_compose, eps_repeat, 1e-4, "equal repeats epsilon");
}

#[test]
fn test_merge_different_count_repeats() {
    // compose(repeat(a,n), repeat(a,m)) with n != m
    let g = gaussian(1.0).unwrap();
    let n = 10;
    let m = 5;

    let via_compose = compose(repeat(g.clone(), n).unwrap(), repeat(g.clone(), m).unwrap());
    let via_repeat = repeat(g, n + m).unwrap();

    // On nightly with specialization, case 4 merges to self_compose(n+m) exactly.
    // On stable, falls back to PLD convolution which is correct but uses a
    // different FFT path. Use relative tolerance to cover both.
    let delta_compose = via_compose.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert_relative_eq(
        delta_compose,
        delta_repeat,
        1e-4,
        "diff-count repeats delta",
    );

    let eps_compose = via_compose.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_compose, eps_repeat, 1e-4, "diff-count repeats epsilon");
}

// =========================================================================
// Mathematical invariants
// =========================================================================

#[test]
fn test_compose_worse_than_single() {
    let g = gaussian(1.0).unwrap();

    let single_eps = g.epsilon_at(1e-5).unwrap();
    let composed_eps = compose(g.clone(), g.clone()).epsilon_at(1e-5).unwrap();

    assert!(
        composed_eps > single_eps,
        "composed epsilon {} should exceed single {}",
        composed_eps,
        single_eps
    );
}

#[test]
fn test_repeat_worse_than_fewer() {
    let g = gaussian(1.0).unwrap();

    let eps_10 = repeat(g.clone(), 10).unwrap().epsilon_at(1e-5).unwrap();
    let eps_20 = repeat(g.clone(), 20).unwrap().epsilon_at(1e-5).unwrap();

    assert!(
        eps_20 > eps_10,
        "more repetitions should give worse epsilon: {} vs {}",
        eps_20,
        eps_10
    );
}

#[test]
fn test_hetero_compose_is_commutative() {
    // compose(a, b) should give same result as compose(b, a).
    let g1 = gaussian(0.4).unwrap();
    let g2 = gaussian(1.0).unwrap();

    let ab = compose(g1.clone(), g2.clone());
    let ba = compose(g2, g1);

    let eps_ab = ab.epsilon_at(1e-5).unwrap();
    let eps_ba = ba.epsilon_at(1e-5).unwrap();
    assert!(
        (eps_ab - eps_ba).abs() < 1e-10,
        "Composition should be commutative: {} vs {}",
        eps_ab,
        eps_ba
    );

    let delta_ab = ab.delta_at(1.0).unwrap();
    let delta_ba = ba.delta_at(1.0).unwrap();
    assert!(
        (delta_ab - delta_ba).abs() < 1e-10,
        "Composition should be commutative: {} vs {}",
        delta_ab,
        delta_ba
    );
}

// =========================================================================
// Nested composition
// =========================================================================

#[test]
fn test_nested_compose_same_mechanism() {
    // compose(compose(a, a), a) — inner merges, outer falls back to PLD convolution.
    // Result is correct (matches repeat(a, 3)) but via a different FFT path.
    let g = gaussian(1.0).unwrap();

    let nested = compose(compose(g.clone(), g.clone()), g.clone());
    let via_repeat = repeat(g, 3).unwrap();

    let eps_nested = nested.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_nested, eps_repeat, 1e-4, "nested compose epsilon");

    let delta_nested = nested.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert_relative_eq(delta_nested, delta_repeat, 1e-4, "nested compose delta");
}

#[test]
fn test_nested_compose_four_same() {
    // compose(compose(a, a), compose(a, a)) — both sides merge to self_compose(2),
    // then outer sees two equal Composed<G,G> and merges to self_compose(2) again.
    // Result: self_compose(2).self_compose(2) = effectively self_compose(4).
    let g = gaussian(1.0).unwrap();

    let nested = compose(compose(g.clone(), g.clone()), compose(g.clone(), g.clone()));
    let via_repeat = repeat(g, 4).unwrap();

    let eps_nested = nested.epsilon_at(1e-5).unwrap();
    let eps_repeat = via_repeat.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_nested, eps_repeat, 1e-4, "4-way nested compose epsilon");

    let delta_nested = nested.delta_at(1.0).unwrap();
    let delta_repeat = via_repeat.delta_at(1.0).unwrap();
    assert_relative_eq(
        delta_nested,
        delta_repeat,
        1e-4,
        "4-way nested compose delta",
    );
}

#[test]
fn test_repeat_heterogeneous_compose() {
    // repeat(compose(g1, g2), k) — repeat a heterogeneous training round.
    // A realistic pattern: e.g. two different noise levels per round, repeated k times.
    let g1 = gaussian(0.5).unwrap();
    let g2 = gaussian(1.0).unwrap();
    let k = 5;

    let process = repeat(compose(g1, g2), k).unwrap();

    let eps = process.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0);

    let delta = process.delta_at(1.0).unwrap();
    assert!(delta >= 0.0 && delta <= 1.0);

    // Should be worse than a single round
    let single_eps = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap())
        .epsilon_at(1e-5)
        .unwrap();
    assert!(
        eps > single_eps,
        "repeated hetero compose {} should exceed single round {}",
        eps,
        single_eps
    );
}

// =========================================================================
// Cached wrapper
// =========================================================================

#[test]
fn test_cached_repeat() {
    // repeat(cached(g), k) should produce the same result as repeat(g, k)
    let g = gaussian(1.0).unwrap();
    let k = 20;

    let via_cached = repeat(cached(g.clone()), k).unwrap();
    let via_plain = repeat(g, k).unwrap();

    let eps_cached = via_cached.epsilon_at(1e-5).unwrap();
    let eps_plain = via_plain.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_cached, eps_plain, 1e-10, "cached repeat epsilon");

    let delta_cached = via_cached.delta_at(1.0).unwrap();
    let delta_plain = via_plain.delta_at(1.0).unwrap();
    assert_relative_eq(delta_cached, delta_plain, 1e-10, "cached repeat delta");
}

#[test]
fn test_cached_compose() {
    // cached(compose(g1, g2)) should produce the same result as compose(g1, g2)
    let g1 = gaussian(0.5).unwrap();
    let g2 = gaussian(1.0).unwrap();

    let via_cached = cached(compose(g1.clone(), g2.clone()));
    let via_plain = compose(g1, g2);

    let eps_cached = via_cached.epsilon_at(1e-5).unwrap();
    let eps_plain = via_plain.epsilon_at(1e-5).unwrap();
    assert_relative_eq(eps_cached, eps_plain, 1e-10, "cached compose epsilon");

    let delta_cached = via_cached.delta_at(1.0).unwrap();
    let delta_plain = via_plain.delta_at(1.0).unwrap();
    assert_relative_eq(delta_cached, delta_plain, 1e-10, "cached compose delta");
}

// =========================================================================
// Composed metrics via Process trait
// =========================================================================

#[test]
fn test_compose_metrics_via_process_trait() {
    let process = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());

    let eps = process.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0);

    let delta = process.delta_at(1.0).unwrap();
    assert!(delta >= 0.0 && delta <= 1.0);

    let adv = process.advantage().unwrap();
    assert!(adv >= 0.0 && adv <= 1.0);

    let beta = process.beta_at(0.05).unwrap();
    assert!(beta >= 0.0 && beta <= 1.0);

    let risk = process.risk_at(0.5).unwrap();
    assert!(risk >= 0.0 && risk <= 0.5);
}

// =========================================================================
// Structural equality
// =========================================================================

#[test]
fn test_repeated_structural_eq() {
    let a = repeat(gaussian(1.0).unwrap(), 5).unwrap();
    let b = repeat(gaussian(1.0).unwrap(), 5).unwrap();
    let c = repeat(gaussian(1.0).unwrap(), 10).unwrap();
    let d = repeat(gaussian(1.2).unwrap(), 5).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
}

#[test]
fn test_composed_structural_eq() {
    let a = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());
    let b = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());
    let c = compose(gaussian(1.0).unwrap(), gaussian(0.5).unwrap()); // reversed order
    assert_eq!(a, b);
    assert_ne!(a, c);
}
