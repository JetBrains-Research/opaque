//! Smoke tests - basic end-to-end sanity checks
//!
//! These tests verify that the basic API works and produces reasonable results.

use super::*;

#[test]
fn test_basic_gaussian_workflow() {
    // Create a mechanism
    let gauss = gaussian(1.1).unwrap();

    // Compute PLD
    let pld = gauss.pld().expect("PLD construction should succeed");

    // Query privacy metrics
    let delta = pld.delta_at(1.0);
    assert!(delta > 0.0 && delta < 1.0, "delta should be in (0, 1)");

    let epsilon = pld.epsilon_at(1e-5);
    assert!(
        epsilon > 0.0 && epsilon < 100.0,
        "epsilon should be reasonable"
    );

    let advantage = pld.advantage();
    assert!(
        advantage >= 0.0 && advantage <= 1.0,
        "advantage should be in [0, 1]"
    );
}

#[test]
fn test_process_trait_methods() {
    let gauss = gaussian(1.0).unwrap();

    // Process trait should delegate to PLD methods
    let epsilon_direct = gauss.pld().unwrap().epsilon_at(1e-5);
    let epsilon_trait = gauss.epsilon_at(1e-5).expect("epsilon_at should work");

    assert!((epsilon_direct - epsilon_trait).abs() < 1e-10);
}

#[test]
fn test_composition_increases_privacy_loss() {
    let g = gaussian(1.0).unwrap();

    // Composed process should have worse privacy (larger epsilon for same delta)
    let eps1 = g.epsilon_at(1e-5).unwrap();
    let eps2 = repeat(g, 2).unwrap().epsilon_at(1e-5).unwrap();

    assert!(eps2 > eps1, "Composition should increase epsilon");
}
