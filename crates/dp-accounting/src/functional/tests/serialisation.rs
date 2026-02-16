//! Serde round-trip tests for all functional API types
//!
//! Verifies that every public type in the functional API can be serialized to
//! JSON and deserialized back with full fidelity. Uses `serde_json` which is
//! already in dev-dependencies.

use crate::functional::calibrate::{
    target_advantage, target_beta_at, target_delta_at, target_epsilon_at, target_risk_at,
    CalibrateConfig, CalibrateResult, Target,
};
use crate::functional::*;

/// Round-trip helper: serialize to JSON then deserialize back
fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq>(
    value: &T,
    label: &str,
) -> T {
    let json = serde_json::to_string(value)
        .unwrap_or_else(|e| panic!("{}: serialize failed: {}", label, e));
    let restored: T = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("{}: deserialize failed: {}\njson: {}", label, e, json));
    assert_eq!(value, &restored, "{}: round-trip mismatch", label);
    restored
}

// =========================================================================
// Leaf mechanisms
// =========================================================================

#[test]
fn test_gaussian_roundtrip() {
    let g = gaussian(1.1).unwrap();
    roundtrip(&g, "Gaussian");
}

#[test]
fn test_gaussian_custom_config_roundtrip() {
    let g = gaussian_with(1.1, DiscretizationConfig::new(1e-3, -30.0).unwrap()).unwrap();
    roundtrip(&g, "Gaussian custom config");
}

#[test]
fn test_adaclip_gaussian_roundtrip() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    roundtrip(&ac, "AdaClip<Gaussian>");
}

#[test]
fn test_eps_delta_roundtrip() {
    let ed = eps_delta(1.0, 1e-5).unwrap();
    roundtrip(&ed, "EpsDelta");
}

#[test]
fn test_identity_roundtrip() {
    let id = identity();
    roundtrip(&id, "Identity");
}

// =========================================================================
// Config & enum types
// =========================================================================

#[test]
fn test_discretization_config_roundtrip() {
    let config = DiscretizationConfig::default();
    roundtrip(&config, "DiscretizationConfig default");

    let custom = DiscretizationConfig::new(1e-3, -30.0).unwrap();
    roundtrip(&custom, "DiscretizationConfig custom");
}

#[test]
fn test_adjacency_roundtrip() {
    use crate::functional::adjacency::Adjacency;
    for adj in [Adjacency::Remove, Adjacency::Add, Adjacency::Replace] {
        roundtrip(&adj, &format!("Adjacency::{:?}", adj));
    }
}

// =========================================================================
// Composition types
// =========================================================================

#[test]
fn test_repeated_roundtrip() {
    let r = repeat(gaussian(1.0).unwrap(), 100).unwrap();
    roundtrip(&r, "Repeated<Gaussian>");
}

#[test]
fn test_composed_roundtrip() {
    let c = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());
    roundtrip(&c, "Composed<Gaussian, Gaussian>");
}

#[test]
fn test_composed_heterogeneous_roundtrip() {
    let c = compose(gaussian(1.0).unwrap(), eps_delta(0.5, 1e-6).unwrap());
    roundtrip(&c, "Composed<Gaussian, EpsDelta>");
}

// =========================================================================
// Amplification types
// =========================================================================

#[test]
fn test_poisson_roundtrip() {
    let p = poisson(gaussian(1.1).unwrap(), 0.01);
    roundtrip(&p, "Poisson<Gaussian>");
}

#[test]
fn test_truncated_poisson_roundtrip() {
    let tp = truncated_poisson(gaussian(4.0).unwrap(), 0.001, 1024, 1_000_000);
    roundtrip(&tp, "TruncatedPoisson<Gaussian>");
}

#[test]
fn test_accumulated_roundtrip() {
    let acc = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap();
    roundtrip(&acc, "Accumulated<Gaussian>");
}

#[test]
fn test_evidence_roundtrip() {
    roundtrip(
        &TightGaussianPoissonEvidence,
        "TightGaussianPoissonEvidence",
    );
    roundtrip(
        &TightGaussianTruncatedPoissonEvidence,
        "TightGaussianTruncatedPoissonEvidence",
    );
    roundtrip(
        &TightGaussianAccumulateEvidence,
        "TightGaussianAccumulateEvidence",
    );
}

// =========================================================================
// Calibration types
// =========================================================================

#[test]
fn test_target_roundtrip() {
    let targets: Vec<Target> = vec![
        target_epsilon_at(1.0, 1e-5),
        target_delta_at(1e-5, 1.0),
        target_advantage(0.5),
        target_beta_at(0.9, 0.05),
        target_risk_at(0.3, 0.5),
    ];
    for (i, t) in targets.iter().enumerate() {
        // Target derives Copy but not PartialEq, so test via JSON equality
        let json1 = serde_json::to_string(t).unwrap();
        let restored: Target = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&restored).unwrap();
        assert_eq!(json1, json2, "Target[{}] round-trip JSON mismatch", i);
    }
}

#[test]
fn test_calibrate_config_roundtrip() {
    let config = CalibrateConfig::default()
        .with_bounds(0.1, 50.0)
        .with_tolerance(1e-4)
        .with_max_iterations(200);
    // CalibrateConfig doesn't derive PartialEq, so compare fields
    let json = serde_json::to_string(&config).unwrap();
    let restored: CalibrateConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.param_min, restored.param_min);
    assert_eq!(config.param_max, restored.param_max);
    assert_eq!(config.tolerance, restored.tolerance);
    assert_eq!(config.max_iterations, restored.max_iterations);
}

#[test]
fn test_calibrate_result_roundtrip() {
    let result = CalibrateResult {
        param: 1.234,
        achieved: 0.987,
        converged: true,
        evaluations: 42,
    };
    let json = serde_json::to_string(&result).unwrap();
    let restored: CalibrateResult = serde_json::from_str(&json).unwrap();
    assert_eq!(result.param, restored.param);
    assert_eq!(result.achieved, restored.achieved);
    assert_eq!(result.converged, restored.converged);
    assert_eq!(result.evaluations, restored.evaluations);
}

// =========================================================================
// Cached (transparent serialization)
// =========================================================================

#[test]
fn test_cached_serializes_as_inner() {
    let g = gaussian(1.1).unwrap();
    let c = cached(g.clone());

    let json_inner = serde_json::to_string(&g).unwrap();
    let json_cached = serde_json::to_string(&c).unwrap();
    assert_eq!(
        json_inner, json_cached,
        "Cached should serialize identically to inner"
    );
}

#[test]
fn test_cached_deserializes_with_empty_cache() {
    let c = cached(gaussian(1.1).unwrap());

    // Populate the cache
    let _ = c.pld().unwrap();

    // Round-trip
    let json = serde_json::to_string(&c).unwrap();
    let restored: Cached<Gaussian> = serde_json::from_str(&json).unwrap();

    // Restored should be equal (PartialEq ignores cache)
    assert_eq!(c, restored);
}

// =========================================================================
// Nested / complex types
// =========================================================================

#[test]
fn test_nested_repeat_poisson_roundtrip() {
    let process = repeat(poisson(gaussian(4.0).unwrap(), 0.001), 1000).unwrap();
    roundtrip(&process, "Repeated<Poisson<Gaussian>>");
}

#[test]
fn test_nested_cached_poisson_roundtrip() {
    let process = cached(poisson(gaussian(4.0).unwrap(), 0.001));
    let json = serde_json::to_string(&process).unwrap();
    let restored: Cached<Poisson<Gaussian, TightGaussianPoissonEvidence>> =
        serde_json::from_str(&json).unwrap();
    assert_eq!(process, restored);
}

#[test]
fn test_nested_repeat_accumulated_roundtrip() {
    let process = repeat(
        accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap(),
        1000,
    )
    .unwrap();
    roundtrip(&process, "Repeated<Accumulated<Gaussian>>");
}

#[test]
fn test_nested_repeat_cached_truncated_roundtrip() {
    let process = repeat(
        cached(truncated_poisson(
            gaussian(4.0).unwrap(),
            0.001,
            1024,
            1_000_000,
        )),
        100,
    )
    .unwrap();
    let json = serde_json::to_string(&process).unwrap();
    let restored: Repeated<
        Cached<TruncatedPoisson<Gaussian, TightGaussianTruncatedPoissonEvidence>>,
    > = serde_json::from_str(&json).unwrap();
    assert_eq!(process, restored);
}

// =========================================================================
// Metric preservation after round-trip
// =========================================================================

#[test]
fn test_metric_preservation_after_roundtrip() {
    let original = poisson(gaussian(1.0).unwrap(), 0.001);
    let json = serde_json::to_string(&original).unwrap();
    let restored: Poisson<Gaussian, TightGaussianPoissonEvidence> =
        serde_json::from_str(&json).unwrap();

    let eps_orig = original.epsilon_at(1e-5).unwrap();
    let eps_rest = restored.epsilon_at(1e-5).unwrap();
    assert_eq!(
        eps_orig, eps_rest,
        "epsilon_at should be identical after round-trip"
    );

    let delta_orig = original.delta_at(1.0).unwrap();
    let delta_rest = restored.delta_at(1.0).unwrap();
    assert_eq!(
        delta_orig, delta_rest,
        "delta_at should be identical after round-trip"
    );
}

// =========================================================================
// JSON format spot-checks
// =========================================================================

#[test]
fn test_gaussian_json_format() {
    let g = gaussian(1.1).unwrap();
    let json: serde_json::Value = serde_json::to_value(&g).unwrap();
    assert_eq!(json["noise_multiplier"], 1.1);
    assert!(json["config"]["discretization"].is_number());
}

#[test]
fn test_repeated_json_format() {
    let r = repeat(gaussian(1.0).unwrap(), 100).unwrap();
    let json: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(json["count"], 100);
    assert!(json["inner"]["noise_multiplier"].is_number());
}

#[test]
fn test_poisson_json_format() {
    let p = poisson(gaussian(4.0).unwrap(), 0.001);
    let json: serde_json::Value = serde_json::to_value(&p).unwrap();
    assert_eq!(json["rate"], 0.001);
    assert!(json["inner"]["noise_multiplier"].is_number());
    // Evidence is a unit struct — serializes as null
    assert!(json["evidence"].is_null());
}

#[test]
fn test_accumulated_json_format() {
    let acc = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap();
    let json: serde_json::Value = serde_json::to_value(&acc).unwrap();
    assert_eq!(json["microbatches"], 8);
    assert_eq!(json["inner"]["rate"], 0.00064);
    assert!(json["inner"]["inner"]["noise_multiplier"].is_number());
    // Evidence is a unit struct — serializes as null
    assert!(json["evidence"].is_null());
}

#[test]
fn test_truncated_poisson_json_format() {
    let tp = truncated_poisson(gaussian(4.0).unwrap(), 0.001, 1024, 1_000_000);
    let json: serde_json::Value = serde_json::to_value(&tp).unwrap();
    assert_eq!(json["rate"], 0.001);
    assert_eq!(json["batch_size_max"], 1024);
    assert_eq!(json["dataset_size"], 1_000_000);
    assert!(json["inner"]["noise_multiplier"].is_number());
}

#[test]
fn test_target_json_format() {
    let t = target_epsilon_at(1.0, 1e-5);
    let json: serde_json::Value = serde_json::to_value(&t).unwrap();
    // Externally tagged enum: {"Epsilon": {"bound": 1.0, "delta": 1e-5}}
    assert!(json["Epsilon"].is_object());
    assert_eq!(json["Epsilon"]["bound"], 1.0);
    assert_eq!(json["Epsilon"]["delta"], 1e-5);
}
