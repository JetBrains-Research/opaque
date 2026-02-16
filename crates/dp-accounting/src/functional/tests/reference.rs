//! Reference value tests — regression protection with hardcoded expected values
//!
//! These tests compare current implementation against pre-computed reference values
//! generated from the functional API. Any deviation indicates a regression.
//! Tolerance is derived from the PLD's actual discretization: `10 · disc² · k`,
//! where disc is the effective grid spacing and k is the composition count.

use super::*;
use std::sync::LazyLock;

// =========================================================================
// Test configuration
// =========================================================================

// Tolerance is computed from DiscretizationConfig via reference_tol(k).
// See tests/mod.rs for the derivation.

// =========================================================================
// Cached PLDs for reference tests
// =========================================================================

/// Build high-accuracy PLD at disc=1e-4 with 10M max grid size.
/// Used ONLY in coarsening regression tests to measure accuracy loss
/// from adaptive coarsening.
fn build_exact_pld(nm: f64) -> PrivacyLossDistribution {
    let config = DiscretizationConfig::default().with_max_grid_size(10_000_000);
    gaussian_with(nm, config).unwrap().pld().unwrap()
}

/// Cached default PLDs — each nm is built at most once across all tests.
/// This avoids redundant PLD construction for commonly-used noise multipliers.
static DEFAULT_010: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.10).unwrap().pld().unwrap());
static DEFAULT_020: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.20).unwrap().pld().unwrap());
static DEFAULT_030: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.30).unwrap().pld().unwrap());
static DEFAULT_040: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.40).unwrap().pld().unwrap());
static DEFAULT_050: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.50).unwrap().pld().unwrap());
static DEFAULT_080: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(0.80).unwrap().pld().unwrap());
static DEFAULT_100: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(1.00).unwrap().pld().unwrap());
static DEFAULT_120: LazyLock<PrivacyLossDistribution> =
    LazyLock::new(|| gaussian(1.20).unwrap().pld().unwrap());

/// Get cached default PLD for a given noise multiplier.
fn default_pld(nm: f64) -> &'static PrivacyLossDistribution {
    match nm {
        x if (x - 0.10).abs() < 1e-10 => &DEFAULT_010,
        x if (x - 0.20).abs() < 1e-10 => &DEFAULT_020,
        x if (x - 0.30).abs() < 1e-10 => &DEFAULT_030,
        x if (x - 0.40).abs() < 1e-10 => &DEFAULT_040,
        x if (x - 0.50).abs() < 1e-10 => &DEFAULT_050,
        x if (x - 0.80).abs() < 1e-10 => &DEFAULT_080,
        x if (x - 1.00).abs() < 1e-10 => &DEFAULT_100,
        x if (x - 1.20).abs() < 1e-10 => &DEFAULT_120,
        _ => panic!("No cached default_pld for nm={}", nm),
    }
}

/// Cached exact PLDs — each nm is built at most once across all tests.
static EXACT_010: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(0.10));
static EXACT_020: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(0.20));
static EXACT_030: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(0.30));
static EXACT_050: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(0.50));
static EXACT_080: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(0.80));
static EXACT_100: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(1.00));
static EXACT_120: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| build_exact_pld(1.20));

pub fn exact_pld(nm: f64) -> &'static PrivacyLossDistribution {
    match nm {
        x if (x - 0.10).abs() < 1e-10 => &EXACT_010,
        x if (x - 0.20).abs() < 1e-10 => &EXACT_020,
        x if (x - 0.30).abs() < 1e-10 => &EXACT_030,
        x if (x - 0.50).abs() < 1e-10 => &EXACT_050,
        x if (x - 0.80).abs() < 1e-10 => &EXACT_080,
        x if (x - 1.00).abs() < 1e-10 => &EXACT_100,
        x if (x - 1.20).abs() < 1e-10 => &EXACT_120,
        _ => panic!("No cached exact_pld for nm={}", nm),
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Assert close with combined relative + absolute tolerance.
/// Passes if EITHER the relative error < rel_tol OR the absolute error < abs_floor.
/// This prevents false failures on small values where relative error amplifies
/// (e.g. beta_at(0.1) = 1e-4 with absolute diff of 5e-8).
fn assert_rel_close(expected: f64, actual: f64, rel_tol: f64, context: &str) {
    const ABS_FLOOR: f64 = 1e-6; // absolute tolerance floor for small values

    if expected.is_infinite() && actual.is_infinite() && expected.signum() == actual.signum() {
        return;
    }
    let abs_diff = (actual - expected).abs();
    if abs_diff < ABS_FLOOR {
        return; // close enough in absolute terms
    }
    if expected.abs() < 1e-15 {
        // Near zero: only absolute tolerance applies (already checked above)
        panic!(
            "{}: expected={}, actual={}, abs_diff={:.2e}, abs_floor={:.2e}",
            context, expected, actual, abs_diff, ABS_FLOOR
        );
    }
    let rel_err = abs_diff / expected.abs();
    assert!(
        rel_err < rel_tol,
        "{}: expected={}, actual={}, rel_err={:.2e}, tol={:.2e}",
        context,
        expected,
        actual,
        rel_err,
        rel_tol
    );
}

/// Helper: compose two default Gaussians (coarsening enabled).
fn compose_default(nm1: f64, nm2: f64) -> PrivacyLossDistribution {
    compose(gaussian(nm1).unwrap(), gaussian(nm2).unwrap())
        .pld()
        .unwrap()
}

// =========================================================================
// Single mechanism reference tests
// =========================================================================

#[test]
fn test_reference_nm_01_all_metrics() {
    let pld = default_pld(0.1);
    let tol = reference_tol(pld_disc(pld), 1);
    assert_rel_close(
        9.9999939733104815e-1,
        pld.delta_at(0.1),
        tol,
        "nm=0.1 delta_at(0.1)",
    );
    assert_rel_close(
        9.9999905917970211e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.1 delta_at(1.0)",
    );
    assert_rel_close(
        9.9994659771906880e-1,
        pld.delta_at(10.0),
        tol,
        "nm=0.1 delta_at(10.0)",
    );
    assert_rel_close(
        9.6717272356592787e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.1 epsilon_at(1e-6)",
    );
    assert_rel_close(
        9.1817289910055834e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.1 epsilon_at(1e-5)",
    );
    assert_rel_close(
        8.6341354204656014e1,
        pld.epsilon_at(1e-4),
        tol,
        "nm=0.1 epsilon_at(1e-4)",
    );
    assert_rel_close(
        9.9999942669677810e-1,
        pld.advantage(),
        tol,
        "nm=0.1 advantage",
    );
    assert_rel_close(
        1.8960781433818559e-11,
        pld.beta_at(0.01),
        tol,
        "nm=0.1 beta_at(0.01)",
    );
    assert_rel_close(0.0, pld.beta_at(0.1), tol, "nm=0.1 beta_at(0.1)");
    assert_rel_close(0.0, pld.beta_at(0.5), tol, "nm=0.1 beta_at(0.5)");
    assert_rel_close(
        2.6205470231272933e-7,
        pld.risk_at(0.3),
        tol,
        "nm=0.1 risk_at(0.3)",
    );
    assert_rel_close(
        2.8686371175516728e-7,
        pld.risk_at(0.5),
        tol,
        "nm=0.1 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_02_all_metrics() {
    let pld = default_pld(0.2);
    let tol = reference_tol(pld_disc(pld), 1);
    assert_rel_close(
        9.8694602336635318e-1,
        pld.delta_at(0.1),
        tol,
        "nm=0.2 delta_at(0.1)",
    );
    assert_rel_close(
        9.7985167809054097e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.2 delta_at(1.0)",
    );
    assert_rel_close(
        6.1662373040419460e-1,
        pld.delta_at(10.0),
        tol,
        "nm=0.2 delta_at(10.0)",
    );
    assert_rel_close(
        3.5566343754898121e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.2 epsilon_at(1e-6)",
    );
    assert_rel_close(
        3.3103732467509140e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.2 epsilon_at(1e-5)",
    );
    assert_rel_close(
        3.0350305511269369e1,
        pld.epsilon_at(1e-4),
        tol,
        "nm=0.2 epsilon_at(1e-4)",
    );
    assert_rel_close(
        9.8758066934888233e-1,
        pld.advantage(),
        tol,
        "nm=0.2 advantage",
    );
    assert_rel_close(
        3.7515128217800726e-3,
        pld.beta_at(0.01),
        tol,
        "nm=0.2 beta_at(0.01)",
    );
    assert_rel_close(
        1.0022581954784625e-4,
        pld.beta_at(0.1),
        tol,
        "nm=0.2 beta_at(0.1)",
    );
    assert_rel_close(
        2.8737233066943496e-7,
        pld.beta_at(0.5),
        tol,
        "nm=0.2 beta_at(0.5)",
    );
    assert_rel_close(
        5.6257111962332799e-3,
        pld.risk_at(0.3),
        tol,
        "nm=0.2 risk_at(0.3)",
    );
    assert_rel_close(
        6.2096660462339377e-3,
        pld.risk_at(0.5),
        tol,
        "nm=0.2 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_03_all_metrics() {
    let pld = default_pld(0.3);
    let tol = reference_tol(pld_disc(pld), 1);
    assert_rel_close(
        8.9955002730602296e-1,
        pld.delta_at(0.1),
        tol,
        "nm=0.3 delta_at(0.1)",
    );
    assert_rel_close(
        8.4723593340447512e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.3 delta_at(1.0)",
    );
    assert_rel_close(
        5.7496922270531221e-2,
        pld.delta_at(10.0),
        tol,
        "nm=0.3 delta_at(10.0)",
    );
    assert_rel_close(
        2.0781222266740006e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.3 epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.9130767847851011e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.3 epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.7284171878630151e1,
        pld.epsilon_at(1e-4),
        tol,
        "nm=0.3 epsilon_at(1e-4)",
    );
    assert_rel_close(
        9.0441929545049826e-1,
        pld.advantage(),
        tol,
        "nm=0.3 advantage",
    );
    assert_rel_close(
        1.5697091451178921e-1,
        pld.beta_at(0.01),
        tol,
        "nm=0.3 beta_at(0.01)",
    );
    assert_rel_close(
        2.0095442253929577e-2,
        pld.beta_at(0.1),
        tol,
        "nm=0.3 beta_at(0.1)",
    );
    assert_rel_close(
        4.2906376752999093e-4,
        pld.beta_at(0.5),
        tol,
        "nm=0.3 beta_at(0.5)",
    );
    assert_rel_close(
        4.2833874718477584e-2,
        pld.risk_at(0.3),
        tol,
        "nm=0.3 risk_at(0.3)",
    );
    assert_rel_close(
        4.7790355696459841e-2,
        pld.risk_at(0.5),
        tol,
        "nm=0.3 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_05_all_metrics() {
    let pld = default_pld(0.5);
    let tol = reference_tol(pld_disc(pld), 1);
    // delta_at
    assert_rel_close(
        6.666395155003334e-1,
        pld.delta_at(0.1),
        tol,
        "nm=0.5 delta_at(0.1)",
    );
    assert_rel_close(
        5.098616600467871e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.5 delta_at(1.0)",
    );
    assert_rel_close(
        1.8381307654979573e-1,
        pld.delta_at(3.0),
        tol,
        "nm=0.5 delta_at(3.0)",
    );
    // epsilon_at
    assert_rel_close(
        1.0997151218459818e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.5 epsilon_at(1e-6)",
    );
    assert_rel_close(
        9.997256150316934e0,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.5 epsilon_at(1e-5)",
    );
    assert_rel_close(
        8.876869466738532e0,
        pld.epsilon_at(1e-4),
        tol,
        "nm=0.5 epsilon_at(1e-4)",
    );
    // advantage
    assert_rel_close(
        6.826894921100117e-1,
        pld.advantage(),
        tol,
        "nm=0.5 advantage",
    );
    // beta_at
    assert_rel_close(
        6.279204212419744e-1,
        pld.beta_at(0.01),
        tol,
        "nm=0.5 beta_at(0.01)",
    );
    assert_rel_close(
        2.3624059855263252e-1,
        pld.beta_at(0.1),
        tol,
        "nm=0.5 beta_at(0.1)",
    );
    assert_rel_close(
        2.2750207177397674e-2,
        pld.beta_at(0.5),
        tol,
        "nm=0.5 beta_at(0.5)",
    );
    // risk_at
    assert_rel_close(
        1.3874859623081026e-1,
        pld.risk_at(0.3),
        tol,
        "nm=0.5 risk_at(0.3)",
    );
    assert_rel_close(
        1.5865532020761086e-1,
        pld.risk_at(0.5),
        tol,
        "nm=0.5 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_08_all_metrics() {
    let pld = default_pld(0.8);
    let tol = reference_tol(pld_disc(pld), 1);
    // delta_at
    assert_rel_close(
        4.4143449613397745e-1,
        pld.delta_at(0.1),
        tol,
        "nm=0.8 delta_at(0.1)",
    );
    assert_rel_close(
        3.3728097145040054e-1,
        pld.delta_at(0.5),
        tol,
        "nm=0.8 delta_at(0.5)",
    );
    assert_rel_close(
        2.2101845753930838e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.8 delta_at(1.0)",
    );
    // epsilon_at
    assert_rel_close(
        6.312060191475913e0,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.8 epsilon_at(1e-6)",
    );
    assert_rel_close(
        5.679586857619628e0,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.8 epsilon_at(1e-5)",
    );
    assert_rel_close(
        4.969367920306141e0,
        pld.epsilon_at(1e-4),
        tol,
        "nm=0.8 epsilon_at(1e-4)",
    );
    // advantage
    assert_rel_close(
        4.6802894190267996e-1,
        pld.advantage(),
        tol,
        "nm=0.8 advantage",
    );
    // beta_at
    assert_rel_close(
        8.59114495800096e-1,
        pld.beta_at(0.01),
        tol,
        "nm=0.8 beta_at(0.01)",
    );
    assert_rel_close(
        5.125852834921053e-1,
        pld.beta_at(0.1),
        tol,
        "nm=0.8 beta_at(0.1)",
    );
    assert_rel_close(
        1.0564982605065638e-1,
        pld.beta_at(0.5),
        tol,
        "nm=0.8 beta_at(0.5)",
    );
    // risk_at
    assert_rel_close(
        2.237414145575775e-1,
        pld.risk_at(0.3),
        tol,
        "nm=0.8 risk_at(0.3)",
    );
    assert_rel_close(
        2.659855641439832e-1,
        pld.risk_at(0.5),
        tol,
        "nm=0.8 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_10_all_metrics() {
    let pld = default_pld(1.0);
    let tol = reference_tol(pld_disc(pld), 1);
    assert_rel_close(
        3.5232517168142130e-1,
        pld.delta_at(0.1),
        tol,
        "nm=1.0 delta_at(0.1)",
    );
    assert_rel_close(
        1.2693673749873599e-1,
        pld.delta_at(1.0),
        tol,
        "nm=1.0 delta_at(1.0)",
    );
    assert_rel_close(
        1.1151320300163309e-19,
        pld.delta_at(10.0),
        tol,
        "nm=1.0 delta_at(10.0)",
    );
    assert_rel_close(
        4.8865541244382156e0,
        pld.epsilon_at(1e-6),
        tol,
        "nm=1.0 epsilon_at(1e-6)",
    );
    assert_rel_close(
        4.3771781000359153e0,
        pld.epsilon_at(1e-5),
        tol,
        "nm=1.0 epsilon_at(1e-5)",
    );
    assert_rel_close(
        3.8044359145590914e0,
        pld.epsilon_at(1e-4),
        tol,
        "nm=1.0 epsilon_at(1e-4)",
    );
    assert_rel_close(
        3.8292492254809146e-1,
        pld.advantage(),
        tol,
        "nm=1.0 advantage",
    );
    assert_rel_close(
        9.0763789583839138e-1,
        pld.beta_at(0.01),
        tol,
        "nm=1.0 beta_at(0.01)",
    );
    assert_rel_close(
        6.1085636973773683e-1,
        pld.beta_at(0.1),
        tol,
        "nm=1.0 beta_at(0.1)",
    );
    assert_rel_close(
        1.5865528479488453e-1,
        pld.beta_at(0.5),
        tol,
        "nm=1.0 beta_at(0.5)",
    );
    assert_rel_close(
        2.5300439807055680e-1,
        pld.risk_at(0.3),
        tol,
        "nm=1.0 risk_at(0.3)",
    );
    assert_rel_close(
        3.0853755752796375e-1,
        pld.risk_at(0.5),
        tol,
        "nm=1.0 risk_at(0.5)",
    );
}

#[test]
fn test_reference_nm_12_all_metrics() {
    let pld = gaussian(1.2).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);
    assert_rel_close(
        2.8978636566875204e-1,
        pld.delta_at(0.1),
        tol,
        "nm=1.2 delta_at(0.1)",
    );
    assert_rel_close(
        7.2714373313958289e-2,
        pld.delta_at(1.0),
        tol,
        "nm=1.2 delta_at(1.0)",
    );
    assert_rel_close(
        1.5668935655211428e-14,
        pld.delta_at(10.0),
        tol,
        "nm=1.2 delta_at(10.0)",
    );
    assert_rel_close(
        3.9756951884166885e0,
        pld.epsilon_at(1e-6),
        tol,
        "nm=1.2 epsilon_at(1e-6)",
    );
    assert_rel_close(
        3.5486966824274755e0,
        pld.epsilon_at(1e-5),
        tol,
        "nm=1.2 epsilon_at(1e-5)",
    );
    assert_rel_close(
        3.0679959175801583e0,
        pld.epsilon_at(1e-4),
        tol,
        "nm=1.2 epsilon_at(1e-4)",
    );
    assert_rel_close(
        3.2307776097868246e-1,
        pld.advantage(),
        tol,
        "nm=1.2 advantage",
    );
    assert_rel_close(
        9.3228337678793616e-1,
        pld.beta_at(0.01),
        tol,
        "nm=1.2 beta_at(0.01)",
    );
    assert_rel_close(
        6.7300218046647309e-1,
        pld.beta_at(0.1),
        tol,
        "nm=1.2 beta_at(0.1)",
    );
    assert_rel_close(
        2.0232839934892291e-1,
        pld.beta_at(0.5),
        tol,
        "nm=1.2 beta_at(0.5)",
    );
    assert_rel_close(
        2.7084098550960700e-1,
        pld.risk_at(0.3),
        tol,
        "nm=1.2 risk_at(0.3)",
    );
    assert_rel_close(
        3.3846113012729762e-1,
        pld.risk_at(0.5),
        tol,
        "nm=1.2 risk_at(0.5)",
    );
}

// =========================================================================
// Self-composition reference tests
// =========================================================================

#[test]
fn test_reference_compose_nm05_k10() {
    let pld = repeat(gaussian(0.5).unwrap(), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);
    // epsilon_at
    assert_rel_close(
        4.9319276677140081e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.5 k=10 epsilon_at(1e-6)",
    );
    assert_rel_close(
        4.6211210204864408e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.5 k=10 epsilon_at(1e-5)",
    );
    assert_rel_close(
        3.8732876548550720e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=0.5 k=10 epsilon_at(1e-3)",
    );
    // delta_at
    assert_rel_close(
        9.9744693136867224e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.5 k=10 delta_at(1.0)",
    );
    assert_rel_close(
        9.8541636969053548e-1,
        pld.delta_at(5.0),
        tol,
        "nm=0.5 k=10 delta_at(5.0)",
    );
}

#[test]
fn test_reference_compose_nm08_k10() {
    let pld = repeat(gaussian(0.8).unwrap(), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);
    // epsilon_at
    assert_rel_close(
        2.5947734982231989e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.8 k=10 epsilon_at(1e-6)",
    );
    assert_rel_close(
        2.3995359005258440e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.8 k=10 epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.9293042507202195e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=0.8 k=10 epsilon_at(1e-3)",
    );
    // delta_at
    assert_rel_close(
        9.2254760175917327e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.8 k=10 delta_at(1.0)",
    );
    assert_rel_close(
        6.7333199952109424e-1,
        pld.delta_at(5.0),
        tol,
        "nm=0.8 k=10 delta_at(5.0)",
    );
}

#[test]
fn test_reference_compose_nm05_k50() {
    let pld = repeat(gaussian(0.5).unwrap(), 50).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 50);
    // epsilon_at
    assert_rel_close(
        1.6636223011249317e2,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.5 k=50 epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.5944148636511122e2,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.5 k=50 epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.4279902366226918e2,
        pld.epsilon_at(1e-3),
        tol,
        "nm=0.5 k=50 epsilon_at(1e-3)",
    );
    // delta_at — saturates to 1.0 after 50 compositions (clamped from ~1.000003)
    assert_rel_close(1.0, pld.delta_at(1.0), tol, "nm=0.5 k=50 delta_at(1.0)");
    assert_rel_close(1.0, pld.delta_at(5.0), tol, "nm=0.5 k=50 delta_at(5.0)");
}

#[test]
fn test_reference_compose_nm08_k50() {
    let pld = repeat(gaussian(0.8).unwrap(), 50).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 50);
    assert_rel_close(
        8.0278334633970431e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=0.8 k=50 epsilon_at(1e-6)",
    );
    assert_rel_close(
        7.5944603619254806e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=0.8 k=50 epsilon_at(1e-5)",
    );
    assert_rel_close(
        6.5520860087695013e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=0.8 k=50 epsilon_at(1e-3)",
    );
    assert_rel_close(
        9.9998379640879642e-1,
        pld.delta_at(1.0),
        tol,
        "nm=0.8 k=50 delta_at(1.0)",
    );
    assert_rel_close(
        9.9989589709362847e-1,
        pld.delta_at(5.0),
        tol,
        "nm=0.8 k=50 delta_at(5.0)",
    );
}

#[test]
fn test_reference_compose_nm10_k10() {
    let pld = repeat(gaussian(1.0).unwrap(), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);
    assert_rel_close(
        1.9423656496201026e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=1.0 k=10 epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.7856586849123278e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=1.0 k=10 epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.4079350606144322e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=1.0 k=10 epsilon_at(1e-3)",
    );
    assert_rel_close(
        8.1851782693487862e-1,
        pld.delta_at(1.0),
        tol,
        "nm=1.0 k=10 delta_at(1.0)",
    );
    assert_rel_close(
        3.8383685427530123e-1,
        pld.delta_at(5.0),
        tol,
        "nm=1.0 k=10 delta_at(5.0)",
    );
}

#[test]
fn test_reference_compose_nm10_k50() {
    let pld = repeat(gaussian(1.0).unwrap(), 50).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 50);
    assert_rel_close(
        5.7848549575799808e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=1.0 k=50 epsilon_at(1e-6)",
    );
    assert_rel_close(
        5.4376639087589389e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=1.0 k=50 epsilon_at(1e-5)",
    );
    assert_rel_close(
        4.6024048138255999e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=1.0 k=50 epsilon_at(1e-3)",
    );
    assert_rel_close(
        9.9933593563886181e-1,
        pld.delta_at(1.0),
        tol,
        "nm=1.0 k=50 delta_at(1.0)",
    );
    assert_rel_close(
        9.9602281396036041e-1,
        pld.delta_at(5.0),
        tol,
        "nm=1.0 k=50 delta_at(5.0)",
    );
}

#[test]
fn test_reference_compose_nm10_k100() {
    let pld = repeat(gaussian(1.0).unwrap(), 100).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 100);
    assert_rel_close(
        9.6717273363118679e1,
        pld.epsilon_at(1e-6),
        tol,
        "nm=1.0 k=100 epsilon_at(1e-6)",
    );
    assert_rel_close(
        9.1817289888109073e1,
        pld.epsilon_at(1e-5),
        tol,
        "nm=1.0 k=100 epsilon_at(1e-5)",
    );
    assert_rel_close(
        8.0032526213812432e1,
        pld.epsilon_at(1e-3),
        tol,
        "nm=1.0 k=100 epsilon_at(1e-3)",
    );
    assert_rel_close(
        9.9999905925612209e-1,
        pld.delta_at(1.0),
        tol,
        "nm=1.0 k=100 delta_at(1.0)",
    );
    assert_rel_close(
        9.9999378410402451e-1,
        pld.delta_at(5.0),
        tol,
        "nm=1.0 k=100 delta_at(5.0)",
    );
}

// =========================================================================
// Heterogeneous composition reference tests
// =========================================================================

#[test]
fn test_reference_hetero_05_08() {
    let pld = compose_default(0.5, 0.8);
    let tol = reference_tol(pld_disc(&pld), 2);
    assert_rel_close(
        1.3446950231148069e1,
        pld.epsilon_at(1e-6),
        tol,
        "hetero(0.5,0.8) epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.2271951026704846e1,
        pld.epsilon_at(1e-5),
        tol,
        "hetero(0.5,0.8) epsilon_at(1e-5)",
    );
    assert_rel_close(
        9.4357598775594234e0,
        pld.epsilon_at(1e-3),
        tol,
        "hetero(0.5,0.8) epsilon_at(1e-3)",
    );
    assert_rel_close(
        6.9797079187003663e-1,
        pld.delta_at(0.5),
        tol,
        "hetero(0.5,0.8) delta_at(0.5)",
    );
    assert_rel_close(
        6.2696654552997932e-1,
        pld.delta_at(1.0),
        tol,
        "hetero(0.5,0.8) delta_at(1.0)",
    );
    assert_rel_close(
        1.0147680533293510e-1,
        pld.delta_at(5.0),
        tol,
        "hetero(0.5,0.8) delta_at(5.0)",
    );
    assert_rel_close(
        7.6170041960908896e-1,
        pld.advantage(),
        tol,
        "hetero(0.5,0.8) advantage",
    );
}

#[test]
fn test_reference_hetero_03_08() {
    let pld = compose_default(0.3, 0.8);
    let tol = reference_tol(pld_disc(&pld), 2);
    assert_rel_close(
        2.2626194273497436e1,
        pld.epsilon_at(1e-6),
        tol,
        "hetero(0.3,0.8) epsilon_at(1e-6)",
    );
    assert_rel_close(
        2.0865261689859995e1,
        pld.epsilon_at(1e-5),
        tol,
        "hetero(0.3,0.8) epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.6622614308966746e1,
        pld.epsilon_at(1e-3),
        tol,
        "hetero(0.3,0.8) epsilon_at(1e-3)",
    );
    assert_rel_close(
        9.0427492528920550e-1,
        pld.delta_at(0.5),
        tol,
        "hetero(0.3,0.8) delta_at(0.5)",
    );
    assert_rel_close(
        8.7964502030243163e-1,
        pld.delta_at(1.0),
        tol,
        "hetero(0.3,0.8) delta_at(1.0)",
    );
    assert_rel_close(
        5.3875351786779724e-1,
        pld.delta_at(5.0),
        tol,
        "hetero(0.3,0.8) delta_at(5.0)",
    );
    assert_rel_close(
        9.2492416928574084e-1,
        pld.advantage(),
        tol,
        "hetero(0.3,0.8) advantage",
    );
}

#[test]
fn test_reference_hetero_05_10() {
    let pld = compose_default(0.5, 1.0);
    let tol = reference_tol(pld_disc(&pld), 2);
    assert_rel_close(
        1.2595246742067737e1,
        pld.epsilon_at(1e-6),
        tol,
        "hetero(0.5,1.0) epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.1480022814157641e1,
        pld.epsilon_at(1e-5),
        tol,
        "hetero(0.5,1.0) epsilon_at(1e-5)",
    );
    assert_rel_close(
        8.7872832659402551e0,
        pld.epsilon_at(1e-3),
        tol,
        "hetero(0.5,1.0) epsilon_at(1e-3)",
    );
    assert_rel_close(
        6.6630540889412970e-1,
        pld.delta_at(0.5),
        tol,
        "hetero(0.5,1.0) delta_at(0.5)",
    );
    assert_rel_close(
        5.8909966626910615e-1,
        pld.delta_at(1.0),
        tol,
        "hetero(0.5,1.0) delta_at(1.0)",
    );
    assert_rel_close(
        7.2690722196340971e-2,
        pld.delta_at(5.0),
        tol,
        "hetero(0.5,1.0) delta_at(5.0)",
    );
    assert_rel_close(
        7.3644752284177073e-1,
        pld.advantage(),
        tol,
        "hetero(0.5,1.0) advantage",
    );
}

#[test]
fn test_reference_hetero_04_12() {
    let pld = compose_default(0.4, 1.2);
    let tol = reference_tol(pld_disc(&pld), 2);
    assert_rel_close(
        1.5429482064631182e1,
        pld.epsilon_at(1e-6),
        tol,
        "hetero(0.4,1.2) epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.4119434288493995e1,
        pld.epsilon_at(1e-5),
        tol,
        "hetero(0.4,1.2) epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.0959129567200877e1,
        pld.epsilon_at(1e-3),
        tol,
        "hetero(0.4,1.2) epsilon_at(1e-3)",
    );
    assert_rel_close(
        7.6172941154196905e-1,
        pld.delta_at(0.5),
        tol,
        "hetero(0.4,1.2) delta_at(0.5)",
    );
    assert_rel_close(
        7.0402684525417780e-1,
        pld.delta_at(1.0),
        tol,
        "hetero(0.4,1.2) delta_at(1.0)",
    );
    assert_rel_close(
        1.8423691409705678e-1,
        pld.delta_at(5.0),
        tol,
        "hetero(0.4,1.2) delta_at(5.0)",
    );
    assert_rel_close(
        8.1236767098011653e-1,
        pld.advantage(),
        tol,
        "hetero(0.4,1.2) advantage",
    );
}

#[test]
fn test_reference_hetero_3way_04_08_12() {
    // Three-way: compose(compose(0.4, 0.8), 1.2)
    let pld = compose(
        compose(gaussian(0.4).unwrap(), gaussian(0.8).unwrap()),
        gaussian(1.2).unwrap(),
    )
    .pld()
    .unwrap();
    let tol = reference_tol(pld_disc(&pld), 3);

    assert_rel_close(
        1.7526795491859911e1,
        pld.epsilon_at(1e-6),
        tol,
        "3way(0.4,0.8,1.2) epsilon_at(1e-6)",
    );
    assert_rel_close(
        1.6079478913248700e1,
        pld.epsilon_at(1e-5),
        tol,
        "3way(0.4,0.8,1.2) epsilon_at(1e-5)",
    );
    assert_rel_close(
        1.2589701700410123e1,
        pld.epsilon_at(1e-3),
        tol,
        "3way(0.4,0.8,1.2) epsilon_at(1e-3)",
    );
    assert_rel_close(
        8.1590195250256914e-1,
        pld.delta_at(0.5),
        tol,
        "3way(0.4,0.8,1.2) delta_at(0.5)",
    );
    assert_rel_close(
        7.7026327285406559e-1,
        pld.delta_at(1.0),
        tol,
        "3way(0.4,0.8,1.2) delta_at(1.0)",
    );
    assert_rel_close(
        2.8688947234388201e-1,
        pld.delta_at(5.0),
        tol,
        "3way(0.4,0.8,1.2) delta_at(5.0)",
    );
    assert_rel_close(
        8.5525131565912194e-1,
        pld.advantage(),
        tol,
        "3way(0.4,0.8,1.2) advantage",
    );
}
