use opaque_accounting::numerics::fft::convolve as real_convolve;
use rustfft::{num_complex::Complex64, FftPlanner};
use serde_json::json;
use std::hint::black_box;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static COMPLEX_FFT_PLANNER: OnceLock<Mutex<FftPlanner<f64>>> = OnceLock::new();

fn complex_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let result_len = a.len() + b.len() - 1;
    let fft_len = result_len.next_power_of_two();
    let (forward, inverse) = {
        let planner = COMPLEX_FFT_PLANNER.get_or_init(|| Mutex::new(FftPlanner::new()));
        let mut planner = planner.lock().expect("complex FFT planner lock poisoned");
        (
            planner.plan_fft_forward(fft_len),
            planner.plan_fft_inverse(fft_len),
        )
    };
    let mut a_complex = vec![Complex64::new(0.0, 0.0); fft_len];
    let mut b_complex = vec![Complex64::new(0.0, 0.0); fft_len];
    for (target, value) in a_complex.iter_mut().zip(a) {
        target.re = *value;
    }
    for (target, value) in b_complex.iter_mut().zip(b) {
        target.re = *value;
    }
    forward.process(&mut a_complex);
    forward.process(&mut b_complex);
    for (left, right) in a_complex.iter_mut().zip(b_complex) {
        *left *= right;
    }
    inverse.process(&mut a_complex);
    a_complex
        .into_iter()
        .take(result_len)
        .map(|value| value.re / fft_len as f64)
        .collect()
}

fn deterministic_input(length: usize, phase: f64) -> Vec<f64> {
    (0..length)
        .map(|index| {
            let x = index as f64 + phase;
            (x * 0.017).sin() + 0.5 * (x * 0.031).cos()
        })
        .collect()
}

fn measure<F>(mut operation: F, warmup: usize, repeats: usize) -> Vec<f64>
where
    F: FnMut() -> Vec<f64>,
{
    for _ in 0..warmup {
        black_box(operation());
    }
    (0..repeats)
        .map(|_| {
            let start = Instant::now();
            black_box(operation());
            start.elapsed().as_secs_f64() * 1_000.0
        })
        .collect()
}

fn parse_args() -> (Vec<usize>, usize, usize) {
    let mut sizes = vec![1024, 4096, 16384, 65536];
    let mut warmup = 3;
    let mut repeats = 10;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        // Cargo passes this libtest compatibility flag even for harness-free benches.
        if argument == "--bench" {
            continue;
        }
        let value = args
            .next()
            .unwrap_or_else(|| panic!("missing value for {argument}"));
        match argument.as_str() {
            "--sizes" => {
                sizes = value
                    .split(',')
                    .map(|item| item.parse().expect("sizes must be positive integers"))
                    .collect();
            }
            "--warmup" => warmup = value.parse().expect("warmup must be an integer"),
            "--repeats" => repeats = value.parse().expect("repeats must be an integer"),
            _ => panic!("unknown argument {argument}"),
        }
    }
    assert!(!sizes.is_empty(), "at least one size is required");
    assert!(sizes.iter().all(|size| *size > 0), "sizes must be positive");
    assert!(repeats > 0, "repeats must be positive");
    (sizes, warmup, repeats)
}

fn main() {
    let (sizes, warmup, repeats) = parse_args();
    let mut measurements = Vec::new();
    for length in sizes {
        let a = deterministic_input(length, 0.25);
        let b = deterministic_input(length, 0.75);
        let real_result = real_convolve(&a, &b);
        let complex_result = complex_convolve(&a, &b);
        let max_abs_error = real_result
            .iter()
            .zip(complex_result)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        let real_samples_ms = measure(|| real_convolve(&a, &b), warmup, repeats);
        let complex_samples_ms = measure(|| complex_convolve(&a, &b), warmup, repeats);
        measurements.push(json!({
            "length": length,
            "real_samples_ms": real_samples_ms,
            "complex_samples_ms": complex_samples_ms,
            "max_abs_error": max_abs_error,
        }));
    }
    println!(
        "OPAQUE_BENCHMARK_JSON={}",
        json!({"measurements": measurements})
    );
}
