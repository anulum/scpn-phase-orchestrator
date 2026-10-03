// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Actual phase quality benchmark

//! Compare direct Rust score/mask costs on the public benchmark's exact inputs.

use spo_oscillators::quality::PhaseQualityScorer;
use std::hint::black_box;
use std::time::Instant;

/// Return five mean call durations after two real warm-up calls.
///
/// # Arguments
///
/// * `operation` - Real public scorer operation, with inputs behind black_box.
///
/// # Returns
///
/// Five seconds-per-call samples, each measured over 1000 actual calls.
fn samples<T>(mut operation: impl FnMut() -> T) -> Vec<f64> {
    for _ in 0..2 {
        black_box(operation());
    }
    (0..5)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..1000 {
                black_box(operation());
            }
            start.elapsed().as_secs_f64() / 1000.0
        })
        .collect()
}

/// Validate real numerical results, then emit one JSON observation per operation.
fn main() {
    let scorer = PhaseQualityScorer::default();
    for n in [10usize, 100, 1000] {
        let qualities: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 10.0).collect();
        let amplitudes = vec![1.0; n];
        let huge = vec![1e308; n];
        assert!((scorer.score(&qualities, &amplitudes) - 0.45).abs() < 1e-12);
        assert!((scorer.score(&qualities, &huge) - 0.45).abs() < 1e-12);
        let mask = scorer.downweight_mask(&qualities);
        assert!(mask
            .iter()
            .zip(&qualities)
            .all(|(&m, &q)| m == if q >= 0.3 { q } else { 0.0 }));
        let ordinary_samples =
            samples(|| scorer.score(black_box(&qualities), black_box(&amplitudes)));
        let huge_samples = samples(|| scorer.score(black_box(&qualities), black_box(&huge)));
        let mask_samples = samples(|| scorer.downweight_mask(black_box(&qualities)));
        for (operation, values) in [
            ("score", ordinary_samples),
            ("score_huge", huge_samples),
            ("mask", mask_samples),
        ] {
            println!(
                "{{\"n\":{n},\"operation\":\"{operation}\",\"calls\":1000,\"repeats\":5,\"normal_weight_bits\":{},\"huge_weight_bits\":{},\"seconds_per_call\":{values:?}}}",
                1.0_f64.to_bits(),
                1e308_f64.to_bits()
            );
        }
    }
}
