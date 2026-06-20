// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Digital-twin divergence benchmark

//! Criterion benchmark for `twin_confidence::twin_divergence`.
//!
//! Times the phase-histogram Jensen–Shannon divergence (O(N) binning over
//! `n_bins` bins) plus the order-window Wasserstein-1 distance (O(W log W) via
//! two ascending sorts) at increasing phase counts, matching the Python
//! benchmark harness in `benchmarks/twin_confidence_benchmark.py`.
//!
//! Run with: ``cargo bench -p spo-engine --bench twin_confidence_bench``.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use spo_engine::twin_confidence::twin_divergence;

fn bench_twin_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("twin_divergence");
    let w = 64usize;
    let n_bins = 36usize;
    let model_order: Vec<f64> = (0..w).map(|i| i as f64 / w as f64).collect();
    let observed_order: Vec<f64> = (0..w)
        .map(|i| ((i as f64 + 0.5) / w as f64).min(1.0))
        .collect();
    for &n in &[64usize, 256, 1024] {
        let model_phases: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37) % 6.283).collect();
        let observed_phases: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37 + 0.1) % 6.283).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let r = twin_divergence(
                    &model_phases,
                    &observed_phases,
                    &model_order,
                    &observed_order,
                    n_bins,
                )
                .expect("valid twin_divergence arguments");
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_twin_divergence);
criterion_main!(benches);
