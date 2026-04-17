// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Rayon parallelisation benchmarks

//! Benchmarks for the Rayon-parallelised paths:
//!
//! * `chimera::local_order_parameter` — per-oscillator neighbourhood sum
//!   partitioned across threads via `par_iter_mut`.
//! * `bifurcation::trace_sync_transition` — K-sweep parallelised via
//!   `par_iter` over independent steady-state simulations.
//!
//! Run with: ``cargo bench -p spo-engine --bench parallel_bench``.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use spo_engine::bifurcation::trace_sync_transition;
use spo_engine::chimera::local_order_parameter;

fn make_ring_knm(n: usize) -> Vec<f64> {
    let mut knm = vec![0.0; n * n];
    for i in 0..n {
        let right = (i + 1) % n;
        let left = (i + n - 1) % n;
        knm[i * n + right] = 0.3;
        knm[i * n + left] = 0.3;
    }
    knm
}

fn bench_chimera_local_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("chimera_local_order");
    for &n in &[64usize, 256, 1024] {
        let knm = make_ring_knm(n);
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.17).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = local_order_parameter(&phases, &knm, n);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_bifurcation_sweep(c: &mut Criterion) {
    // Light-weight K-sweep so the benchmark finishes quickly; the point is to
    // confirm the parallel `par_iter` path does not regress relative to a
    // stable baseline, not to drive the Dormand-Prince integrator hard.
    let mut group = c.benchmark_group("bifurcation_sweep");
    for &n_points in &[8usize, 16, 32] {
        let n = 16usize;
        let knm = make_ring_knm(n);
        let alpha = vec![0.0; n * n];
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.02 * i as f64).collect();
        let phases_init: Vec<f64> = (0..n).map(|i| i as f64 * 0.13).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_points),
            &n_points,
            |b, &n_points| {
                b.iter(|| {
                    let result = trace_sync_transition(
                        &omegas,
                        &knm,
                        &alpha,
                        n,
                        &phases_init,
                        0.0,
                        4.0,
                        n_points,
                        0.01,
                        80,
                        40,
                    );
                    criterion::black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chimera_local_order, bench_bifurcation_sweep);
criterion_main!(benches);
