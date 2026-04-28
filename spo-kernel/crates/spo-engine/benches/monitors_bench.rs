// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Monitor-family compute benchmarks

//! Criterion benchmarks for sequential compute paths in the monitor /
//! topology family. The wiring-pipeline rule requires a benchmark for
//! every Rust module; this file covers the high-cost paths that did
//! not fit naturally into `parallel_bench.rs` (no Rayon partitioning).
//!
//! Covered modules:
//!
//! * `spectral::fiedler_value` and `spectral::symmetric_eigen` — Jacobi
//!   eigen decomposition on small symmetric matrices.
//! * `lyapunov::lyapunov_spectrum` — finite-time spectrum via QR.
//! * `transfer_entropy::phase_transfer_entropy` — pairwise TE on phase
//!   bins; matrix variant for the cross-oscillator panel.
//! * `embedding::delay_embed` — Takens reconstruction.
//! * `recurrence::recurrence_matrix` and `recurrence::rqa` — RQA on a
//!   trajectory.
//! * `hodge::hodge_decomposition` — gradient / curl / harmonic split of
//!   the coupling-weighted phase field.
//! * `entropy_prod::entropy_production_rate` — irreversibility proxy.
//! * `pid::redundancy` and `pid::synergy` — partial-information
//!   decomposition of two oscillator groups.
//!
//! Run with: ``cargo bench -p spo-engine --bench monitors_bench``.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use spo_engine::embedding::delay_embed;
use spo_engine::entropy_prod::entropy_production_rate;
use spo_engine::hodge::hodge_decomposition;
use spo_engine::lyapunov::lyapunov_spectrum;
use spo_engine::pid::{redundancy, synergy};
use spo_engine::recurrence::{recurrence_matrix, rqa};
use spo_engine::spectral::{fiedler_value, symmetric_eigen};
use spo_engine::transfer_entropy::{phase_transfer_entropy, transfer_entropy_matrix};

fn ring_knm(n: usize, strength: f64) -> Vec<f64> {
    let mut knm = vec![0.0; n * n];
    for i in 0..n {
        let right = (i + 1) % n;
        let left = (i + n - 1) % n;
        knm[i * n + right] = strength;
        knm[i * n + left] = strength;
    }
    knm
}

fn signal_sin(t: usize, freq: f64) -> Vec<f64> {
    (0..t).map(|i| (i as f64 * freq).sin()).collect()
}

fn bench_fiedler_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_fiedler_value");
    for &n in &[8usize, 32, 128] {
        let knm = ring_knm(n, 0.3);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let f = fiedler_value(&knm, n);
                criterion::black_box(f);
            });
        });
    }
    group.finish();
}

fn bench_symmetric_eigen(c: &mut Criterion) {
    // Jacobi eigen decomposition on a small symmetric matrix.
    let mut group = c.benchmark_group("spectral_symmetric_eigen");
    for &n in &[4usize, 8, 16] {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = ((i + 2 * j) as f64 * 0.1).sin();
                m[i * n + j] = v;
                m[j * n + i] = v;
            }
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = symmetric_eigen(&m, n, 50, 1e-10);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_lyapunov_spectrum(c: &mut Criterion) {
    // Lyapunov spectrum is O(N²·n_steps) with periodic QR — keep small.
    let mut group = c.benchmark_group("lyapunov_spectrum");
    for &n in &[4usize, 8] {
        let knm = ring_knm(n, 0.4);
        let alpha = vec![0.0; n * n];
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.05 * i as f64).collect();
        let phases_init: Vec<f64> = (0..n).map(|i| i as f64 * 0.13).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let r =
                    lyapunov_spectrum(&phases_init, &omegas, &knm, &alpha, 0.01, 200, 20, 0.0, 0.0)
                        .expect("valid lyapunov_spectrum arguments");
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_transfer_entropy_pairwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("transfer_entropy_pairwise");
    for &t in &[200usize, 500, 1000] {
        let src = signal_sin(t, 0.2);
        let tgt = signal_sin(t, 0.21);
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, _| {
            b.iter(|| {
                let te = phase_transfer_entropy(&src, &tgt, 16);
                criterion::black_box(te);
            });
        });
    }
    group.finish();
}

fn bench_transfer_entropy_matrix(c: &mut Criterion) {
    // Matrix variant grows as O(N²·T·n_bins).
    let mut group = c.benchmark_group("transfer_entropy_matrix");
    for &params in &[(8usize, 200usize), (16, 400)] {
        let (n, t) = params;
        let series: Vec<f64> = (0..n * t).map(|i| (i as f64 * 0.07).sin()).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_t{t}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let m = transfer_entropy_matrix(&series, n, t, 12)
                        .expect("valid transfer_entropy_matrix arguments");
                    criterion::black_box(m);
                });
            },
        );
    }
    group.finish();
}

fn bench_delay_embed(c: &mut Criterion) {
    let mut group = c.benchmark_group("delay_embed");
    let signal = signal_sin(8192, 0.05);
    for &(delay, dim) in &[(1usize, 3usize), (5, 5), (10, 10)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{delay}_m{dim}")),
            &(delay, dim),
            |b, _| {
                b.iter(|| {
                    let r = delay_embed(&signal, delay, dim).expect("valid delay_embed arguments");
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_recurrence_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrence_matrix");
    for &t in &[64usize, 128, 256] {
        // Single-dimension scalar trajectory (d=1) — RP grows as T².
        let trajectory = signal_sin(t, 0.1);
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, &t| {
            b.iter(|| {
                let r = recurrence_matrix(&trajectory, t, 1, 0.2, false)
                    .expect("valid recurrence_matrix arguments");
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_rqa(c: &mut Criterion) {
    // RQA over a precomputed recurrence matrix.
    let t = 128usize;
    let trajectory = signal_sin(t, 0.1);
    let recurrence = recurrence_matrix(&trajectory, t, 1, 0.2, false).unwrap();
    c.bench_function("rqa_t128", |b| {
        b.iter(|| {
            let r = rqa(&recurrence, t, 2, 2, true).expect("valid rqa arguments");
            criterion::black_box(r);
        });
    });
}

fn bench_hodge_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("hodge_decomposition");
    for &n in &[6usize, 12, 24] {
        let knm = ring_knm(n, 0.5);
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.21).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let (g, c_part, h) = hodge_decomposition(&knm, &phases, n);
                criterion::black_box((g, c_part, h));
            });
        });
    }
    group.finish();
}

fn bench_entropy_production(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_production_rate");
    for &n in &[8usize, 32, 128] {
        let knm = ring_knm(n, 0.3);
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.17).collect();
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let r = entropy_production_rate(&phases, &omegas, &knm, 0.0, 0.01);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pid_redundancy_synergy(c: &mut Criterion) {
    // Phases include two halves; PID groups defined over indices.
    let mut group = c.benchmark_group("pid");
    for &n in &[8usize, 16] {
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let group_a: Vec<usize> = (0..n / 2).collect();
        let group_b: Vec<usize> = (n / 2..n).collect();
        group.bench_with_input(BenchmarkId::new("redundancy", n), &n, |b, _| {
            b.iter(|| {
                let r = redundancy(&phases, &group_a, &group_b, 16);
                criterion::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("synergy", n), &n, |b, _| {
            b.iter(|| {
                let s = synergy(&phases, &group_a, &group_b, 16);
                criterion::black_box(s);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fiedler_value,
    bench_symmetric_eigen,
    bench_lyapunov_spectrum,
    bench_transfer_entropy_pairwise,
    bench_transfer_entropy_matrix,
    bench_delay_embed,
    bench_recurrence_matrix,
    bench_rqa,
    bench_hodge_decomposition,
    bench_entropy_production,
    bench_pid_redundancy_synergy,
);
criterion_main!(benches);
