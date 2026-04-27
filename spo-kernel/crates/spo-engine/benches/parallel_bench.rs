// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Rayon parallelisation benchmarks

//! Criterion benchmarks for every Rayon-parallelised hot path in
//! `spo-engine`. The wiring-pipeline rule requires a benchmark for each
//! Rust module, so a regression between releases surfaces as a criterion
//! delta rather than an anecdotal observation.
//!
//! Covered paths:
//!
//! * `chimera::local_order_parameter` — per-oscillator neighbourhood
//!   sum partitioned across threads via `par_iter_mut`.
//! * `bifurcation::trace_sync_transition` — K-sweep parallelised via
//!   `par_iter` over independent steady-state simulations.
//! * `dimension::correlation_integral` — ε-scale sampling parallelised
//!   across reference points.
//! * `dimension::kaplan_yorke_dimension` — linear pass over the
//!   Lyapunov spectrum (baseline scaling test).
//! * `poincare::poincare_section` — hyperplane crossings on a scalar
//!   trajectory.
//! * `market::market_order_parameter` — column-wise R(t) reduction.
//! * `market::market_plv` — windowed pairwise PLV.
//! * `market::detect_regimes` — regime classification over R(t).
//! * `coupling_est::estimate_coupling` — least-squares K_ij inference.
//! * `sindy::sindy_fit` — STLSQ symbolic discovery, parallel across
//!   oscillators.
//!
//! Run with: ``cargo bench -p spo-engine --bench parallel_bench``.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use spo_engine::bifurcation::trace_sync_transition;
use spo_engine::chimera::local_order_parameter;
use spo_engine::coupling_est::estimate_coupling;
use spo_engine::dimension::{correlation_integral, kaplan_yorke_dimension};
use spo_engine::market::{detect_regimes, market_order_parameter, market_plv};
use spo_engine::poincare::{poincare_section, CrossingDirection};
use spo_engine::sindy::sindy_fit;

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

fn make_ring_phases(n_osc: usize, n_time: usize, freq: f64) -> Vec<f64> {
    // Row-major (n_time × n_osc).
    let dt = 0.01;
    let mut out = Vec::with_capacity(n_time * n_osc);
    for t in 0..n_time {
        for i in 0..n_osc {
            let phase = freq * (t as f64 * dt + 0.1 * i as f64);
            out.push(phase.rem_euclid(std::f64::consts::TAU));
        }
    }
    out
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
    // Light-weight K-sweep so the benchmark finishes quickly; the goal is to
    // confirm the parallel `par_iter` path does not regress relative to a
    // stable baseline, not to drive the integrator hard.
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

fn bench_dimension_correlation_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_correlation_integral");
    for &t in &[200usize, 500, 1000] {
        let d = 3usize;
        // A simple non-trivial trajectory (Lissajous-style) in 3D.
        let trajectory: Vec<f64> = (0..t * d)
            .map(|i| {
                let row = (i / d) as f64;
                let dim = i % d;
                match dim {
                    0 => (0.02 * row).sin(),
                    1 => (0.03 * row).cos(),
                    _ => (0.05 * row).sin(),
                }
            })
            .collect();
        let epsilons: Vec<f64> = (1..=16).map(|k| 0.02 * k as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, &t| {
            b.iter(|| {
                let r = correlation_integral(&trajectory, t, d, &epsilons, 16, 0)
                    .expect("valid correlation_integral arguments");
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_dimension_kaplan_yorke(c: &mut Criterion) {
    // kaplan_yorke is sequential but cheap — sanity check against
    // accidental algorithmic regressions.
    let exponents: Vec<f64> = vec![0.5, 0.2, 0.05, -0.1, -0.3, -0.6, -1.0];
    c.bench_function("dimension_kaplan_yorke_small_spectrum", |b| {
        b.iter(|| {
            let d = kaplan_yorke_dimension(&exponents);
            criterion::black_box(d);
        });
    });
}

fn bench_poincare_section(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_section");
    // 1-D trajectory: scalar oscillatory signal; normal [1.0], offset 0.0
    // counts zero crossings in both directions.
    let d = 1usize;
    let normal = vec![1.0];
    for &t in &[512usize, 2048, 8192] {
        let trajectory: Vec<f64> = (0..t).map(|i| (i as f64 * 0.05).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, &t| {
            b.iter(|| {
                let result =
                    poincare_section(&trajectory, t, d, &normal, 0.0, CrossingDirection::Both)
                        .expect("valid poincare_section arguments");
                criterion::black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_market_order_parameter(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_order_parameter");
    for &params in &[(64usize, 200usize), (128, 500), (256, 1000)] {
        let (n, t) = params;
        let phases = make_ring_phases(n, t, 0.5);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_t{t}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let r = market_order_parameter(&phases, t, n);
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_market_plv(c: &mut Criterion) {
    // Smaller grid — PLV is O(N² · window).
    let mut group = c.benchmark_group("market_plv");
    for &params in &[(16usize, 100usize, 32usize), (32, 200, 64)] {
        let (n, t, window) = params;
        let phases = make_ring_phases(n, t, 0.3);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_t{t}_w{window}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let plv = market_plv(&phases, t, n, window);
                    criterion::black_box(plv);
                });
            },
        );
    }
    group.finish();
}

fn bench_market_detect_regimes(c: &mut Criterion) {
    // regime classification scales linearly with T.
    let r_series: Vec<f64> = (0..4096)
        .map(|i| 0.5 + 0.4 * (i as f64 * 0.01).sin())
        .collect();
    c.bench_function("market_detect_regimes_n4096", |b| {
        b.iter(|| {
            let out = detect_regimes(&r_series, 0.8, 0.2);
            criterion::black_box(out);
        });
    });
}

fn bench_coupling_est(c: &mut Criterion) {
    let mut group = c.benchmark_group("coupling_est");
    for &params in &[(8usize, 200usize), (16, 400)] {
        let (n, t) = params;
        let phases = make_ring_phases(n, t, 0.4);
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
        let dt = 0.01;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_t{t}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let k = estimate_coupling(&phases, &omegas, n, t, dt);
                    criterion::black_box(k);
                });
            },
        );
    }
    group.finish();
}

fn bench_sindy_fit(c: &mut Criterion) {
    // sindy is heavy — restrict to small networks for bench footprint.
    let mut group = c.benchmark_group("sindy_fit");
    for &n in &[4usize, 8] {
        let t = 200usize;
        let phases = make_ring_phases(n, t, 0.5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let result = sindy_fit(&phases, n, t, 0.01, 0.05, 10);
                criterion::black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chimera_local_order,
    bench_bifurcation_sweep,
    bench_dimension_correlation_integral,
    bench_dimension_kaplan_yorke,
    bench_poincare_section,
    bench_market_order_parameter,
    bench_market_plv,
    bench_market_detect_regimes,
    bench_coupling_est,
    bench_sindy_fit,
);
criterion_main!(benches);
