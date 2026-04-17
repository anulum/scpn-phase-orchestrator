// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Utility & monitor compute benchmarks

//! Closing batch of criterion benchmarks for the remaining
//! compute-heavy `spo-engine` modules — basin stability, ITPC, E/I
//! balance, Strang splitting, imprint plasticity, network probability
//! entropy, EVS, Hilbert phase extraction, carrier decode, ethical
//! cost, HCP connectome generation and Ott–Antonsen reduction.
//!
//! With this file the wiring-pipeline rule (every compute path needs a
//! benchmark) is satisfied for ~24 modules in the engine crate across
//! `upde_bench.rs`, `parallel_bench.rs`, `monitors_bench.rs` and this
//! file.
//!
//! Run with: ``cargo bench -p spo-engine --bench utility_bench``.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use spo_engine::attnres::attnres_modulate;
use spo_engine::basin_stability::basin_stability;
use spo_engine::carrier::decode;
use spo_engine::connectome::load_hcp_connectome;
use spo_engine::ei_balance::compute_ei_balance;
use spo_engine::ethical::compute_ethical_cost;
use spo_engine::evs::frequency_specificity;
use spo_engine::imprint::ImprintModel;
use spo_engine::itpc::compute_itpc;
use spo_engine::npe::compute_npe;
use spo_engine::phase_extract::extract_phases;
use spo_engine::reduction::oa_run;
use spo_engine::splitting::splitting_run;

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

fn bench_basin_stability(c: &mut Criterion) {
    // Monte Carlo: keep n_samples small for the bench footprint.
    let mut group = c.benchmark_group("basin_stability");
    for &n in &[4usize, 8] {
        let knm = ring_knm(n, 0.4);
        let alpha = vec![0.0; n * n];
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.05 * i as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = basin_stability(
                    &omegas, &knm, &alpha, n, 0.01, 100, 50, 16, 0.5, 42,
                );
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_compute_itpc(c: &mut Criterion) {
    // Inter-trial phase coherence: O(N_trials · N_tp).
    let mut group = c.benchmark_group("compute_itpc");
    for &params in &[(50usize, 32usize), (200, 64), (500, 128)] {
        let (n_trials, n_tp) = params;
        let phases: Vec<f64> = (0..n_trials * n_tp)
            .map(|i| (i as f64 * 0.05).sin())
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("trials{n_trials}_tp{n_tp}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let r = compute_itpc(&phases, n_trials, n_tp);
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_ei_balance(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_ei_balance");
    for &n in &[16usize, 64, 256] {
        let knm = ring_knm(n, 0.3);
        let half = n / 2;
        let exc: Vec<usize> = (0..half).collect();
        let inh: Vec<usize> = (half..n).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = compute_ei_balance(&knm, n, &exc, &inh);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_splitting_run(c: &mut Criterion) {
    let mut group = c.benchmark_group("splitting_run");
    for &n in &[8usize, 32, 64] {
        let knm = ring_knm(n, 0.3);
        let alpha = vec![0.0; n * n];
        let omegas = vec![1.0; n];
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let r = splitting_run(
                    &phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100,
                );
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_imprint_update(c: &mut Criterion) {
    // Plasticity tick: per-step exposure update on N units.
    let mut group = c.benchmark_group("imprint_update");
    for &n in &[16usize, 128, 1024] {
        let mut model = ImprintModel::new(n, 0.01, 1.5)
            .expect("valid ImprintModel arguments");
        let exposure: Vec<f64> =
            (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                model.update(&exposure, 0.01);
                criterion::black_box(&model.m);
            });
        });
    }
    group.finish();
}

fn bench_compute_npe(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_npe");
    for &n in &[16usize, 64, 256] {
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.21).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let r = compute_npe(&phases, std::f64::consts::PI);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_evs_frequency_specificity(c: &mut Criterion) {
    let mut group = c.benchmark_group("evs_frequency_specificity");
    for &params in &[(40usize, 64usize), (100, 128), (200, 256)] {
        let (n_trials, n_tp) = params;
        let phases: Vec<f64> = (0..n_trials * n_tp)
            .map(|i| (i as f64 * 0.05).sin())
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("tr{n_trials}_tp{n_tp}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let r = frequency_specificity(
                        &phases, n_trials, n_tp, 1.5, 3.0,
                    );
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_phase_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase_extract");
    // DFT-based Hilbert is O(T²); keep T small.
    for &t in &[64usize, 128, 256] {
        let signal = signal_sin(t, 0.1);
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, _| {
            b.iter(|| {
                let r = extract_phases(&signal, 1.0);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_carrier_decode(c: &mut Criterion) {
    // decode: O(N²·z_dim) over the carrier coefficient matrix.
    let mut group = c.benchmark_group("carrier_decode");
    for &params in &[(8usize, 4usize), (16, 8), (32, 16)] {
        let (n, z_dim) = params;
        let z: Vec<f64> = (0..z_dim).map(|i| i as f64 * 0.1).collect();
        let a: Vec<f64> = (0..n * n * z_dim)
            .map(|i| (i as f64).sin() * 0.05)
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_zdim{z_dim}")),
            &params,
            |b, _| {
                b.iter(|| {
                    let r = decode(&z, &a, n);
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_compute_ethical_cost(c: &mut Criterion) {
    // SSGF ethical cost: per-step controller call, O(N²) network sums.
    let mut group = c.benchmark_group("compute_ethical_cost");
    for &n in &[16usize, 64, 256] {
        let knm = ring_knm(n, 0.2);
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.13).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = compute_ethical_cost(
                    &phases, &knm, n, 1.0, 0.5, 0.1, 0.1, 1.0, 0.3, 0.1, 5.0,
                );
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_load_hcp_connectome(c: &mut Criterion) {
    // Connectome load + symmetrise; deterministic per seed.
    let mut group = c.benchmark_group("load_hcp_connectome");
    for &n in &[16usize, 64, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let r = load_hcp_connectome(n, 42);
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_oa_run(c: &mut Criterion) {
    // Ott-Antonsen reduction is O(n_steps) on a 2-D mean-field.
    let mut group = c.benchmark_group("oa_run");
    for &n_steps in &[100usize, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_steps),
            &n_steps,
            |b, &n_steps| {
                b.iter(|| {
                    let r =
                        oa_run(0.5, 0.0, 1.0, 0.1, 0.5, 0.01, n_steps);
                    criterion::black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_attnres_modulate(c: &mut Criterion) {
    // State-dependent coupling modulation: O(N²) per call.
    let mut group = c.benchmark_group("attnres_modulate");
    for &n in &[16usize, 64, 128, 256] {
        let knm = ring_knm(n, 0.3);
        let theta: Vec<f64> = (0..n).map(|i| i as f64 * 0.13).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                let r = attnres_modulate(&knm, &theta, n, 4, 0.1, 0.5)
                    .expect("valid attnres_modulate arguments");
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_attnres_modulate,
    bench_basin_stability,
    bench_compute_itpc,
    bench_ei_balance,
    bench_splitting_run,
    bench_imprint_update,
    bench_compute_npe,
    bench_evs_frequency_specificity,
    bench_phase_extract,
    bench_carrier_decode,
    bench_compute_ethical_cost,
    bench_load_hcp_connectome,
    bench_oa_run,
);
criterion_main!(benches);
