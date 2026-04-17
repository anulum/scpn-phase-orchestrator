// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — UPDE benchmark

use criterion::{criterion_group, criterion_main, Criterion};

use spo_engine::coupling::CouplingBuilder;
use spo_engine::order_params::compute_order_parameter;
use spo_engine::pac::modulation_index;
use spo_engine::stuart_landau::StuartLandauStepper;
use spo_engine::upde::UPDEStepper;
use spo_types::{CouplingConfig, IntegrationConfig, Method};

fn bench_euler_step_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::Euler,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let mut stepper = UPDEStepper::new(n, config).unwrap();
    let cs = CouplingBuilder::build(n, &CouplingConfig::default()).unwrap();
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mut phases: Vec<f64> = (0..n)
        .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
        .collect();

    c.bench_function("euler_step_n64", |b| {
        b.iter(|| {
            stepper
                .step(&mut phases, &omegas, &cs.knm, 0.0, 0.0, &cs.alpha)
                .unwrap();
        })
    });
}

fn bench_rk4_step_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::RK4,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let mut stepper = UPDEStepper::new(n, config).unwrap();
    let cs = CouplingBuilder::build(n, &CouplingConfig::default()).unwrap();
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mut phases: Vec<f64> = (0..n)
        .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
        .collect();

    c.bench_function("rk4_step_n64", |b| {
        b.iter(|| {
            stepper
                .step(&mut phases, &omegas, &cs.knm, 0.0, 0.0, &cs.alpha)
                .unwrap();
        })
    });
}

fn bench_order_parameter_n64(c: &mut Criterion) {
    let n = 64;
    let phases: Vec<f64> = (0..n)
        .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
        .collect();

    c.bench_function("order_parameter_n64", |b| {
        b.iter(|| compute_order_parameter(&phases))
    });
}

fn bench_euler_1000_steps_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::Euler,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let cs = CouplingBuilder::build(n, &CouplingConfig::default()).unwrap();
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();

    c.bench_function("euler_1000steps_n64", |b| {
        b.iter(|| {
            let mut stepper = UPDEStepper::new(n, config.clone()).unwrap();
            let mut phases: Vec<f64> = (0..n)
                .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
                .collect();
            stepper
                .run(&mut phases, &omegas, &cs.knm, 0.0, 0.0, &cs.alpha, 1000)
                .unwrap();
        })
    });
}

fn bench_sl_euler_step_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::Euler,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let mut stepper = StuartLandauStepper::new(n, config).unwrap();
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mu: Vec<f64> = vec![1.0; n];
    let knm: Vec<f64> = {
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = (i as f64 - j as f64).abs();
                    k[i * n + j] = 0.3 * (-0.3 * d).exp();
                }
            }
        }
        k
    };
    let knm_r = knm.clone();
    let alpha = vec![0.0; n * n];
    let mut state: Vec<f64> = (0..n)
        .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
        .chain(std::iter::repeat(1.0).take(n))
        .collect();

    c.bench_function("sl_euler_step_n64", |b| {
        b.iter(|| {
            stepper
                .step(
                    &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 1.0,
                )
                .unwrap();
        })
    });
}

fn bench_sl_rk4_step_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::RK4,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let mut stepper = StuartLandauStepper::new(n, config).unwrap();
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mu: Vec<f64> = vec![1.0; n];
    let knm: Vec<f64> = {
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = (i as f64 - j as f64).abs();
                    k[i * n + j] = 0.3 * (-0.3 * d).exp();
                }
            }
        }
        k
    };
    let knm_r = knm.clone();
    let alpha = vec![0.0; n * n];
    let mut state: Vec<f64> = (0..n)
        .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
        .chain(std::iter::repeat(1.0).take(n))
        .collect();

    c.bench_function("sl_rk4_step_n64", |b| {
        b.iter(|| {
            stepper
                .step(
                    &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 1.0,
                )
                .unwrap();
        })
    });
}

fn bench_sl_1000_steps_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::Euler,
        n_substeps: 1,
        ..IntegrationConfig::default()
    };
    let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mu: Vec<f64> = vec![1.0; n];
    let knm: Vec<f64> = {
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = (i as f64 - j as f64).abs();
                    k[i * n + j] = 0.3 * (-0.3 * d).exp();
                }
            }
        }
        k
    };
    let knm_r = knm.clone();
    let alpha = vec![0.0; n * n];

    c.bench_function("sl_1000steps_n64", |b| {
        b.iter(|| {
            let mut stepper = StuartLandauStepper::new(n, config.clone()).unwrap();
            let mut state: Vec<f64> = (0..n)
                .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
                .chain(std::iter::repeat(1.0).take(n))
                .collect();
            stepper
                .run(
                    &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 1.0, 1000,
                )
                .unwrap();
        })
    });
}

fn bench_pac_modulation_index_1000(c: &mut Criterion) {
    let n = 1000;
    let theta: Vec<f64> = (0..n)
        .map(|i| (i as f64 / n as f64) * std::f64::consts::TAU)
        .collect();
    let amp: Vec<f64> = theta.iter().map(|t| (t.sin() + 1.0) * 0.5).collect();

    c.bench_function("pac_mi_n1000", |b| {
        b.iter(|| modulation_index(&theta, &amp, 18))
    });
}

criterion_group!(
    benches,
    bench_euler_step_n64,
    bench_rk4_step_n64,
    bench_order_parameter_n64,
    bench_euler_1000_steps_n64,
    bench_sl_euler_step_n64,
    bench_sl_rk4_step_n64,
    bench_sl_1000_steps_n64,
    bench_pac_modulation_index_1000,
);
criterion_main!(benches);
