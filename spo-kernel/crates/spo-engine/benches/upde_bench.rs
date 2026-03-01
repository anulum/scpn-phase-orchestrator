// SCPN Phase Orchestrator — UPDE Benchmarks

use criterion::{criterion_group, criterion_main, Criterion};

use spo_engine::coupling::CouplingBuilder;
use spo_engine::order_params::compute_order_parameter;
use spo_engine::upde::UPDEStepper;
use spo_types::{CouplingConfig, IntegrationConfig, Method};

fn bench_euler_step_n64(c: &mut Criterion) {
    let n = 64;
    let config = IntegrationConfig {
        dt: 0.001,
        method: Method::Euler,
        n_substeps: 1,
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

criterion_group!(
    benches,
    bench_euler_step_n64,
    bench_rk4_step_n64,
    bench_order_parameter_n64,
    bench_euler_1000_steps_n64,
);
criterion_main!(benches);
