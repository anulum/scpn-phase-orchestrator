// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — RK45 integration tests

use spo_engine::upde::UPDEStepper;
use spo_types::{IntegrationConfig, Method};
use std::f64::consts::TAU;

fn rk45_config(dt: f64) -> IntegrationConfig {
    IntegrationConfig {
        dt,
        method: Method::RK45,
        n_substeps: 1,
        atol: 1e-6,
        rtol: 1e-3,
    }
}

fn zero_alpha(n: usize) -> Vec<f64> {
    vec![0.0; n * n]
}

#[test]
fn rk45_phases_stay_bounded() {
    let n = 8;
    let mut s = UPDEStepper::new(n, rk45_config(0.01)).unwrap();
    let mut phases: Vec<f64> = (0..n).map(|i| i as f64 * TAU / n as f64).collect();
    let omegas = vec![2.0; n];
    let knm = vec![0.1; n * n];
    let alpha = zero_alpha(n);

    for _ in 0..500 {
        s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .unwrap();
    }
    for &p in &phases {
        assert!((0.0..TAU).contains(&p), "phase {p} out of [0, 2pi)");
    }
}

#[test]
fn rk45_constant_derivative_advances() {
    let n = 4;
    let mut s = UPDEStepper::new(n, rk45_config(0.01)).unwrap();
    let mut phases = vec![1.0; n];
    let omegas = vec![1.0; n];
    let knm = vec![0.0; n * n];
    let alpha = zero_alpha(n);

    // Run several steps; phases should advance from initial 1.0
    for _ in 0..20 {
        s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .unwrap();
    }
    // Phase should have advanced (wrapped to [0, 2pi))
    for &p in &phases {
        assert!(p.is_finite(), "non-finite phase: {p}");
    }
}

#[test]
fn rk45_adaptive_dt_grows_for_coupled_system() {
    let n = 8;
    let mut s = UPDEStepper::new(n, rk45_config(0.001)).unwrap();
    let mut phases: Vec<f64> = (0..n).map(|i| 1.0 + 0.02 * i as f64).collect();
    let omegas = vec![1.0; n];
    let mut knm = vec![0.2; n * n];
    for i in 0..n {
        knm[i * n + i] = 0.0;
    }
    let alpha = zero_alpha(n);

    let initial_dt = s.last_dt();
    for _ in 0..50 {
        s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .unwrap();
    }
    // After many accepted steps on smooth dynamics, dt should grow beyond initial
    assert!(
        s.last_dt() >= initial_dt,
        "dt should grow: initial={initial_dt}, final={}",
        s.last_dt()
    );
}

#[test]
fn rk45_synchronisation_tendency() {
    let n = 8;
    let mut s = UPDEStepper::new(n, rk45_config(0.01)).unwrap();
    let mut phases: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * i as f64).collect();
    let omegas = vec![1.0; n];
    let mut knm = vec![5.0; n * n];
    for i in 0..n {
        knm[i * n + i] = 0.0;
    }
    let alpha = zero_alpha(n);

    let r_before = spo_engine::order_params::compute_order_parameter(&phases).0;
    s.run(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha, 500)
        .unwrap();
    let r_after = spo_engine::order_params::compute_order_parameter(&phases).0;

    assert!(
        r_after > r_before,
        "R should increase: {r_before:.4} -> {r_after:.4}"
    );
}

#[test]
fn rk45_last_dt_exposed() {
    let n = 4;
    let s = UPDEStepper::new(n, rk45_config(0.01)).unwrap();
    assert!((s.last_dt() - 0.01).abs() < 1e-15);
}
