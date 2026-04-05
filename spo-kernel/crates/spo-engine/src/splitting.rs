// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Strang operator splitting integrator

//! Strang splitting: exact rotation for ω, RK4 for coupling.
//!
//! Splits dθ/dt = ω + coupling into:
//!   A: dθ/dt = ω          (exact: θ += ω·dt)
//!   B: dθ/dt = coupling    (RK4 on nonlinear part)
//!
//! Strang scheme: A(dt/2) → B(dt) → A(dt/2), second-order symmetric.
//!
//! Hairer, Lubich & Wanner 2006, Geometric Numerical Integration, §II.5.

use std::f64::consts::TAU;

/// Run Strang splitting integration for `n_steps`.
///
/// Each step: A(dt/2) → B(dt) → A(dt/2) where A is exact rotation
/// and B is RK4 on the coupling-only nonlinear part.
#[must_use]
pub fn splitting_run(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let n = phases.len();
    let mut p = phases.to_vec();
    let half_dt = 0.5 * dt;

    for _ in 0..n_steps {
        // A(dt/2): exact rotation
        for i in 0..n {
            p[i] = (p[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }

        // B(dt): RK4 on coupling-only derivative
        rk4_coupling(&mut p, knm, alpha, n, zeta, psi, dt);

        // A(dt/2): exact rotation
        for i in 0..n {
            p[i] = (p[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }
    }

    p
}

/// RK4 step on coupling-only derivative (no ω).
fn rk4_coupling(
    p: &mut [f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
    dt: f64,
) {
    let k1 = coupling_deriv(p, knm, alpha, n, zeta, psi);

    let mut p2 = vec![0.0; n];
    for i in 0..n {
        p2[i] = (p[i] + 0.5 * dt * k1[i]).rem_euclid(TAU);
    }
    let k2 = coupling_deriv(&p2, knm, alpha, n, zeta, psi);

    let mut p3 = vec![0.0; n];
    for i in 0..n {
        p3[i] = (p[i] + 0.5 * dt * k2[i]).rem_euclid(TAU);
    }
    let k3 = coupling_deriv(&p3, knm, alpha, n, zeta, psi);

    let mut p4 = vec![0.0; n];
    for i in 0..n {
        p4[i] = (p[i] + dt * k3[i]).rem_euclid(TAU);
    }
    let k4 = coupling_deriv(&p4, knm, alpha, n, zeta, psi);

    for i in 0..n {
        p[i] = (p[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .rem_euclid(TAU);
    }
}

/// Coupling-only derivative: Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(ψ - θ_i).
fn coupling_deriv(
    theta: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            let diff = theta[j] - theta[i] - alpha[i * n + j];
            coupling += knm[i * n + j] * diff.sin();
        }
        result[i] = coupling;
        if zeta != 0.0 {
            result[i] += zeta * (psi - theta[i]).sin();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_free_rotation() {
        let n = 3;
        let phases = vec![0.0, PI / 2.0, PI];
        let omegas = vec![1.0, 2.0, 3.0];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100);
        // Free rotation: θ_i = θ_i0 + ω_i·dt·n_steps (exact in A steps)
        for i in 0..n {
            let expected = (phases[i] + omegas[i] * 0.01 * 100.0).rem_euclid(TAU);
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "i={i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_synchronisation() {
        let n = 5;
        let phases = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let omegas = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 2.0;
                }
            }
        }
        let alpha = vec![0.0; n * n];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 1000);
        let mut sx = 0.0;
        let mut cx = 0.0;
        for p in &result {
            sx += p.sin();
            cx += p.cos();
        }
        let r = (sx * sx + cx * cx).sqrt() / n as f64;
        assert!(r > 0.8, "R={r} should be > 0.8");
    }

    #[test]
    fn test_phases_in_range() {
        let n = 4;
        let phases = vec![0.1, 2.0, 4.0, 5.5];
        let omegas = vec![3.0, -1.0, 2.0, 7.0];
        let knm = vec![0.3; n * n];
        let alpha = vec![0.0; n * n];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 0.5, 1.0, 0.01, 500);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} out of [0, 2π)");
        }
    }

    #[test]
    fn test_zero_steps() {
        let phases = vec![1.0, 2.0, 3.0];
        let omegas = vec![1.0; 3];
        let knm = vec![0.0; 9];
        let alpha = vec![0.0; 9];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 0);
        for (a, b) in result.iter().zip(phases.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_external_drive() {
        let n = 3;
        let phases = vec![0.0; n];
        let omegas = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 1.0, PI / 2.0, 0.01, 100);
        for p in &result {
            assert!(*p > 0.0, "should move toward psi");
        }
    }

    #[test]
    fn test_second_order_accuracy() {
        // Strang splitting should be second-order accurate
        // Compare coarse (dt) vs fine (dt/2) steps
        let n = 3;
        let phases = vec![0.0, 1.0, 2.0];
        let omegas = vec![1.0, 1.5, 2.0];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 0.5;
                }
            }
        }
        let alpha = vec![0.0; n * n];

        let coarse = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.02, 50);
        let fine = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100);
        let finer = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.005, 200);

        // Error ratio should be ~4 for second-order method
        let mut err_coarse = 0.0;
        let mut err_fine = 0.0;
        for i in 0..n {
            // Use finer as reference
            let dc = (coarse[i] - finer[i]).abs();
            let df = (fine[i] - finer[i]).abs();
            err_coarse += dc;
            err_fine += df;
        }
        // Second-order: err_coarse / err_fine ≈ 4
        if err_fine > 1e-12 {
            let ratio = err_coarse / err_fine;
            assert!(
                ratio > 2.0 && ratio < 8.0,
                "ratio={ratio}, expected ~4 for second-order"
            );
        }
    }

    #[test]
    fn test_identical_phases() {
        let n = 4;
        let phases = vec![1.0; n];
        let omegas = vec![2.0; n];
        let knm = vec![1.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = splitting_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100);
        // All should advance by ω·dt·n_steps
        let expected = (1.0_f64 + 2.0 * 0.01 * 100.0).rem_euclid(TAU);
        for p in &result {
            assert!((p - expected).abs() < 1e-6, "p={p}, expected={expected}");
        }
    }
}
