// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Second-order inertial Kuramoto (swing equation)

//! Second-order Kuramoto model with inertia for power grid dynamics.
//!
//! m_i θ̈_i + d_i θ̇_i = P_i + Σ_j K_ij sin(θ_j − θ_i)
//!
//! The swing equation is the standard model for power system transient
//! stability analysis. Desynchronisation → cascading blackout.
//!
//! References:
//!   Filatrella, Nielsen & Pedersen 2008, Eur. Phys. J. B 61:485-491.
//!   Dörfler & Bullo 2014, Automatica 50:1539-1564.

use std::f64::consts::TAU;

/// Single RK4 step of the swing equation.
///
/// Returns (new_theta, new_omega_dot) as contiguous Vec<f64> of length 2*N.
///
/// # Arguments
/// * `theta` — (N,) rotor angles
/// * `omega_dot` — (N,) angular velocities (deviation from nominal)
/// * `power` — (N,) power injection
/// * `knm_flat` — (N×N) row-major coupling matrix
/// * `inertia` — (N,) inertia constants m_i
/// * `damping` — (N,) damping coefficients d_i
/// * `n` — number of oscillators
/// * `dt` — timestep
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn inertial_step(
    theta: &[f64],
    omega_dot: &[f64],
    power: &[f64],
    knm_flat: &[f64],
    inertia: &[f64],
    damping: &[f64],
    n: usize,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    // RK4 for the coupled 2nd-order ODE system
    let (k1t, k1o) = swing_deriv(theta, omega_dot, power, knm_flat, inertia, damping, n);

    let th2: Vec<f64> = (0..n).map(|i| theta[i] + 0.5 * dt * k1t[i]).collect();
    let od2: Vec<f64> = (0..n).map(|i| omega_dot[i] + 0.5 * dt * k1o[i]).collect();
    let (k2t, k2o) = swing_deriv(&th2, &od2, power, knm_flat, inertia, damping, n);

    let th3: Vec<f64> = (0..n).map(|i| theta[i] + 0.5 * dt * k2t[i]).collect();
    let od3: Vec<f64> = (0..n).map(|i| omega_dot[i] + 0.5 * dt * k2o[i]).collect();
    let (k3t, k3o) = swing_deriv(&th3, &od3, power, knm_flat, inertia, damping, n);

    let th4: Vec<f64> = (0..n).map(|i| theta[i] + dt * k3t[i]).collect();
    let od4: Vec<f64> = (0..n).map(|i| omega_dot[i] + dt * k3o[i]).collect();
    let (k4t, k4o) = swing_deriv(&th4, &od4, power, knm_flat, inertia, damping, n);

    let dt6 = dt / 6.0;
    let new_theta: Vec<f64> = (0..n)
        .map(|i| {
            let raw = theta[i] + dt6 * (k1t[i] + 2.0 * k2t[i] + 2.0 * k3t[i] + k4t[i]);
            ((raw % TAU) + TAU) % TAU
        })
        .collect();
    let new_omega: Vec<f64> = (0..n)
        .map(|i| omega_dot[i] + dt6 * (k1o[i] + 2.0 * k2o[i] + 2.0 * k3o[i] + k4o[i]))
        .collect();

    (new_theta, new_omega)
}

/// Swing equation derivative: dθ/dt = ω, dω/dt = (P + coupling − d·ω) / m
fn swing_deriv(
    theta: &[f64],
    omega_dot: &[f64],
    power: &[f64],
    knm_flat: &[f64],
    inertia: &[f64],
    damping: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut accel = Vec::with_capacity(n);
    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            let k_ij = knm_flat[i * n + j];
            if k_ij.abs() > 1e-30 {
                coupling += k_ij * (theta[j] - theta[i]).sin();
            }
        }
        let a = (power[i] + coupling - damping[i] * omega_dot[i]) / inertia[i];
        accel.push(a);
    }
    (omega_dot.to_vec(), accel)
}

/// Run n_steps of inertial Kuramoto, returning final (theta, omega_dot)
/// and trajectory arrays (flat, row-major: n_steps × n).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn inertial_run(
    theta_init: &[f64],
    omega_init: &[f64],
    power: &[f64],
    knm_flat: &[f64],
    inertia: &[f64],
    damping: &[f64],
    n: usize,
    dt: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut th = theta_init.to_vec();
    let mut od = omega_init.to_vec();
    let mut theta_traj = Vec::with_capacity(n_steps * n);
    let mut omega_traj = Vec::with_capacity(n_steps * n);

    for _ in 0..n_steps {
        let (new_th, new_od) = inertial_step(&th, &od, power, knm_flat, inertia, damping, n, dt);
        th = new_th;
        od = new_od;
        theta_traj.extend_from_slice(&th);
        omega_traj.extend_from_slice(&od);
    }

    (th, od, theta_traj, omega_traj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_grid(n: usize) -> Vec<f64> {
        // Ring topology: nearest-neighbour coupling
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            let left = (i + n - 1) % n;
            let right = (i + 1) % n;
            knm[i * n + left] = 1.0;
            knm[i * n + right] = 1.0;
        }
        knm
    }

    #[test]
    fn test_step_preserves_length() {
        let n = 5;
        let theta = vec![0.0; n];
        let omega = vec![0.0; n];
        let power = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let inertia = vec![1.0; n];
        let damping = vec![0.1; n];

        let (new_th, new_od) = inertial_step(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.01);
        assert_eq!(new_th.len(), n);
        assert_eq!(new_od.len(), n);
    }

    #[test]
    fn test_zero_power_zero_coupling_no_motion() {
        let n = 3;
        let theta = vec![0.1, 0.2, 0.3];
        let omega = vec![0.0; n];
        let power = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let inertia = vec![1.0; n];
        let damping = vec![1.0; n]; // damping kills any velocity

        let (_, new_od) = inertial_step(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.01);
        for &v in &new_od {
            assert!(v.abs() < 1e-12, "should have no acceleration, got {v}");
        }
    }

    #[test]
    fn test_uniform_power_accelerates() {
        let n = 3;
        let theta = vec![0.0; n];
        let omega = vec![0.0; n];
        let power = vec![1.0; n]; // positive power → acceleration
        let knm = vec![0.0; n * n];
        let inertia = vec![1.0; n];
        let damping = vec![0.0; n]; // no damping

        let (_, new_od) = inertial_step(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.1);
        for &v in &new_od {
            assert!(v > 0.0, "positive power should accelerate, got {v}");
        }
    }

    #[test]
    fn test_damping_decelerates() {
        let n = 2;
        let theta = vec![0.0, PI];
        let omega = vec![1.0, -1.0]; // initial velocity
        let power = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let inertia = vec![1.0; n];
        let damping = vec![2.0; n]; // strong damping

        let (_, new_od) = inertial_step(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.1);
        // Velocity should decrease in magnitude
        assert!(new_od[0].abs() < 1.0, "damping should reduce |ω|, got {}", new_od[0]);
        assert!(new_od[1].abs() < 1.0, "damping should reduce |ω|, got {}", new_od[1]);
    }

    #[test]
    fn test_coupling_attracts() {
        let n = 2;
        let theta = vec![0.0, PI / 2.0]; // 90° apart
        let omega = vec![0.0; n];
        let power = vec![0.0; n];
        let mut knm = vec![0.0; 4];
        knm[0 * 2 + 1] = 1.0; // osc 0 coupled to osc 1
        knm[1 * 2 + 0] = 1.0; // osc 1 coupled to osc 0
        let inertia = vec![1.0; n];
        let damping = vec![0.0; n];

        let (_, new_od) = inertial_step(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.01);
        // Osc 0 should accelerate toward osc 1 (positive direction)
        assert!(new_od[0] > 0.0, "coupling should attract osc 0 toward osc 1");
        // Osc 1 should decelerate (negative direction toward osc 0)
        assert!(new_od[1] < 0.0, "coupling should attract osc 1 toward osc 0");
    }

    #[test]
    fn test_run_trajectory_shape() {
        let n = 4;
        let theta = vec![0.0; n];
        let omega = vec![0.0; n];
        let power = vec![0.1; n];
        let knm = make_grid(n);
        let inertia = vec![1.0; n];
        let damping = vec![0.5; n];
        let n_steps = 50;

        let (th_f, od_f, th_traj, od_traj) = inertial_run(
            &theta, &omega, &power, &knm, &inertia, &damping, n, 0.01, n_steps,
        );
        assert_eq!(th_f.len(), n);
        assert_eq!(od_f.len(), n);
        assert_eq!(th_traj.len(), n_steps * n);
        assert_eq!(od_traj.len(), n_steps * n);
    }

    #[test]
    fn test_theta_bounded() {
        let n = 3;
        let theta = vec![0.0, 2.0, 4.0];
        let omega = vec![10.0, -10.0, 5.0]; // large velocities
        let power = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let inertia = vec![1.0; n];
        let damping = vec![0.0; n];

        let (th, _, _, _) = inertial_run(&theta, &omega, &power, &knm, &inertia, &damping, n, 0.01, 100);
        for &v in &th {
            assert!(v >= 0.0 && v < TAU, "theta = {v} out of [0, 2π)");
        }
    }

    #[test]
    fn test_grid_synchronisation() {
        // Ring grid with asymmetric IC breaks symmetry → coupling drives sync
        let n = 6;
        // Asymmetric initial phases (not uniformly spaced)
        let theta = vec![0.0, 0.3, 0.5, 0.2, 0.8, 0.1];
        let omega = vec![0.0; n];
        let power = vec![0.5; n];
        let knm = make_grid(n);
        let knm: Vec<f64> = knm.iter().map(|&v| v * 5.0).collect();
        let inertia = vec![1.0; n];
        let damping = vec![1.0; n];

        let (th, _, _, _) = inertial_run(
            &theta, &omega, &power, &knm, &inertia, &damping, n, 0.01, 5000,
        );

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for &t in &th {
            sum_cos += t.cos();
            sum_sin += t.sin();
        }
        let r = ((sum_cos / n as f64).powi(2) + (sum_sin / n as f64).powi(2)).sqrt();
        assert!(r > 0.5, "grid should partially synchronise, R = {r}");
    }

    #[test]
    fn test_inertia_slows_response() {
        // Higher inertia → slower acceleration
        let n = 2;
        let theta = vec![0.0, PI];
        let omega = vec![0.0; n];
        let power = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let damping = vec![0.0; n];

        let low_inertia = vec![0.5; n];
        let high_inertia = vec![5.0; n];

        let (_, od_low) = inertial_step(&theta, &omega, &power, &knm, &low_inertia, &damping, n, 0.1);
        let (_, od_high) = inertial_step(&theta, &omega, &power, &knm, &high_inertia, &damping, n, 0.1);

        assert!(
            od_low[0].abs() > od_high[0].abs(),
            "low inertia should accelerate faster: {} vs {}",
            od_low[0], od_high[0]
        );
    }
}
