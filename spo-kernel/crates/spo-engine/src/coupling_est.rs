// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Coupling estimation from phase data

//! Least-squares estimation of K_ij coupling matrix from observed
//! phase trajectories.
//!
//! dθ_i/dt - ω_i = Σ_j K_ij sin(θ_j - θ_i)
//!
//! Constructs the regression matrix from pairwise sin(Δθ) and solves
//! for K_ij via normal equations (Aᵀ A)⁻¹ Aᵀ b.

/// Estimate coupling matrix from phase trajectories.
///
/// # Arguments
/// * `phases` – row-major (n × T) phase trajectories
/// * `omegas` – natural frequencies, length `n`
/// * `n` – number of oscillators
/// * `t` – number of timesteps
/// * `dt` – timestep between samples
///
/// # Returns
/// Row-major (n × n) estimated coupling matrix, diagonal zeroed.
#[must_use]
pub fn estimate_coupling(phases: &[f64], omegas: &[f64], n: usize, t: usize, dt: f64) -> Vec<f64> {
    if t < 3 || n == 0 {
        return vec![0.0; n * n];
    }
    let t_eff = t - 1;
    let dphase = unwrapped_deriv(phases, n, t, dt);
    let mut knm = vec![0.0; n * n];
    for i in 0..n {
        let row = solve_node(phases, &dphase, omegas, n, t, t_eff, i);
        for j in 0..n {
            knm[i * n + j] = row[j];
        }
    }
    for i in 0..n {
        knm[i * n + i] = 0.0;
    }
    knm
}

/// Compute unwrapped phase derivatives (n × t_eff, row-major).
fn unwrapped_deriv(phases: &[f64], n: usize, t: usize, dt: f64) -> Vec<f64> {
    let t_eff = t - 1;
    let mut dp = vec![0.0; n * t_eff];
    for i in 0..n {
        let mut prev = phases[i * t];
        for tt in 0..t_eff {
            let curr = phases[i * t + tt + 1];
            let mut d = curr - prev;
            while d > std::f64::consts::PI {
                d -= std::f64::consts::TAU;
            }
            while d < -std::f64::consts::PI {
                d += std::f64::consts::TAU;
            }
            dp[i * t_eff + tt] = d / dt;
            prev = curr;
        }
    }
    dp
}

/// Solve coupling coefficients for one oscillator via normal equations.
fn solve_node(
    phases: &[f64],
    dphase: &[f64],
    omegas: &[f64],
    n: usize,
    t: usize,
    t_eff: usize,
    i: usize,
) -> Vec<f64> {
    let mut target = vec![0.0; t_eff];
    for tt in 0..t_eff {
        target[tt] = dphase[i * t_eff + tt] - omegas[i];
    }
    let mut ata = vec![0.0; n * n];
    let mut atb = vec![0.0; n];
    for tt in 0..t_eff {
        let phi_i = phases[i * t + tt];
        for j in 0..n {
            let s = (phases[j * t + tt] - phi_i).sin();
            atb[j] += s * target[tt];
            for k in 0..n {
                ata[j * n + k] += s * (phases[k * t + tt] - phi_i).sin();
            }
        }
    }
    solve_linear(n, &mut ata, &mut atb).unwrap_or_else(|| vec![0.0; n])
}

/// Gaussian elimination with partial pivoting.
fn solve_linear(n: usize, a: &mut [f64], b: &mut [f64]) -> Option<Vec<f64>> {
    forward_elimination(a, b, n);
    Some(back_substitution(a, b, n))
}

/// Forward elimination with partial pivoting.
fn forward_elimination(a: &mut [f64], b: &mut [f64], n: usize) {
    for col in 0..n {
        let mut max_val = 0.0;
        let mut max_row = col;
        for row in col..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            continue;
        }
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
}

/// Back substitution after forward elimination.
fn back_substitution(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let diag = a[col * n + col];
        if diag.abs() < 1e-14 {
            continue;
        }
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col * n + k] * x[k];
        }
        x[col] = sum / diag;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_zero_coupling_recovered() {
        // If coupling is zero, estimated K should be near zero
        let n = 3;
        let t = 100;
        let dt = 0.01;
        let omegas = vec![1.0, 2.0, 3.0];
        let mut phases = vec![0.0; n * t];
        for i in 0..n {
            for tt in 0..t {
                phases[i * t + tt] = (omegas[i] * dt * tt as f64) % TAU;
            }
        }
        let knm = estimate_coupling(&phases, &omegas, n, t, dt);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    knm[i * n + j].abs() < 0.5,
                    "K[{i},{j}]={} should be near 0",
                    knm[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_diagonal_always_zero() {
        let n = 3;
        let t = 50;
        let phases = vec![1.0; n * t];
        let omegas = vec![1.0; n];
        let knm = estimate_coupling(&phases, &omegas, n, t, 0.01);
        for i in 0..n {
            assert_eq!(knm[i * n + i], 0.0);
        }
    }

    #[test]
    fn test_short_trajectory_returns_zeros() {
        let knm = estimate_coupling(&[0.0; 4], &[1.0, 1.0], 2, 2, 0.01);
        // t=2 < 3, should return zeros
        for v in &knm {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_strong_coupling_detected() {
        // Generate trajectory with known coupling
        let n = 2;
        let t = 200;
        let dt: f64 = 0.01;
        let k_true: f64 = 2.0;
        let omegas: Vec<f64> = vec![1.0, 1.5];
        let mut phases: Vec<f64> = vec![0.0; n * t];
        phases[0] = 0.0; // osc 0 at t=0
        phases[t] = 1.0; // osc 1 at t=0

        for tt in 1..t {
            let p0: f64 = phases[tt - 1];
            let p1: f64 = phases[t + tt - 1];
            phases[tt] = p0 + dt * (omegas[0] + k_true * (p1 - p0).sin());
            phases[t + tt] = p1 + dt * (omegas[1] + k_true * (p0 - p1).sin());
        }

        let knm = estimate_coupling(&phases, &omegas, n, t, dt);
        // K_01 and K_10 should be positive (coupling detected)
        assert!(knm[1] > 0.5, "K_01={} should detect coupling > 0.5", knm[1]);
    }

    #[test]
    fn test_output_size() {
        let n = 4;
        let t = 30;
        let knm = estimate_coupling(&vec![0.0; n * t], &vec![1.0; n], n, t, 0.01);
        assert_eq!(knm.len(), n * n);
    }

    #[test]
    fn test_solve_linear_identity() {
        // Solve I·x = b → x = b
        let mut a = vec![1.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, 5.0];
        let x = solve_linear(2, &mut a, &mut b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_2x2() {
        // 2x + y = 5, x + 3y = 7 → x=1.6, y=1.8
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 7.0];
        let x = solve_linear(2, &mut a, &mut b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
}
