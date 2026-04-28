// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase-SINDy Symbolic Discovery

//! Symbolic Discovery of Phase Dynamics using SINDy.
//!
//! Discovers governing equations of a coupled oscillator network by
//! performing sparse regression (STLSQ) on a library of trigonometric
//! interaction terms.
//!
//! Brunton, Proctor & Kutz 2016, PNAS 113(15):3932-3937.

use rayon::prelude::*;

use std::f64::consts::{PI, TAU};

/// Run Phase-SINDy: discover coupling coefficients for each oscillator.
///
/// Returns (N × N) where `[i][i]` = ω_i and `[i][j]` = K_ij for j≠i.
#[must_use]
pub fn sindy_fit(
    phases: &[f64],
    n_osc: usize,
    n_time: usize,
    dt: f64,
    threshold: f64,
    max_iter: usize,
) -> Vec<f64> {
    if n_time < 3 || n_osc == 0 {
        return vec![0.0; n_osc * n_osc];
    }

    let t_eff = n_time - 1;
    let theta_dot = compute_theta_dot(phases, n_osc, n_time, dt);
    let mut result = vec![0.0; n_osc * n_osc];

    // Parallelize over oscillators
    result
        .par_chunks_mut(n_osc)
        .enumerate()
        .for_each(|(i, res_row)| {
            let (library, target) = build_library(phases, &theta_dot, n_osc, t_eff, i);
            let xi = stlsq_node(&library, &target, t_eff, n_osc, threshold, max_iter);
            res_row[..n_osc].copy_from_slice(&xi[..n_osc]);
        });

    result
}

/// Compute θ̇ via finite differences with phase unwrapping.
fn compute_theta_dot(phases: &[f64], n_osc: usize, n_time: usize, dt: f64) -> Vec<f64> {
    let t_eff = n_time - 1;
    let mut theta_dot = vec![0.0; n_osc * t_eff];

    // Use chunks if we change layout, but let is keep layout and use a different approach.
    // Or just parallelize over time steps tt? No, dependencies between tt.
    // Let is stick to sequential for theta_dot as it is O(N*T) while STLSQ is O(N^2 * T) or more.

    for i in 0..n_osc {
        let mut prev = phases[i];
        for tt in 0..t_eff {
            let curr = phases[(tt + 1) * n_osc + i];
            let mut diff = curr - prev;
            if diff > PI {
                diff -= TAU;
            } else if diff < -PI {
                diff += TAU;
            }
            // Repeat once more for edge cases
            if diff > PI {
                diff -= TAU;
            } else if diff < -PI {
                diff += TAU;
            }

            theta_dot[tt * n_osc + i] = diff / dt;
            prev = curr;
        }
    }
    theta_dot
}

/// Build trigonometric library for oscillator `i`.
///
/// Features: constant (at index i) + sin(θ_j - θ_i) for j ≠ i.
fn build_library(
    phases: &[f64],
    theta_dot: &[f64],
    n_osc: usize,
    t_eff: usize,
    i: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut library = vec![0.0; t_eff * n_osc];
    let mut target = vec![0.0; t_eff];

    for tt in 0..t_eff {
        target[tt] = theta_dot[tt * n_osc + i];
        library[tt * n_osc + i] = 1.0; // constant feature
        for j in 0..n_osc {
            if j != i {
                let phi_j = phases[tt * n_osc + j];
                let phi_i = phases[tt * n_osc + i];
                library[tt * n_osc + j] = (phi_j - phi_i).sin();
            }
        }
    }

    (library, target)
}

/// STLSQ (Sequential Thresholded Least Squares) for one node.
fn stlsq_node(
    library: &[f64],
    target: &[f64],
    t_eff: usize,
    n_features: usize,
    threshold: f64,
    max_iter: usize,
) -> Vec<f64> {
    let mut xi = lstsq(library, target, t_eff, n_features);

    for _ in 0..max_iter {
        for v in xi.iter_mut() {
            if v.abs() < threshold {
                *v = 0.0;
            }
        }
        let big: Vec<usize> = xi
            .iter()
            .enumerate()
            .filter(|(_, v)| v.abs() >= threshold)
            .map(|(idx, _)| idx)
            .collect();
        if big.is_empty() {
            break;
        }

        let n_big = big.len();
        let mut lib_red = vec![0.0; t_eff * n_big];
        for tt in 0..t_eff {
            for (k, &feat_idx) in big.iter().enumerate() {
                lib_red[tt * n_big + k] = library[tt * n_features + feat_idx];
            }
        }
        let xi_red = lstsq(&lib_red, target, t_eff, n_big);
        let mut xi_new = vec![0.0; n_features];
        for (k, &feat_idx) in big.iter().enumerate() {
            xi_new[feat_idx] = xi_red[k];
        }
        xi = xi_new;
    }

    xi
}

/// Least squares via normal equations: (AᵀA)⁻¹ Aᵀ b.
fn lstsq(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut ata = vec![0.0; n * n];

    // ATA is symmetric, parallelize over rows
    ata.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
        for k in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += a[i * n + j] * a[i * n + k];
            }
            row[k] = sum;
        }
    });

    let mut atb = vec![0.0; n];
    atb.par_iter_mut().enumerate().for_each(|(j, val)| {
        let mut sum = 0.0;
        for i in 0..m {
            sum += a[i * n + j] * b[i];
        }
        *val = sum;
    });

    solve_linear_sys(n, &mut ata, &mut atb)
}

/// Gaussian elimination with partial pivoting.
fn solve_linear_sys(n: usize, a: &mut [f64], b: &mut [f64]) -> Vec<f64> {
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

    fn generate_kuramoto_trajectory(
        n: usize,
        t: usize,
        dt: f64,
        omegas: &[f64],
        coupling: &[f64],
    ) -> Vec<f64> {
        let mut phases = vec![0.0; t * n];
        for i in 0..n {
            phases[i] = i as f64 * 0.5;
        }
        for tt in 1..t {
            for i in 0..n {
                let mut deriv = omegas[i];
                for j in 0..n {
                    if i != j {
                        let diff = phases[(tt - 1) * n + j] - phases[(tt - 1) * n + i];
                        deriv += coupling[i * n + j] * diff.sin();
                    }
                }
                phases[tt * n + i] = phases[(tt - 1) * n + i] + dt * deriv;
            }
        }
        phases
    }

    #[test]
    fn test_discovers_coupling() {
        let n = 3;
        let t = 500;
        let dt = 0.01;
        let omegas = vec![1.0, 1.5, 2.0];
        let coupling = vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.8, 0.5, 0.8, 0.0];
        let phases = generate_kuramoto_trajectory(n, t, dt, &omegas, &coupling);
        let result = sindy_fit(&phases, n, t, dt, 0.05, 10);
        for i in 0..n {
            assert!(
                (result[i * n + i] - omegas[i]).abs() < 0.5,
                "ω_{i}: got {}, expected {}",
                result[i * n + i],
                omegas[i],
            );
        }
    }

    #[test]
    fn test_sparse_zero_coupling() {
        let n = 2;
        let t = 200;
        let dt = 0.01;
        let omegas = vec![1.0, 2.0];
        let coupling = vec![0.0; n * n];
        let phases = generate_kuramoto_trajectory(n, t, dt, &omegas, &coupling);
        let result = sindy_fit(&phases, n, t, dt, 0.1, 10);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        result[i * n + j].abs() < 0.3,
                        "K[{i},{j}]={}",
                        result[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_short_data_returns_zeros() {
        let result = sindy_fit(&[0.0; 4], 2, 2, 0.01, 0.05, 10);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_output_size() {
        let result = sindy_fit(&vec![0.5; 50 * 4], 4, 50, 0.01, 0.05, 10);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_lstsq_simple() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let x = lstsq(&a, &b, 2, 2);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
    }
}
