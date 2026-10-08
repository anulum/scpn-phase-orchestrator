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

use nalgebra::{DMatrix, DVector, SVD};
use rayon::prelude::*;

use std::f64::consts::{PI, TAU};

/// Run Phase-SINDy: discover coupling coefficients for each oscillator.
///
/// `phases` is a row-major `n_time × n_osc` trajectory in radians; `dt` is
/// the positive sample period in seconds. Adjacent phase increments are
/// reduced to the principal interval, with the original sign retained for
/// exact half turns. Sampling must resolve increments below half a turn to
/// identify physical angular velocity rather than its sampled alias.
///
/// Sequential thresholded least squares uses a rectangular SVD, retaining
/// singular values above epsilon * max(rows, columns) * the largest value.
/// Rank-deficient libraries receive a minimum-norm solution; their individual
/// coupling coefficients are not identifiable from the trajectory alone.
///
/// Returns a row-major `N × N` matrix: `[i][i] = ω_i` and `[i][j] = K_ij`
/// for `j != i`, so rows are targets and columns are sources. Coefficients
/// have units of radians per second. Terms strictly below `threshold` are
/// removed, then the remaining features are refitted for `max_iter` rounds.
///
/// # Errors
///
/// Returns an error for invalid dimensions, controls, non-finite input or
/// derived arithmetic, failed SVD convergence, or non-finite coefficients.
pub fn sindy_fit(
    phases: &[f64],
    n_osc: usize,
    n_time: usize,
    dt: f64,
    threshold: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    validate_sindy_inputs(phases, n_osc, n_time, dt, threshold, max_iter)?;

    let t_eff = n_time - 1;
    let theta_dot = compute_theta_dot(phases, n_osc, n_time, dt);
    if theta_dot.iter().any(|value| !value.is_finite()) {
        return Err("Phase-SINDy phase derivatives must remain finite".to_string());
    }
    let mut result = vec![0.0; n_osc * n_osc];

    result.par_chunks_mut(n_osc).enumerate().try_for_each(
        |(i, res_row)| -> Result<(), String> {
            let (library, target) = build_library(phases, &theta_dot, n_osc, t_eff, i);
            let xi = stlsq_node(&library, &target, t_eff, n_osc, threshold, max_iter)?;
            res_row[i] = xi[0];
            let mut feature = 1;
            for (j, value) in res_row.iter_mut().enumerate() {
                if j != i {
                    *value = xi[feature];
                    feature += 1;
                }
            }
            Ok(())
        },
    )?;

    if result.iter().any(|v| !v.is_finite()) {
        return Err("Phase-SINDy produced non-finite coefficients".to_string());
    }

    Ok(result)
}

fn validate_sindy_inputs(
    phases: &[f64],
    n_osc: usize,
    n_time: usize,
    dt: f64,
    threshold: f64,
    max_iter: usize,
) -> Result<(), String> {
    if n_osc == 0 {
        return Err("Phase-SINDy requires at least one oscillator".to_string());
    }
    if n_time < 2 {
        return Err("Phase-SINDy requires at least two time samples".to_string());
    }
    if n_time - 1 < n_osc {
        return Err("Phase-SINDy requires at least one derivative sample per feature".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("Phase-SINDy dt must be finite and positive".to_string());
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err("Phase-SINDy threshold must be finite and non-negative".to_string());
    }
    if max_iter == 0 {
        return Err("Phase-SINDy max_iter must be at least one".to_string());
    }
    let expected = n_osc
        .checked_mul(n_time)
        .ok_or_else(|| "Phase-SINDy input dimensions overflow".to_string())?;
    if phases.len() != expected {
        return Err(format!(
            "Phase-SINDy phase buffer length mismatch: {} != {}",
            phases.len(),
            expected
        ));
    }
    if phases.iter().any(|v| !v.is_finite()) {
        return Err("Phase-SINDy phases must be finite".to_string());
    }

    Ok(())
}

/// Compute θ̇ via finite differences with phase unwrapping.
fn compute_theta_dot(phases: &[f64], n_osc: usize, n_time: usize, dt: f64) -> Vec<f64> {
    let t_eff = n_time - 1;
    let mut theta_dot = vec![0.0; n_osc * t_eff];

    for i in 0..n_osc {
        let mut prev = phases[i];
        for tt in 0..t_eff {
            let curr = phases[(tt + 1) * n_osc + i];
            let raw_diff = curr - prev;
            let mut diff = (raw_diff + PI).rem_euclid(TAU) - PI;
            if diff == -PI && raw_diff > 0.0 {
                diff = PI;
            }
            if raw_diff.abs() < PI {
                diff = raw_diff;
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
        library[tt * n_osc] = 1.0; // match the Python constant-first library
        let mut feature = 1;
        for j in 0..n_osc {
            if j != i {
                let phi_j = phases[tt * n_osc + j];
                let phi_i = phases[tt * n_osc + i];
                library[tt * n_osc + feature] = (phi_j - phi_i).sin();
                feature += 1;
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
) -> Result<Vec<f64>, String> {
    let mut xi = lstsq(library, target, t_eff, n_features)?;

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
        let xi_red = lstsq(&lib_red, target, t_eff, n_big)?;
        let mut xi_new = vec![0.0; n_features];
        for (k, &feat_idx) in big.iter().enumerate() {
            xi_new[feat_idx] = xi_red[k];
        }
        xi = xi_new;
    }

    Ok(xi)
}

/// Solve the rectangular least-squares problem without squaring its condition number.
fn lstsq(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>, String> {
    if a.iter().any(|value| !value.is_finite()) {
        return Err("Phase-SINDy feature library must remain finite".to_string());
    }
    let matrix = DMatrix::from_row_slice(m, n, a);
    // Scale only the target: it leaves rank and the minimum-norm solution
    // unchanged while avoiding overflow in the orthogonal projection.
    let scale = b.iter().map(|value| value.abs()).fold(1.0_f64, f64::max);
    let target = DVector::from_iterator(m, b.iter().map(|value| value / scale));
    let decomposition = SVD::try_new(matrix, true, true, 5.0 * f64::EPSILON, 100_000)
        .ok_or_else(|| "Phase-SINDy least-squares SVD did not converge".to_string())?;
    let cutoff = f64::EPSILON * m.max(n) as f64 * decomposition.singular_values[0];
    let coefficients = decomposition
        .solve(&target, cutoff)
        .map_err(|error| format!("Phase-SINDy least-squares failed: {error}"))?;
    Ok(coefficients.iter().map(|value| value * scale).collect())
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
        for (i, phase) in phases.iter_mut().take(n).enumerate() {
            *phase = i as f64 * 0.5;
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
        let result = sindy_fit(&phases, n, t, dt, 0.05, 10).expect("valid SINDy fit");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    omegas[i]
                } else {
                    coupling[i * n + j]
                };
                assert!(
                    (result[i * n + j] - expected).abs() < 2e-8,
                    "coefficient[{i},{j}]: {} != {expected}",
                    result[i * n + j]
                );
            }
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
        let result = sindy_fit(&phases, n, t, dt, 0.1, 10).expect("valid SINDy fit");
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(result[i * n + j] == 0.0, "K[{i},{j}]={}", result[i * n + j]);
                }
            }
        }
    }

    #[test]
    fn test_short_data_rejects_underdetermined_fit() {
        let err = sindy_fit(&[0.0; 4], 2, 2, 0.01, 0.05, 10)
            .expect_err("underdetermined regression must fail closed");
        assert!(err.contains("derivative sample per feature"));
    }

    #[test]
    fn test_output_size() {
        let result = sindy_fit(&vec![0.5; 50 * 4], 4, 50, 0.01, 0.05, 10).expect("valid SINDy fit");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_rejects_invalid_controls_and_payloads() {
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(sindy_fit(&[0.0; 6], 2, 3, dt, 0.05, 10)
                .expect_err("invalid dt must fail closed")
                .contains("dt"));
        }
        for threshold in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(sindy_fit(&[0.0; 6], 2, 3, 0.01, threshold, 10)
                .expect_err("invalid threshold must fail closed")
                .contains("threshold"));
        }
        assert!(sindy_fit(&[0.0; 6], 2, 3, 0.01, 0.05, 0)
            .expect_err("zero iterations must fail closed")
            .contains("max_iter"));
        assert!(sindy_fit(
            &[0.0, f64::INFINITY, 0.0, 0.1, 0.2, 0.3],
            2,
            3,
            0.01,
            0.05,
            10
        )
        .expect_err("non-finite phases must fail closed")
        .contains("finite"));
        assert!(sindy_fit(&[0.0; 5], 2, 3, 0.01, 0.05, 10)
            .expect_err("shape mismatch must fail closed")
            .contains("length mismatch"));
    }

    #[test]
    fn test_multiple_turn_alias_has_principal_angular_velocity() {
        let phases: Vec<f64> = (0..8).map(|time| time as f64 * (0.1 + 3.0 * TAU)).collect();
        let result = sindy_fit(&phases, 1, 8, 0.1, 0.0, 3).expect("finite alias fit");
        assert!((result[0] - 1.0).abs() < 1e-11);
    }

    #[test]
    fn test_dependent_features_have_closed_form_minimum_norm_solution() {
        for (dt, offset) in [(0.01, 0.4), (0.015625, 0.5)] {
            let phases: Vec<f64> = (0..40)
                .flat_map(|time| [time as f64 * dt, time as f64 * dt + offset])
                .collect();
            let result = sindy_fit(&phases, 2, 40, dt, 0.0, 3).expect("finite rank-deficient fit");
            let sine = f64::sin(offset);
            let omega = 1.0 / (1.0 + sine * sine);
            for (actual, expected) in result
                .iter()
                .zip([omega, sine * omega, -sine * omega, omega])
            {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn test_half_turns_retain_the_original_increment_sign() {
        for increment in [PI, -PI, 3.0 * PI, -3.0 * PI] {
            let result =
                sindy_fit(&[0.0, increment], 1, 2, 1.0, 0.0, 1).expect("finite half-turn fit");
            assert!((result[0] - increment.signum() * PI).abs() < 1e-14);
        }
    }

    #[test]
    fn test_all_terms_below_threshold_return_zero_equation() {
        let result =
            sindy_fit(&[0.0, 0.1, 0.2], 1, 3, 1.0, 1.0, 2).expect("finite thresholded fit");
        assert_eq!(result, [0.0]);
    }

    #[test]
    fn test_non_finite_derived_arithmetic_is_rejected() {
        assert!(sindy_fit(&[0.0, 1.0], 1, 2, 1e-320, 0.0, 1)
            .expect_err("overflowing derivative must be refused")
            .contains("derivatives"));
        let phases = [-1e308, 1e308, -1e308, 1e308, -1e308, 1e308];
        assert!(sindy_fit(&phases, 2, 3, 1.0, 0.0, 1)
            .expect_err("overflowing feature must be refused")
            .contains("feature library"));
        let phases = [0.0, 1e-8, 0.1, 0.1 + 2e-8, 0.1, 0.1 + 3e-8];
        assert!(sindy_fit(&phases, 2, 3, 1e-309, 0.0, 1)
            .expect_err("overflowing solve output must be refused")
            .contains("non-finite coefficients"));
    }

    #[test]
    fn test_empty_dimensions_and_overflow_are_rejected_before_allocation() {
        assert!(sindy_fit(&[], 0, 3, 1.0, 0.0, 1)
            .expect_err("zero nodes must be refused")
            .contains("oscillator"));
        assert!(sindy_fit(&[0.0], 1, 1, 1.0, 0.0, 1)
            .expect_err("one sample must be refused")
            .contains("time samples"));
        assert!(sindy_fit(&[], usize::MAX / 2, usize::MAX, 1.0, 0.0, 1)
            .expect_err("dimension product must not overflow")
            .contains("dimensions overflow"));
    }
}
