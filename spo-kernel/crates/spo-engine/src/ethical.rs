// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — C15_sec ethical cost term

//! Ethical Lagrangian from R5 Insight 19:
//! L_ethical = U_total + w_c15 · C15_sec
//! C15_sec = (1 - J_sec) + phi_ethics, where phi_ethics = κ · Φ_ethics.
//!
//! J_sec = α·R + β·K_norm + γ·Q - ν·S_dev  (SEC functional)
//! Φ_ethics = Σ max(0, g_k)²                (CBF constraint penalties)
//!
//! This numerical diagnostic does not establish ethical compliance or safety.

use std::f64::consts::PI;

/// Compute C15_sec ethical cost term.
///
/// Returns `(j_sec, phi_ethics, c15_sec, n_violated)`.
/// Density counts exactly nonzero entries, including signed and diagonal
/// weights. Connectivity uses reciprocal magnitude averages without self-loops.
/// Dispersion is the population standard deviation of the raw phases divided
/// by pi. Finite signed parameters are supported; kappa weights the residual
/// sum once and does not change the strictly positive violation count.
///
/// # Errors
/// Rejects mismatched shapes, non-finite inputs or parameters, and arithmetic
/// that cannot produce finite constraint penalties and costs.
#[allow(clippy::too_many_arguments)]
pub fn compute_ethical_cost(
    phases: &[f64],
    knm: &[f64],
    n: usize,
    alpha_r: f64,
    beta_k: f64,
    gamma_q: f64,
    nu_s: f64,
    kappa: f64,
    r_min: f64,
    connectivity_min: f64,
    max_coupling: f64,
) -> Result<(f64, f64, f64, usize), String> {
    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| "ethical coupling dimension overflow".to_string())?;
    if phases.len() != n || knm.len() != matrix_len {
        return Err("ethical phases and coupling dimensions must match n".to_string());
    }
    if !phases.iter().chain(knm).all(|value| value.is_finite()) {
        return Err("ethical phases and coupling must contain only finite values".to_string());
    }
    if ![
        alpha_r,
        beta_k,
        gamma_q,
        nu_s,
        kappa,
        r_min,
        connectivity_min,
        max_coupling,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err("ethical weights and thresholds must be finite".to_string());
    }
    if n == 0 {
        return Ok((0.0, 0.0, 1.0, 0));
    }

    let (r, lam2, q, s_dev) = compute_sec_inputs(phases, knm, n)?;
    let k_norm = if n > 0 { lam2 / n as f64 } else { 0.0 };
    let j_sec = alpha_r * r + beta_k * k_norm + gamma_q * q - nu_s * s_dev;
    let (phi_ethics, n_violated) =
        compute_cbf_penalties(r, lam2, knm, kappa, r_min, connectivity_min, max_coupling)?;
    let c15_sec = (1.0 - j_sec) + phi_ethics;

    if ![j_sec, phi_ethics, c15_sec]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("ethical cost arithmetic must remain finite".to_string());
    }
    Ok((j_sec, phi_ethics, c15_sec, n_violated))
}

/// Compute SEC functional inputs: R, λ₂, Q (density), S_dev.
fn compute_sec_inputs(
    phases: &[f64],
    knm: &[f64],
    n: usize,
) -> Result<(f64, f64, f64, f64), String> {
    let sx: f64 = phases.iter().map(|p| p.sin()).sum();
    let cx: f64 = phases.iter().map(|p| p.cos()).sum();
    let r = sx.hypot(cx) / n as f64;
    let lam2 = fiedler_value_inline(knm, n)?;
    let n_nonzero = knm.iter().filter(|&&v| v != 0.0).count();
    let n_possible = n * (n - 1);
    let q = if n_possible > 0 {
        n_nonzero as f64 / n_possible as f64
    } else {
        0.0
    };
    let mean_phase = phases.iter().sum::<f64>() / n as f64;
    let var = phases.iter().map(|p| (p - mean_phase).powi(2)).sum::<f64>() / n as f64;
    let s_dev = var.sqrt() / PI;
    if ![r, lam2, q, s_dev].iter().all(|value| value.is_finite()) {
        return Err("ethical SEC arithmetic must remain finite".to_string());
    }
    Ok((r, lam2, q, s_dev))
}

/// Compute CBF constraint penalties.
fn compute_cbf_penalties(
    r: f64,
    lam2: f64,
    knm: &[f64],
    kappa: f64,
    r_min: f64,
    connectivity_min: f64,
    max_coupling: f64,
) -> Result<(f64, usize), String> {
    let g = [
        r_min - r,
        connectivity_min - lam2,
        if knm.iter().any(|&v| v > 0.0) {
            knm.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - max_coupling
        } else {
            0.0
        },
    ];
    if !g.iter().all(|value| value.is_finite()) {
        return Err("ethical constraint arithmetic must remain finite".to_string());
    }
    let phi: f64 = kappa * g.iter().map(|&gi| gi.max(0.0).powi(2)).sum::<f64>();
    let n_v = g.iter().filter(|&&gi| gi > 0.0).count();
    Ok((phi, n_v))
}

/// Inline Fiedler value (algebraic connectivity λ₂) computation.
///
/// Uses the graph Laplacian L = D - W where W_ij = |K_ij|.
/// λ₂ is the second-smallest eigenvalue of L.
///
/// For small n, uses Jacobi eigenvalue algorithm on the symmetric Laplacian.
fn fiedler_value_inline(knm: &[f64], n: usize) -> Result<f64, String> {
    if n < 2 {
        return Ok(0.0);
    }

    // Build the Laplacian of the symmetrised graph, matching the Python
    // coupling.spectral.fiedler_value: edge weight (|w_ij| + |w_ji|) / 2 and no
    // self-loops. The Jacobi solver below assumes a symmetric matrix; the raw
    // D - |W| of an asymmetric coupling gave a different lambda_2 than NumPy.
    let mut laplacian = vec![0.0; n * n];
    for i in 0..n {
        let mut deg = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let w = 0.5 * (knm[i * n + j].abs() + knm[j * n + i].abs());
            laplacian[i * n + j] = -w;
            deg += w;
        }
        laplacian[i * n + i] = deg;
    }

    if !laplacian.iter().all(|value| value.is_finite()) {
        return Err("ethical Laplacian arithmetic must remain finite".to_string());
    }
    // Find eigenvalues via iterative Jacobi rotation
    let eigs = jacobi_eigenvalues(&laplacian, n);
    if !eigs.iter().all(|value| value.is_finite()) {
        return Err("ethical eigenvalue arithmetic must remain finite".to_string());
    }
    // Second smallest eigenvalue
    if eigs.len() < 2 {
        return Ok(0.0);
    }
    let mut sorted = eigs;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(sorted[1].max(0.0))
}

/// Jacobi eigenvalue algorithm for symmetric matrix.
fn jacobi_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let scale = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return vec![0.0; n];
    }
    let mut mat: Vec<f64> = a.iter().map(|v| v / scale).collect();
    let max_iter = 100 * n * n;
    let eps = 8.0 * f64::EPSILON * n as f64;

    for _ in 0..max_iter {
        let (max_val, p, q) = find_max_offdiag(&mat, n);
        if max_val < eps {
            break;
        }
        jacobi_rotate(&mut mat, n, p, q);
    }

    (0..n).map(|i| mat[i * n + i] * scale).collect()
}

/// Find largest off-diagonal element and its position.
fn find_max_offdiag(mat: &[f64], n: usize) -> (f64, usize, usize) {
    let mut max_val = 0.0;
    let mut p = 0;
    let mut q = 1;
    for i in 0..n {
        for j in (i + 1)..n {
            let v = mat[i * n + j].abs();
            if v > max_val {
                max_val = v;
                p = i;
                q = j;
            }
        }
    }
    (max_val, p, q)
}

/// Apply one Jacobi rotation at position (p, q).
fn jacobi_rotate(mat: &mut [f64], n: usize, p: usize, q: usize) {
    let app = mat[p * n + p];
    let aqq = mat[q * n + q];
    let apq = mat[p * n + q];
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + tau.hypot(1.0))
    } else {
        -1.0 / (-tau + tau.hypot(1.0))
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;

    mat[p * n + p] = app - t * apq;
    mat[q * n + q] = aqq + t * apq;
    mat[p * n + q] = 0.0;
    mat[q * n + p] = 0.0;

    for r in 0..n {
        if r != p && r != q {
            let rp = mat[r * n + p];
            let rq = mat[r * n + q];
            mat[r * n + p] = c * rp - s * rq;
            mat[p * n + r] = c * rp - s * rq;
            mat[r * n + q] = s * rp + c * rq;
            mat[q * n + r] = s * rp + c * rq;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_phase_cannot_be_hidden_by_a_constraint_clamp() {
        let result = compute_ethical_cost(
            &[0.0, f64::NAN],
            &[0.0; 4],
            2,
            0.4,
            0.3,
            0.2,
            0.1,
            1.0,
            0.2,
            0.1,
            5.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn finite_coupling_with_unrepresentable_penalty_is_refused() {
        let result = compute_ethical_cost(
            &[0.0; 2],
            &[0.0, 1e200, 1e200, 0.0],
            2,
            0.4,
            0.3,
            0.2,
            0.1,
            1.0,
            0.2,
            0.1,
            5.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn malformed_dimensions_are_refused_before_matrix_access() {
        let result =
            compute_ethical_cost(&[0.0], &[0.0; 4], 2, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn overflowing_dimension_is_refused() {
        let result =
            compute_ethical_cost(&[], &[], usize::MAX, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
        assert!(result
            .expect_err("dimension overflow")
            .contains("dimension overflow"));
    }

    #[test]
    fn empty_state_still_validates_parameters() {
        let result = compute_ethical_cost(&[], &[], 0, f64::NAN, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
        assert!(result
            .expect_err("invalid parameter")
            .contains("must be finite"));
    }

    #[test]
    fn finite_inputs_with_nonfinite_derived_arithmetic_are_refused() {
        let cases = [
            (vec![1e200, -1e200], vec![0.0; 4], 2, 5.0, "SEC"),
            (
                vec![0.0; 2],
                vec![0.0, 1e308, 1e308, 0.0],
                2,
                5.0,
                "Laplacian",
            ),
            (vec![0.0], vec![1e308], 1, -1e308, "constraint"),
        ];
        for (phases, knm, n, max_coupling, expected) in cases {
            let result = compute_ethical_cost(
                &phases,
                &knm,
                n,
                0.4,
                0.3,
                0.2,
                0.1,
                1.0,
                0.2,
                0.1,
                max_coupling,
            );
            assert!(result
                .expect_err("unrepresentable arithmetic")
                .contains(expected));
        }
    }

    #[test]
    fn test_empty_phases() {
        let (j, phi, c15, n_v) =
            compute_ethical_cost(&[], &[], 0, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0)
                .expect("finite valid ethical fixture");
        assert_eq!(j, 0.0);
        assert_eq!(phi, 0.0);
        assert_eq!(c15, 1.0);
        assert_eq!(n_v, 0);
    }

    #[test]
    fn test_synchronised_high_coupling() {
        let n = 4;
        let phases = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 1.0;
                }
            }
        }
        let (j, _, c15, _) =
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0)
                .expect("finite valid ethical fixture");
        // R=1.0 (perfect sync), high J_sec → lower C15
        assert!(j > 0.5, "J_sec={j} should be > 0.5 for sync+coupling");
        assert!(c15 < 0.8, "C15={c15} should be < 0.8");
    }

    #[test]
    fn test_no_coupling_violation() {
        let n = 3;
        let phases = vec![0.0, 1.0, 2.0];
        let knm = vec![0.0; n * n]; // no coupling → λ₂=0
        let (_, _, _, n_v) =
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.5, 0.1, 5.0)
                .expect("finite valid ethical fixture");
        // connectivity_min=0.1 > λ₂=0 → violation
        assert!(n_v >= 1, "should violate connectivity constraint");
    }

    #[test]
    fn test_high_coupling_violation() {
        let n = 2;
        let phases = vec![0.0, 0.0];
        let knm = vec![0.0, 10.0, 10.0, 0.0]; // max=10 > max_coupling=5
        let (_, phi, _, n_v) =
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 5.0)
                .expect("finite valid ethical fixture");
        assert!(n_v >= 1, "should violate max coupling constraint");
        assert!(phi > 0.0, "phi={phi} should be > 0");
    }

    #[test]
    fn test_c15_decomposition() {
        // C15 = (1 - J_sec) + kappa * Φ
        let n = 3;
        let phases = vec![0.5; n];
        let knm = vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0];
        let (j, phi, c15, _) =
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0)
                .expect("finite valid ethical fixture");
        assert!(
            ((1.0 - j) + phi - c15).abs() < 1e-10,
            "C15 = (1-J) + Φ: j={j}, phi={phi}, c15={c15}"
        );
    }

    #[test]
    fn test_fiedler_complete_graph() {
        let n = 4;
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 1.0;
                }
            }
        }
        let (j, _, _, _) =
            compute_ethical_cost(&[0.0; 4], &knm, n, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
                .expect("finite public connectivity score");
        let lam2 = j * n as f64;
        // Complete graph K_n: λ₂ = n
        assert!((lam2 - n as f64).abs() < 0.1, "λ₂={lam2}, expected {n}");
    }

    #[test]
    fn test_fiedler_disconnected() {
        let n = 4;
        let knm = vec![0.0; n * n];
        let (j, _, _, _) =
            compute_ethical_cost(&[0.0; 4], &knm, n, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
                .expect("finite public connectivity score");
        let lam2 = j * n as f64;
        assert!(lam2 < 1e-10, "disconnected graph: λ₂={lam2} should be ~0");
    }

    #[test]
    fn test_fiedler_asymmetric_matches_symmetrised_graph() {
        // A directed coupling and its symmetrised, self-loop-free form describe
        // the same undirected graph, so lambda_2 must agree.
        let n = 3;
        let asym = vec![0.7, 2.0, 0.0, 0.0, 0.3, 1.0, 4.0, 0.0, 0.9];
        let mut sym = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sym[i * n + j] = 0.5 * (asym[i * n + j] + asym[j * n + i]);
                }
            }
        }
        let (a, _, _, _) =
            compute_ethical_cost(&[0.0; 3], &asym, n, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
                .expect("finite public asymmetric graph");
        let (b, _, _, _) =
            compute_ethical_cost(&[0.0; 3], &sym, n, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
                .expect("finite public symmetric graph");
        assert!(
            (a - b).abs() < 1e-9,
            "asymmetric λ₂={a} vs symmetrised λ₂={b}"
        );
    }
    #[test]
    fn density_counts_tiny_signed_and_diagonal_nonzero_weights() {
        for weight in [f64::from_bits(1), 1e-300, 1e-16, 1.0] {
            let result = compute_ethical_cost(
                &[0.0; 2],
                &[-weight; 4],
                2,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
            )
            .expect("finite signed density diagnostic");
            assert_eq!(result, (2.0, 0.0, -1.0, 0));
        }
        let result = compute_ethical_cost(
            &[0.0; 2],
            &[0.0, -0.0, -0.0, 0.0],
            2,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            5.0,
        )
        .expect("zero density diagnostic");
        assert_eq!(result, (0.0, 0.0, 1.0, 0));
    }

    #[test]
    fn connectivity_preserves_graph_scale() {
        for weight in [1e-300, 1e-16, 1.0, 1e100] {
            let (j, phi, c15, violations) = compute_ethical_cost(
                &[0.0; 2],
                &[0.0, -weight, -weight, 0.0],
                2,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
            )
            .expect("finite graph scale diagnostic");
            assert!((j / weight - 1.0).abs() < 1e-14, "scale {weight}: {j}");
            assert_eq!(phi, 0.0);
            assert_eq!(c15, 1.0 - j);
            assert_eq!(violations, 0);
        }
    }

    #[test]
    fn raw_phase_population_dispersion_is_not_wrapped() {
        let phases = [0.0, 2.0 * PI];
        let (j, phi, c15, violations) = compute_ethical_cost(
            &phases, &[0.0; 4], 2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0,
        )
        .expect("raw phase dispersion");
        assert!((j + 1.0).abs() < 1e-15);
        assert_eq!(phi, 0.0);
        assert!((c15 - 2.0).abs() < 1e-15);
        assert_eq!(violations, 0);
    }

    #[test]
    fn signed_multiplier_weights_the_penalty_once() {
        let (j, phi, c15, violations) = compute_ethical_cost(
            &[0.0],
            &[0.0],
            1,
            -0.4,
            0.0,
            0.0,
            0.0,
            -2.0,
            1.5,
            0.25,
            -10.0,
        )
        .expect("finite signed parameters");
        assert_eq!(j, -0.4);
        assert_eq!(phi, -2.0 * (0.5_f64.powi(2) + 0.25_f64.powi(2)));
        assert_eq!(c15, 1.0 - j + phi);
        assert_eq!(violations, 2);
    }

    #[test]
    fn constraint_equality_does_not_count_as_a_violation() {
        let result = compute_ethical_cost(
            &[0.0; 2],
            &[0.0, 0.5, 0.5, 0.0],
            2,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            1.0,
            1.0,
            0.5,
        )
        .expect("threshold equality");
        assert_eq!(result, (0.0, 0.0, 1.0, 0));
    }
    #[test]
    fn finite_degrees_with_unrepresentable_eigenvalues_are_refused() {
        let mut matrix = vec![-7e307; 9];
        for i in 0..3 {
            matrix[i * 3 + i] = 0.0;
        }
        let result = compute_ethical_cost(
            &[0.0; 3], &matrix, 3, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0,
        );
        assert!(result
            .expect_err("lambda2=3*7e307 exceeds binary64")
            .contains("eigenvalue arithmetic"));
    }
}
