// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — C15_sec ethical cost term

//! Ethical Lagrangian from R5 Insight 19:
//! L_ethical = U_total + w_c15 · C15_sec
//! C15_sec = (1 - J_sec) + κ · Φ_ethics
//!
//! J_sec = α·R + β·K_norm + γ·Q - ν·S_dev  (SEC functional)
//! Φ_ethics = Σ max(0, g_k)²                (CBF constraint penalties)
//!
//! Grounded in: Harsanyi aggregation, MacAskill ECW,
//! Lyapunov/CBF safety, Wiener cybernetic ethics.

use std::f64::consts::PI;

/// Compute C15_sec ethical cost term.
///
/// Returns `(j_sec, phi_ethics, c15_sec, n_violated)`.
#[must_use]
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
) -> (f64, f64, f64, usize) {
    if n == 0 {
        return (0.0, 0.0, 1.0, 0);
    }

    let (r, lam2, q, s_dev) = compute_sec_inputs(phases, knm, n);
    let k_norm = if n > 0 { lam2 / n as f64 } else { 0.0 };
    let j_sec = alpha_r * r + beta_k * k_norm + gamma_q * q - nu_s * s_dev;
    let (phi_ethics, n_violated) =
        compute_cbf_penalties(r, lam2, knm, kappa, r_min, connectivity_min, max_coupling);
    let c15_sec = (1.0 - j_sec) + phi_ethics;

    (j_sec, phi_ethics, c15_sec, n_violated)
}

/// Compute SEC functional inputs: R, λ₂, Q (density), S_dev.
fn compute_sec_inputs(phases: &[f64], knm: &[f64], n: usize) -> (f64, f64, f64, f64) {
    let sx: f64 = phases.iter().map(|p| p.sin()).sum();
    let cx: f64 = phases.iter().map(|p| p.cos()).sum();
    let r = (sx * sx + cx * cx).sqrt() / n as f64;
    let lam2 = fiedler_value_inline(knm, n);
    let n_nonzero = knm.iter().filter(|&&v| v.abs() > 1e-15).count();
    let n_possible = n * (n - 1);
    let q = if n_possible > 0 {
        n_nonzero as f64 / n_possible as f64
    } else {
        0.0
    };
    let mean_phase = phases.iter().sum::<f64>() / n as f64;
    let var = phases.iter().map(|p| (p - mean_phase).powi(2)).sum::<f64>() / n as f64;
    let s_dev = var.sqrt() / PI;
    (r, lam2, q, s_dev)
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
) -> (f64, usize) {
    let g = [
        r_min - r,
        connectivity_min - lam2,
        if knm.iter().any(|&v| v > 0.0) {
            knm.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - max_coupling
        } else {
            0.0
        },
    ];
    let phi: f64 = kappa * g.iter().map(|&gi| gi.max(0.0).powi(2)).sum::<f64>();
    let n_v = g.iter().filter(|&&gi| gi > 0.0).count();
    (phi, n_v)
}

/// Inline Fiedler value (algebraic connectivity λ₂) computation.
///
/// Uses the graph Laplacian L = D - W where W_ij = |K_ij|.
/// λ₂ is the second-smallest eigenvalue of L.
///
/// For small n, uses Jacobi eigenvalue algorithm on the symmetric Laplacian.
fn fiedler_value_inline(knm: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }

    // Build Laplacian L = D - |W|
    let mut laplacian = vec![0.0; n * n];
    for i in 0..n {
        let mut deg = 0.0;
        for j in 0..n {
            let w = knm[i * n + j].abs();
            laplacian[i * n + j] = -w;
            deg += w;
        }
        laplacian[i * n + i] = deg;
    }

    // Find eigenvalues via iterative Jacobi rotation
    let eigs = jacobi_eigenvalues(&laplacian, n);
    // Second smallest eigenvalue
    if eigs.len() < 2 {
        return 0.0;
    }
    let mut sorted = eigs;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[1].max(0.0)
}

/// Jacobi eigenvalue algorithm for symmetric matrix.
fn jacobi_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let mut mat = a.to_vec();
    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        let (max_val, p, q) = find_max_offdiag(&mat, n);
        if max_val < eps {
            break;
        }
        jacobi_rotate(&mut mat, n, p, q);
    }

    (0..n).map(|i| mat[i * n + i]).collect()
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
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
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
    fn test_empty_phases() {
        let (j, phi, c15, n_v) =
            compute_ethical_cost(&[], &[], 0, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
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
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
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
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.5, 0.1, 5.0);
        // connectivity_min=0.1 > λ₂=0 → violation
        assert!(n_v >= 1, "should violate connectivity constraint");
    }

    #[test]
    fn test_high_coupling_violation() {
        let n = 2;
        let phases = vec![0.0, 0.0];
        let knm = vec![0.0, 10.0, 10.0, 0.0]; // max=10 > max_coupling=5
        let (_, phi, _, n_v) =
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 5.0);
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
            compute_ethical_cost(&phases, &knm, n, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0);
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
        let lam2 = fiedler_value_inline(&knm, n);
        // Complete graph K_n: λ₂ = n
        assert!((lam2 - n as f64).abs() < 0.1, "λ₂={lam2}, expected {n}");
    }

    #[test]
    fn test_fiedler_disconnected() {
        let n = 4;
        let knm = vec![0.0; n * n];
        let lam2 = fiedler_value_inline(&knm, n);
        assert!(lam2 < 1e-10, "disconnected graph: λ₂={lam2} should be ~0");
    }
}
