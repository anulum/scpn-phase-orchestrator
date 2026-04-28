// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Spectral graph analysis (Dörfler & Bullo 2014)

//! Spectral analysis of coupling networks via graph Laplacian eigendecomposition.
//!
//! Uses the cyclic Jacobi eigenvalue algorithm (Golub & Van Loan 2013, §8.4)
//! for symmetric matrices. No external linear algebra dependency.
//!
//! References:
//! - Dörfler & Bullo 2014, Automatica 50(6):1539-1564.
//! - Fiedler 1973, Czech. Math. J. 23:298-305 (algebraic connectivity).
//! - Golub & Van Loan 2013, Matrix Computations, 4th ed., §8.4.

use rayon::prelude::*;

/// Spectral decomposition result: sorted eigenvalues + eigenvectors.
pub struct SpectralResult {
    /// Eigenvalues sorted ascending: λ_1 ≤ λ_2 ≤ ... ≤ λ_N.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as column-major N×N matrix.
    /// Column i corresponds to eigenvalue i.
    pub eigenvectors: Vec<f64>,
}

/// Compute the combinatorial graph Laplacian L = D - |W|.
///
/// D = diag(row sums of |W|), diagonal of W is zeroed first.
///
/// # Arguments
/// * `knm` - row-major N×N coupling matrix
/// * `n` - matrix dimension
///
/// # Returns
/// Row-major N×N Laplacian.
#[must_use]
pub fn graph_laplacian(knm: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    l.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let offset = i * n;
        let mut degree = 0.0;
        for j in 0..n {
            if i != j {
                let w = knm[offset + j].abs();
                row[j] = -w;
                degree += w;
            }
        }
        row[i] = degree;
    });
    l
}

/// Full spectral decomposition of a symmetric matrix via cyclic Jacobi.
///
/// Golub & Van Loan 2013, Matrix Computations, 4th ed., Algorithm 8.4.3.
/// Converges quadratically for distinct eigenvalues.
///
/// # Arguments
/// * `matrix` - row-major N×N symmetric matrix (copied internally)
/// * `n` - matrix dimension
/// * `max_sweeps` - maximum number of Jacobi sweeps (typically 20-50 suffice)
/// * `tol` - convergence tolerance on off-diagonal Frobenius norm
///
/// # Returns
/// `SpectralResult` with eigenvalues sorted ascending and corresponding eigenvectors.
#[must_use]
pub fn symmetric_eigen(matrix: &[f64], n: usize, max_sweeps: usize, tol: f64) -> SpectralResult {
    if n == 0 {
        return SpectralResult {
            eigenvalues: vec![],
            eigenvectors: vec![],
        };
    }
    if n == 1 {
        return SpectralResult {
            eigenvalues: vec![matrix[0]],
            eigenvectors: vec![1.0],
        };
    }

    // Work on a copy (row-major)
    let mut a = matrix.to_vec();
    // Eigenvector accumulator (starts as identity)
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        // Check convergence: off-diagonal Frobenius norm
        let mut off_norm = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[i * n + j] * a[i * n + j];
            }
        }
        off_norm = off_norm.sqrt();
        if off_norm < tol {
            break;
        }

        // Cyclic sweep over all (i, j) pairs with i < j
        for i in 0..n {
            for j in (i + 1)..n {
                let a_ij = a[i * n + j];
                if a_ij.abs() < 1e-15 {
                    continue;
                }

                // Compute Jacobi rotation angle (Golub & Van Loan §8.4.1)
                let a_ii = a[i * n + i];
                let a_jj = a[j * n + j];
                let tau = (a_jj - a_ii) / (2.0 * a_ij);

                // t = sign(τ) / (|τ| + √(1 + τ²)) — Rutishauser's formula
                let t = if tau.abs() < 1e-15 {
                    1.0
                } else {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt(); // cos θ
                let s = t * c; // sin θ

                // Apply Jacobi rotation to A: A' = G^T A G
                // Update rows/columns i and j
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    let a_ki = a[k * n + i];
                    let a_kj = a[k * n + j];
                    a[k * n + i] = c * a_ki - s * a_kj;
                    a[k * n + j] = s * a_ki + c * a_kj;
                    a[i * n + k] = a[k * n + i]; // symmetric
                    a[j * n + k] = a[k * n + j];
                }

                // Update diagonal and off-diagonal (i,j)
                let new_ii = c * c * a_ii - 2.0 * s * c * a_ij + s * s * a_jj;
                let new_jj = s * s * a_ii + 2.0 * s * c * a_ij + c * c * a_jj;
                a[i * n + i] = new_ii;
                a[j * n + j] = new_jj;
                a[i * n + j] = 0.0;
                a[j * n + i] = 0.0;

                // Accumulate eigenvectors: V' = V G
                for k in 0..n {
                    let v_ki = v[k * n + i];
                    let v_kj = v[k * n + j];
                    v[k * n + i] = c * v_ki - s * v_kj;
                    v[k * n + j] = s * v_ki + c * v_kj;
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of A) and sort ascending
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let eval = a[i * n + i];
            let evec: Vec<f64> = (0..n).map(|k| v[k * n + i]).collect();
            (eval, evec)
        })
        .collect();
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(e, _)| *e).collect();
    let mut eigenvectors = vec![0.0_f64; n * n];
    for (col, (_, evec)) in eigen_pairs.iter().enumerate() {
        for (row, &val) in evec.iter().enumerate() {
            eigenvectors[row * n + col] = val;
        }
    }

    SpectralResult {
        eigenvalues,
        eigenvectors,
    }
}

/// Algebraic connectivity λ₂(L) — Fiedler value.
///
/// Fiedler 1973, Czech. Math. J. 23:298-305.
/// Dörfler & Bullo 2014, Automatica 50(6):1539-1564.
#[must_use]
pub fn fiedler_value(knm: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let l = graph_laplacian(knm, n);
    let result = symmetric_eigen(&l, n, 50, 1e-12);
    result.eigenvalues[1]
}

/// Fiedler vector — eigenvector of λ₂(L).
///
/// Sign partitions the network into synchronisation clusters.
#[must_use]
pub fn fiedler_vector(knm: &[f64], n: usize) -> Vec<f64> {
    if n < 2 {
        return vec![0.0; n];
    }
    let l = graph_laplacian(knm, n);
    let result = symmetric_eigen(&l, n, 50, 1e-12);
    // Column 1 of eigenvectors (second eigenvector)
    (0..n).map(|i| result.eigenvectors[i * n + 1]).collect()
}

/// Spectral gap λ₃ - λ₂.
///
/// Larger gap → cleaner two-cluster structure.
#[must_use]
pub fn spectral_gap(knm: &[f64], n: usize) -> f64 {
    if n < 3 {
        return 0.0;
    }
    let l = graph_laplacian(knm, n);
    let result = symmetric_eigen(&l, n, 50, 1e-12);
    result.eigenvalues[2] - result.eigenvalues[1]
}

/// Dörfler-Bullo critical coupling: K_c = max|ω_i - ω_j| / λ₂(L).
///
/// Synchronisation requires K > K_c. Returns f64::INFINITY if disconnected.
#[must_use]
pub fn critical_coupling(omegas: &[f64], knm: &[f64], n: usize) -> f64 {
    let lambda2 = fiedler_value(knm, n);
    if lambda2 < 1e-12 {
        return f64::INFINITY;
    }
    let omega_max = omegas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let omega_min = omegas.iter().cloned().fold(f64::INFINITY, f64::min);
    (omega_max - omega_min) / lambda2
}

/// Estimated synchronisation convergence rate.
///
/// μ = K_eff · λ₂ · cos(γ_max) / N
///
/// Dörfler & Bullo 2014, §III.B.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sync_convergence_rate(knm: &[f64], omegas: &[f64], n: usize, gamma_max: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let lambda2 = fiedler_value(knm, n);
    let mut sum_positive = 0.0_f64;
    let mut count_positive = 0usize;
    for i in 0..n {
        for j in 0..n {
            let v = knm[i * n + j];
            if v > 0.0 {
                sum_positive += v;
                count_positive += 1;
            }
        }
    }
    let k_eff = if count_positive > 0 {
        sum_positive / count_positive as f64
    } else {
        0.0
    };
    let _ = omegas; // used for API consistency with Python
    k_eff * lambda2 * gamma_max.cos() / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_knm(n: usize, k: f64) -> Vec<f64> {
        let mut knm = vec![k; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        knm
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let knm = uniform_knm(4, 1.0);
        let l = graph_laplacian(&knm, 4);
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| l[i * 4 + j]).sum();
            assert!(row_sum.abs() < 1e-12, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_laplacian_symmetric() {
        let knm = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let l = graph_laplacian(&knm, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (l[i * 3 + j] - l[j * 3 + i]).abs() < 1e-12,
                    "L not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_fiedler_value_connected() {
        // Complete graph K_4 with weight 1: λ₂ = N = 4
        let knm = uniform_knm(4, 1.0);
        let f = fiedler_value(&knm, 4);
        assert!((f - 4.0).abs() < 0.01, "K_4 Fiedler = {} (expected 4)", f);
    }

    #[test]
    fn test_fiedler_value_disconnected() {
        // Two disconnected pairs: λ₂ should be 0
        let knm = vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let f = fiedler_value(&knm, 4);
        assert!(f.abs() < 0.01, "disconnected Fiedler = {} (expected ~0)", f);
    }

    #[test]
    fn test_fiedler_vector_partition() {
        // Two clusters: {0,1} coupled, {2,3} coupled, weak cross-coupling
        let mut knm = vec![0.0; 16];
        knm[0 * 4 + 1] = 5.0;
        knm[1 * 4 + 0] = 5.0;
        knm[2 * 4 + 3] = 5.0;
        knm[3 * 4 + 2] = 5.0;
        knm[1 * 4 + 2] = 0.1;
        knm[2 * 4 + 1] = 0.1;
        let v2 = fiedler_vector(&knm, 4);
        // Nodes 0,1 should have same sign; 2,3 same sign; different from 0,1
        assert!(
            v2[0].signum() == v2[1].signum(),
            "nodes 0,1 should cluster: v2={:?}",
            v2
        );
        assert!(
            v2[2].signum() == v2[3].signum(),
            "nodes 2,3 should cluster: v2={:?}",
            v2
        );
        assert!(
            v2[0].signum() != v2[2].signum(),
            "clusters should separate: v2={:?}",
            v2
        );
    }

    #[test]
    fn test_spectral_gap() {
        let knm = uniform_knm(4, 1.0);
        let gap = spectral_gap(&knm, 4);
        // K_4: eigenvalues of L are 0, 4, 4, 4 → gap = 0
        assert!(gap.abs() < 0.01, "K_4 gap = {} (expected 0)", gap);
    }

    #[test]
    fn test_critical_coupling_finite() {
        let knm = uniform_knm(4, 1.0);
        let omegas = vec![0.8, 0.9, 1.0, 1.1];
        let kc = critical_coupling(&omegas, &knm, 4);
        assert!(kc.is_finite(), "K_c should be finite for connected graph");
        assert!(kc > 0.0, "K_c should be positive");
    }

    #[test]
    fn test_critical_coupling_disconnected() {
        let knm = vec![0.0; 9]; // no coupling
        let omegas = vec![1.0, 2.0, 3.0];
        let kc = critical_coupling(&omegas, &knm, 3);
        assert!(kc.is_infinite(), "K_c should be inf for disconnected");
    }

    #[test]
    fn test_convergence_rate_positive() {
        let knm = uniform_knm(4, 2.0);
        let omegas = vec![1.0; 4];
        let mu = sync_convergence_rate(&knm, &omegas, 4, 0.0);
        assert!(mu > 0.0, "convergence rate should be positive: {}", mu);
    }

    #[test]
    fn test_jacobi_identity_matrix() {
        // Identity should have eigenvalues all 1
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = symmetric_eigen(&id, 3, 50, 1e-12);
        for &e in &result.eigenvalues {
            assert!((e - 1.0).abs() < 1e-10, "identity eigenvalue {} != 1", e);
        }
    }

    #[test]
    fn test_jacobi_known_spectrum() {
        // Diagonal matrix with known eigenvalues
        let diag = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let result = symmetric_eigen(&diag, 3, 50, 1e-12);
        assert!((result.eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((result.eigenvalues[1] - 2.0).abs() < 1e-10);
        assert!((result.eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_eigenvectors_orthogonal() {
        let knm = uniform_knm(4, 1.0);
        let l = graph_laplacian(&knm, 4);
        let result = symmetric_eigen(&l, 4, 50, 1e-12);
        // Check orthogonality of eigenvectors
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dot: f64 = (0..4)
                    .map(|k| result.eigenvectors[k * 4 + i] * result.eigenvectors[k * 4 + j])
                    .sum();
                assert!(dot.abs() < 1e-8, "v{} · v{} = {} (expected 0)", i, j, dot);
            }
        }
    }

    #[test]
    fn test_empty() {
        assert_eq!(fiedler_value(&[], 0), 0.0);
        assert!(fiedler_vector(&[], 0).is_empty());
        assert_eq!(spectral_gap(&[], 0), 0.0);
    }
}
