// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — SSGF cost terms

//! SSGF cost terms for geometry optimisation.
//!
//! C1: synchronisation deficit (1 − R)
//! C2: negative algebraic connectivity (−λ₂)
//! C3: sparsity regulariser (‖W‖₁ / N²)
//! C4: symmetry deviation (‖W − Wᵀ‖_F / N)

use crate::order_params::compute_order_parameter;
use crate::spectral::fiedler_value;

/// SSGF cost result.
pub struct SSGFCostsResult {
    pub c1_sync: f64,
    pub c2_spectral_gap: f64,
    pub c3_sparsity: f64,
    pub c4_symmetry: f64,
    pub u_total: f64,
}

/// Compute SSGF cost terms.
///
/// # Arguments
/// * `w_flat` — (N×N) row-major geometry coupling matrix
/// * `phases` — (N,) current phases
/// * `n` — number of oscillators
/// * `weights` — (w1, w2, w3, w4) cost weights
#[must_use]
pub fn compute_ssgf_costs(
    w_flat: &[f64],
    phases: &[f64],
    n: usize,
    weights: (f64, f64, f64, f64),
) -> SSGFCostsResult {
    let (w1, w2, w3, w4) = weights;

    // C1: 1 − R (synchronisation deficit)
    let (r, _) = compute_order_parameter(phases);
    let c1 = 1.0 - r;

    // C2: −λ₂(L(W)) (negative algebraic connectivity)
    let lam2 = fiedler_value(w_flat, n);
    let c2 = -lam2;

    // C3: ‖W‖₁ / N² (sparsity regulariser)
    let c3 = if n > 0 {
        w_flat.iter().map(|v| v.abs()).sum::<f64>() / (n * n) as f64
    } else {
        0.0
    };

    // C4: ‖W − Wᵀ‖_F / N (symmetry deviation)
    let c4 = if n > 0 {
        let mut fro_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = w_flat[i * n + j] - w_flat[j * n + i];
                fro_sq += diff * diff;
            }
        }
        fro_sq.sqrt() / n as f64
    } else {
        0.0
    };

    let u_total = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4;

    SSGFCostsResult {
        c1_sync: c1,
        c2_spectral_gap: c2,
        c3_sparsity: c3,
        c4_symmetry: c4,
        u_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_perfect_sync_zero_c1() {
        // All phases equal → R = 1 → C1 = 0
        let n = 4;
        let w = vec![0.0; n * n];
        let phases = vec![0.0; n];
        let r = compute_ssgf_costs(&w, &phases, n, (1.0, 0.5, 0.1, 0.1));
        assert!(r.c1_sync.abs() < 1e-10);
    }

    #[test]
    fn test_desync_high_c1() {
        // Uniformly spread phases → R ≈ 0 → C1 ≈ 1
        let n = 100;
        let w = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let r = compute_ssgf_costs(&w, &phases, n, (1.0, 0.5, 0.1, 0.1));
        assert!(r.c1_sync > 0.9);
    }

    #[test]
    fn test_symmetric_w_zero_c4() {
        let n = 3;
        let w = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let phases = vec![0.0; n];
        let r = compute_ssgf_costs(&w, &phases, n, (1.0, 0.5, 0.1, 0.1));
        assert!(r.c4_symmetry < 1e-10);
    }

    #[test]
    fn test_asymmetric_w_positive_c4() {
        let n = 2;
        let w = vec![0.0, 1.0, 0.0, 0.0]; // asymmetric
        let phases = vec![0.0; n];
        let r = compute_ssgf_costs(&w, &phases, n, (1.0, 0.5, 0.1, 0.1));
        assert!(r.c4_symmetry > 0.0);
    }

    #[test]
    fn test_sparsity_increases_with_density() {
        let n = 3;
        let w_sparse = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let w_dense = vec![1.0; n * n];
        let phases = vec![0.0; n];
        let r_sparse = compute_ssgf_costs(&w_sparse, &phases, n, (1.0, 0.5, 0.1, 0.1));
        let r_dense = compute_ssgf_costs(&w_dense, &phases, n, (1.0, 0.5, 0.1, 0.1));
        assert!(r_dense.c3_sparsity > r_sparse.c3_sparsity);
    }

    #[test]
    fn test_u_total_weighted_sum() {
        let n = 3;
        let w = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let phases = vec![0.0, PI / 4.0, PI / 2.0];
        let weights = (1.0, 0.5, 0.1, 0.1);
        let r = compute_ssgf_costs(&w, &phases, n, weights);
        let expected = weights.0 * r.c1_sync
            + weights.1 * r.c2_spectral_gap
            + weights.2 * r.c3_sparsity
            + weights.3 * r.c4_symmetry;
        assert!((r.u_total - expected).abs() < 1e-10);
    }

    #[test]
    fn test_empty() {
        let r = compute_ssgf_costs(&[], &[], 0, (1.0, 0.5, 0.1, 0.1));
        assert_eq!(r.c3_sparsity, 0.0);
        assert_eq!(r.c4_symmetry, 0.0);
    }
}
