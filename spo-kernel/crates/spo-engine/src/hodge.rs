// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hodge decomposition of coupling dynamics

//! Hodge decomposition of coupling flow into gradient, curl, and harmonic parts.
//!
//! Jiang, Lim, Yao & Ye 2011, Math. Program. 127(1):203-244.

/// Hodge decomposition result: (gradient, curl, harmonic), each Vec<f64> of length N.
#[must_use]
pub fn hodge_decomposition(
    knm_flat: &[f64],
    phases: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![], vec![]);
    }

    let mut gradient = vec![0.0; n];
    let mut curl = vec![0.0; n];
    let mut harmonic = vec![0.0; n];

    for i in 0..n {
        let mut g = 0.0;
        let mut c = 0.0;
        let mut total = 0.0;
        for j in 0..n {
            let cos_diff = (phases[j] - phases[i]).cos();
            let k_ij = knm_flat[i * n + j];
            let k_ji = knm_flat[j * n + i];

            // K_sym = (K + K^T) / 2, K_anti = (K - K^T) / 2
            let k_sym = 0.5 * (k_ij + k_ji);
            let k_anti = 0.5 * (k_ij - k_ji);

            total += k_ij * cos_diff;
            g += k_sym * cos_diff;
            c += k_anti * cos_diff;
        }
        gradient[i] = g;
        curl[i] = c;
        harmonic[i] = total - g - c;
    }

    (gradient, curl, harmonic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_symmetric_coupling_no_curl() {
        // Symmetric K → curl = 0
        let n = 3;
        let knm = vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.8, 0.5, 0.8, 0.0];
        let phases = vec![0.0, PI / 4.0, PI / 2.0];
        let (_, curl, _) = hodge_decomposition(&knm, &phases, n);
        for (i, &c) in curl.iter().enumerate() {
            assert!(
                c.abs() < 1e-12,
                "curl[{i}] = {c}, expected 0 for symmetric K"
            );
        }
    }

    #[test]
    fn test_antisymmetric_coupling_no_gradient() {
        // Antisymmetric K → gradient = 0
        let n = 3;
        let knm = vec![0.0, 1.0, -0.5, -1.0, 0.0, 0.8, 0.5, -0.8, 0.0];
        let phases = vec![0.0, PI / 3.0, PI];
        let (gradient, _, _) = hodge_decomposition(&knm, &phases, n);
        for (i, &g) in gradient.iter().enumerate() {
            assert!(
                g.abs() < 1e-12,
                "gradient[{i}] = {g}, expected 0 for antisymmetric K"
            );
        }
    }

    #[test]
    fn test_total_equals_gradient_plus_curl() {
        // gradient + curl + harmonic = total (by construction, harmonic ≈ 0)
        let n = 4;
        let knm: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.3).sin()).collect();
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.7).collect();
        let (g, c, h) = hodge_decomposition(&knm, &phases, n);
        for i in 0..n {
            assert!(h[i].abs() < 1e-10, "harmonic[{i}] = {}, should be ~0", h[i]);
            // Check total = g + c + h
            let mut total = 0.0;
            for j in 0..n {
                total += knm[i * n + j] * (phases[j] - phases[i]).cos();
            }
            let sum = g[i] + c[i] + h[i];
            assert!((total - sum).abs() < 1e-10, "decomposition mismatch at {i}");
        }
    }

    #[test]
    fn test_empty() {
        let (g, c, h) = hodge_decomposition(&[], &[], 0);
        assert!(g.is_empty());
        assert!(c.is_empty());
        assert!(h.is_empty());
    }

    #[test]
    fn test_single_oscillator() {
        let (g, c, h) = hodge_decomposition(&[0.0], &[1.0], 1);
        assert_eq!(g.len(), 1);
        assert!((g[0]).abs() < 1e-12);
        assert!((c[0]).abs() < 1e-12);
        assert!((h[0]).abs() < 1e-12);
    }

    #[test]
    fn test_identical_phases_gradient_equals_total() {
        // All phases equal → cos(0) = 1 → gradient = K_sym row sums
        let n = 3;
        let knm = vec![0.0, 2.0, 3.0, 2.0, 0.0, 1.0, 3.0, 1.0, 0.0];
        let phases = vec![0.0; n]; // all same
        let (g, c, _) = hodge_decomposition(&knm, &phases, n);
        // Symmetric K → gradient = row sums, curl = 0
        assert!((g[0] - 5.0).abs() < 1e-10); // 0 + 2 + 3
        assert!((g[1] - 3.0).abs() < 1e-10); // 2 + 0 + 1
        assert!((g[2] - 4.0).abs() < 1e-10); // 3 + 1 + 0
        for &cv in &c {
            assert!(cv.abs() < 1e-12);
        }
    }
}
