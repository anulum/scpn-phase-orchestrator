// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Transfer entropy directed adaptive coupling

//! Adapt coupling matrix using transfer entropy as learning signal.
//!
//! K_ij(t+1) = (1-decay) · K_ij(t) + lr · TE(i→j)
//!
//! Lizier 2012, "Local Information Transfer as a Spatiotemporal Filter
//! for Complex Systems," Physical Review E 77(2):026110.

/// Adapt coupling matrix given a TE matrix.
///
/// `knm_new = (1 - decay) * knm + lr * te`, clamped to non-negative,
/// diagonal zeroed.
///
/// # Arguments
/// * `knm` – current coupling matrix (row-major, n×n)
/// * `te` – transfer entropy matrix (row-major, n×n)
/// * `n` – number of oscillators
/// * `lr` – learning rate
/// * `decay` – coupling decay rate
///
/// # Returns
/// Updated coupling matrix (row-major, n×n).
#[must_use]
pub fn te_adapt_coupling(knm: &[f64], te: &[f64], n: usize, lr: f64, decay: f64) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            if i == j {
                result[idx] = 0.0;
            } else {
                let val = (1.0 - decay) * knm[idx] + lr * te[idx];
                result[idx] = val.max(0.0);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_always_zero() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let te = vec![1.0; n * n];
        let result = te_adapt_coupling(&knm, &te, n, 0.5, 0.0);
        for i in 0..n {
            assert_eq!(result[i * n + i], 0.0);
        }
    }

    #[test]
    fn test_no_decay_adds_te() {
        let n = 2;
        let knm = vec![0.0, 1.0, 0.5, 0.0];
        let te = vec![0.0, 0.2, 0.3, 0.0];
        let result = te_adapt_coupling(&knm, &te, n, 0.5, 0.0);
        // K_01 = 1.0 + 0.5*0.2 = 1.1
        assert!((result[1] - 1.1).abs() < 1e-10);
        // K_10 = 0.5 + 0.5*0.3 = 0.65
        assert!((result[2] - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_full_decay() {
        let n = 2;
        let knm = vec![0.0, 1.0, 0.5, 0.0];
        let te = vec![0.0, 0.0, 0.0, 0.0];
        let result = te_adapt_coupling(&knm, &te, n, 0.0, 1.0);
        // Full decay, no TE → all off-diagonal = 0
        for i in 0..n {
            for j in 0..n {
                assert_eq!(result[i * n + j], 0.0);
            }
        }
    }

    #[test]
    fn test_clamp_non_negative() {
        let n = 2;
        let knm = vec![0.0, 0.1, 0.1, 0.0];
        let te = vec![0.0, -10.0, -10.0, 0.0]; // negative TE
        let result = te_adapt_coupling(&knm, &te, n, 1.0, 0.0);
        for val in &result {
            assert!(*val >= 0.0, "val={val} should be >= 0");
        }
    }

    #[test]
    fn test_preserves_asymmetry() {
        let n = 3;
        let knm = vec![0.0; n * n];
        let te = vec![0.0, 0.5, 0.1, 0.2, 0.0, 0.8, 0.0, 0.3, 0.0];
        let result = te_adapt_coupling(&knm, &te, n, 1.0, 0.0);
        // K_01 ≠ K_10
        assert!((result[1] - result[3]).abs() > 0.1);
    }

    #[test]
    fn test_zero_lr_no_change() {
        let n = 2;
        let knm = vec![0.0, 0.5, 0.3, 0.0];
        let te = vec![0.0, 1.0, 1.0, 0.0];
        let result = te_adapt_coupling(&knm, &te, n, 0.0, 0.0);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.3).abs() < 1e-10);
    }
}
