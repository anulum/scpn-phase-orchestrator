// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Universal coupling prior

//! Domain-agnostic coupling prior from 25-domainpack empirical distribution.
//!
//! K_base ~ N(0.47, 0.09), decay_alpha ~ N(0.25, 0.07).
//! Combined with Dörfler-Bullo K_c, collapses auto-tune from 5D to 2D.
//!
//! Source: R4-A3 cross-domain transfer analysis (Stankovski 2017, Rev. Mod. Phys.).

/// Unnormalised log-probability under the Gaussian prior.
///
/// lp = -0.5 · ((K - μ_K) / σ_K)² - 0.5 · ((α - μ_α) / σ_α)²
#[must_use]
pub fn log_probability(
    k_base: f64,
    decay_alpha: f64,
    k_mean: f64,
    k_std: f64,
    alpha_mean: f64,
    alpha_std: f64,
) -> f64 {
    let lp_k = -0.5 * ((k_base - k_mean) / k_std).powi(2);
    let lp_a = -0.5 * ((decay_alpha - alpha_mean) / alpha_std).powi(2);
    lp_k + lp_a
}

/// Build a distance-decay coupling matrix: K_base · exp(-decay_alpha · |i-j|).
///
/// Returns row-major (n×n) matrix with zero diagonal.
#[must_use]
pub fn distance_decay_matrix(n: usize, k_base: f64, decay_alpha: f64) -> Vec<f64> {
    let mut knm = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dist = i.abs_diff(j);
                knm[i * n + j] = k_base * (-decay_alpha * dist as f64).exp();
            }
        }
    }
    knm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_probability_at_mean() {
        let lp = log_probability(0.47, 0.25, 0.47, 0.09, 0.25, 0.07);
        assert!((lp - 0.0).abs() < 1e-10, "at mean, lp should be 0");
    }

    #[test]
    fn test_log_probability_decreases_away() {
        let at_mean = log_probability(0.47, 0.25, 0.47, 0.09, 0.25, 0.07);
        let away = log_probability(1.0, 0.25, 0.47, 0.09, 0.25, 0.07);
        assert!(away < at_mean);
    }

    #[test]
    fn test_log_probability_symmetric() {
        let above = log_probability(0.56, 0.25, 0.47, 0.09, 0.25, 0.07);
        let below = log_probability(0.38, 0.25, 0.47, 0.09, 0.25, 0.07);
        assert!((above - below).abs() < 1e-10);
    }

    #[test]
    fn test_distance_decay_diagonal_zero() {
        let n = 4;
        let knm = distance_decay_matrix(n, 0.5, 0.3);
        for i in 0..n {
            assert_eq!(knm[i * n + i], 0.0);
        }
    }

    #[test]
    fn test_distance_decay_symmetric() {
        let n = 5;
        let knm = distance_decay_matrix(n, 0.47, 0.25);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (knm[i * n + j] - knm[j * n + i]).abs() < 1e-15,
                    "K[{i},{j}] != K[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_distance_decay_decreasing() {
        let n = 10;
        let knm = distance_decay_matrix(n, 1.0, 0.5);
        // K[0,1] > K[0,2] > K[0,3]
        assert!(knm[1] > knm[2]);
        assert!(knm[2] > knm[3]);
    }

    #[test]
    fn test_distance_decay_values() {
        let n = 3;
        let knm = distance_decay_matrix(n, 1.0, 0.0);
        // With decay=0, all off-diagonal = K_base = 1.0
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!((knm[i * n + j] - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}
