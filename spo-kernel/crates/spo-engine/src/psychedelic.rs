// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Psychedelic entropy (Carhart-Harris et al. 2014)

//! Circular Shannon entropy for psychedelic simulation monitoring.
//!
//! Carhart-Harris et al. 2014, Front. Hum. Neurosci. 8:20:
//! "The entropic brain: a theory of conscious states informed by
//! neuroimaging research with psychedelic drugs."

use std::f64::consts::TAU;

/// Circular Shannon entropy of a phase distribution.
///
/// Discretises phases into `n_bins` equal bins on [0, 2π)
/// and computes H = −Σ p_i ln(p_i) in nats.
///
/// # Arguments
/// * `phases` — phase values in radians (arbitrary range, wrapped to [0, 2π))
/// * `n_bins` — number of histogram bins (default 36 = 10° resolution)
#[must_use]
pub fn entropy_from_phases(phases: &[f64], n_bins: usize) -> f64 {
    if phases.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let mut counts = vec![0u64; n_bins];
    let bin_width = TAU / n_bins as f64;

    for &phase in phases {
        let wrapped = ((phase % TAU) + TAU) % TAU; // wrap to [0, 2π)
        let bin = (wrapped / bin_width) as usize;
        let bin = bin.min(n_bins - 1); // clamp edge case
        counts[bin] += 1;
    }

    let n = phases.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Scale coupling matrix by (1 - reduction_factor).
///
/// Models psychedelic-induced reduction in serotonergic gating.
#[must_use]
pub fn reduce_coupling(knm: &[f64], reduction_factor: f64) -> Vec<f64> {
    let scale = 1.0 - reduction_factor;
    knm.iter().map(|&v| v * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_entropy_empty() {
        assert_eq!(entropy_from_phases(&[], 36), 0.0);
    }

    #[test]
    fn test_entropy_single_point() {
        // Single point → all mass in one bin → H = 0
        assert_eq!(entropy_from_phases(&[1.0], 36), 0.0);
    }

    #[test]
    fn test_entropy_uniform_maximum() {
        // Uniformly distributed phases → maximum entropy = ln(n_bins)
        let n_bins = 36;
        let mut phases = Vec::new();
        for i in 0..n_bins {
            // Place 100 phases per bin
            let center = TAU * i as f64 / n_bins as f64 + 0.01;
            for _ in 0..100 {
                phases.push(center);
            }
        }
        let h = entropy_from_phases(&phases, n_bins);
        let h_max = (n_bins as f64).ln();
        assert!((h - h_max).abs() < 0.01, "expected H ≈ {h_max}, got {h}");
    }

    #[test]
    fn test_entropy_concentrated_low() {
        // All phases in one bin → H = 0
        let phases = vec![1.0; 1000];
        let h = entropy_from_phases(&phases, 36);
        assert!(h < 1e-12, "concentrated phases should give H ≈ 0, got {h}");
    }

    #[test]
    fn test_entropy_two_bins() {
        // Equal split between two bins → H = ln(2)
        let mut phases = Vec::new();
        for _ in 0..500 {
            phases.push(0.1); // bin 0
        }
        for _ in 0..500 {
            phases.push(PI); // bin 18 (for 36 bins)
        }
        let h = entropy_from_phases(&phases, 36);
        let expected = (2.0_f64).ln();
        assert!(
            (h - expected).abs() < 0.01,
            "expected H ≈ {expected}, got {h}"
        );
    }

    #[test]
    fn test_entropy_negative_phases() {
        // Negative phases should be wrapped correctly
        let phases = vec![-PI, -PI / 2.0, 0.0, PI / 2.0, PI];
        let h = entropy_from_phases(&phases, 36);
        assert!(h > 0.0, "entropy should be positive for spread phases");
    }

    #[test]
    fn test_entropy_nonnegative() {
        // Shannon entropy is always ≥ 0
        let phases: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).sin() * 10.0).collect();
        let h = entropy_from_phases(&phases, 36);
        assert!(h >= 0.0, "entropy must be non-negative, got {h}");
    }

    #[test]
    fn test_entropy_bounded_by_log_bins() {
        // H ≤ ln(n_bins) always
        let n_bins = 36;
        let h_max = (n_bins as f64).ln();
        let phases: Vec<f64> = (0..10000).map(|i| TAU * i as f64 / 10000.0).collect();
        let h = entropy_from_phases(&phases, n_bins);
        assert!(h <= h_max + 1e-10, "H = {h} exceeds ln({n_bins}) = {h_max}");
    }

    #[test]
    fn test_reduce_coupling_zero() {
        let knm = vec![1.0, 2.0, 3.0, 4.0];
        let result = reduce_coupling(&knm, 0.0);
        assert_eq!(result, knm);
    }

    #[test]
    fn test_reduce_coupling_full() {
        let knm = vec![1.0, 2.0, 3.0, 4.0];
        let result = reduce_coupling(&knm, 1.0);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_reduce_coupling_half() {
        let knm = vec![2.0, 4.0];
        let result = reduce_coupling(&knm, 0.5);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
    }
}
