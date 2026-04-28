// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Basin stability (Menck et al. 2013)

//! Basin stability estimation for Kuramoto synchronisation.
//!
//! Monte Carlo estimation of the synchronised state's basin of attraction.
//!
//! References:
//!   Menck, Heitzig, Marwan & Kurths 2013, Nature Physics 9:89-92.
//!   Ji, Peron, Rodrigues & Kurths 2014, Sci. Reports 4:4783.

use crate::bifurcation::steady_state_r;
use std::f64::consts::TAU;

/// Basin stability result.
pub struct BasinStabilityResult {
    /// Fraction of ICs that converged to synchronised state.
    pub s_b: f64,
    /// Final R for each sample.
    pub r_finals: Vec<f64>,
    /// Number of converged samples.
    pub n_converged: usize,
}

/// Estimate basin stability of the synchronised state.
///
/// Draws n_samples random ICs from [0, 2π)^N, integrates each to
/// steady state, and checks if R_final ≥ r_threshold.
///
/// Uses LCG PRNG for deterministic randomness without external deps.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn basin_stability(
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    n_samples: usize,
    r_threshold: f64,
    seed: u64,
) -> BasinStabilityResult {
    let mut r_finals = Vec::with_capacity(n_samples);
    let mut rng_state = seed;
    let mut n_converged = 0usize;

    for _ in 0..n_samples {
        // Generate random initial phases in [0, 2π)
        let mut phases_init = Vec::with_capacity(n);
        for _ in 0..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            phases_init.push(u * TAU);
        }

        let r = steady_state_r(
            &phases_init,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            1.0, // k_scale = 1.0 (knm already scaled)
            dt,
            n_transient,
            n_measure,
        );
        if r >= r_threshold {
            n_converged += 1;
        }
        r_finals.push(r);
    }

    let s_b = if n_samples > 0 {
        n_converged as f64 / n_samples as f64
    } else {
        0.0
    };

    BasinStabilityResult {
        s_b,
        r_finals,
        n_converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_all_to_all(n: usize) -> Vec<f64> {
        let mut knm = vec![1.0 / n as f64; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        knm
    }

    #[test]
    fn test_basin_stability_strong_coupling() {
        // Strong coupling → most ICs converge → S_B high
        let n = 4;
        let omegas = vec![1.0; n];
        let knm: Vec<f64> = make_all_to_all(n).iter().map(|&v| v * 10.0).collect();
        let alpha = vec![0.0; n * n];

        let result = basin_stability(&omegas, &knm, &alpha, n, 0.01, 500, 200, 20, 0.8, 42);
        assert!(
            result.s_b > 0.5,
            "strong coupling should give S_B > 0.5, got {}",
            result.s_b
        );
        assert_eq!(result.r_finals.len(), 20);
    }

    #[test]
    fn test_basin_stability_zero_coupling() {
        // Zero coupling → no sync → S_B should be low
        let n = 6;
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];

        let result = basin_stability(&omegas, &knm, &alpha, n, 0.01, 300, 100, 30, 0.8, 42);
        assert!(
            result.s_b < 0.3,
            "zero coupling should give low S_B, got {}",
            result.s_b
        );
    }

    #[test]
    fn test_basin_stability_deterministic() {
        // Same seed → same result
        let n = 4;
        let omegas = vec![1.0, 1.5, 2.0, 2.5];
        let knm: Vec<f64> = make_all_to_all(n).iter().map(|&v| v * 5.0).collect();
        let alpha = vec![0.0; n * n];

        let r1 = basin_stability(&omegas, &knm, &alpha, n, 0.01, 200, 100, 10, 0.5, 123);
        let r2 = basin_stability(&omegas, &knm, &alpha, n, 0.01, 200, 100, 10, 0.5, 123);
        assert_eq!(r1.s_b, r2.s_b);
        assert_eq!(r1.n_converged, r2.n_converged);
    }

    #[test]
    fn test_basin_stability_bounds() {
        let n = 3;
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];

        let result = basin_stability(&omegas, &knm, &alpha, n, 0.01, 200, 100, 15, 0.5, 42);
        assert!(result.s_b >= 0.0 && result.s_b <= 1.0);
        assert!(result.n_converged <= 15);
        for &r in &result.r_finals {
            assert!(r >= 0.0 && r <= 1.0 + 1e-10, "R = {r} out of range");
        }
    }

    #[test]
    fn test_basin_stability_empty() {
        let result = basin_stability(&[], &[], &[], 0, 0.01, 100, 50, 0, 0.8, 42);
        assert_eq!(result.s_b, 0.0);
        assert_eq!(result.r_finals.len(), 0);
    }

    #[test]
    fn test_lower_threshold_higher_stability() {
        // Lower R_threshold → more samples "converge" → higher S_B
        let n = 4;
        let omegas = vec![1.0, 2.0, 3.0, 4.0];
        let knm: Vec<f64> = make_all_to_all(n).iter().map(|&v| v * 3.0).collect();
        let alpha = vec![0.0; n * n];

        let r_low = basin_stability(&omegas, &knm, &alpha, n, 0.01, 300, 100, 20, 0.3, 42);
        let r_high = basin_stability(&omegas, &knm, &alpha, n, 0.01, 300, 100, 20, 0.9, 42);
        assert!(
            r_low.s_b >= r_high.s_b,
            "lower threshold should give higher S_B: {:.2} vs {:.2}",
            r_low.s_b,
            r_high.s_b
        );
    }
}
