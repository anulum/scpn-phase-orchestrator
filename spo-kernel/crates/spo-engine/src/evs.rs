// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Entrainment Verification Score (EVS)

//! Combines ITPC, persistence, and frequency specificity into
//! a single entrainment verification score.

/// Compute frequency specificity ratio: ITPC at target / ITPC at control.
///
/// `phases_flat` – row-major (n_trials × n_timepoints) phases at target freq.
/// Control phases are rescaled by `control_freq / target_freq`.
///
/// Returns the ratio target_mean / control_mean, or `f64::INFINITY` if control ≈ 0.
#[must_use]
pub fn frequency_specificity(
    phases_flat: &[f64],
    n_trials: usize,
    n_timepoints: usize,
    target_freq: f64,
    control_freq: f64,
) -> f64 {
    if target_freq <= 0.0 || control_freq <= 0.0 || n_trials == 0 || n_timepoints == 0 {
        return 0.0;
    }

    // ITPC at target frequency
    let target_itpc = mean_itpc(phases_flat, n_trials, n_timepoints);

    // Phase at control frequency: rescale by freq ratio
    let ratio = control_freq / target_freq;
    let control_phases: Vec<f64> = phases_flat.iter().map(|&p| p * ratio).collect();
    let control_itpc = mean_itpc(&control_phases, n_trials, n_timepoints);

    if control_itpc < 1e-12 {
        return if target_itpc > 0.0 { f64::INFINITY } else { 0.0 };
    }

    target_itpc / control_itpc
}

/// Compute mean ITPC across timepoints from trial-by-timepoint phases.
fn mean_itpc(phases_flat: &[f64], n_trials: usize, n_timepoints: usize) -> f64 {
    if n_trials == 0 || n_timepoints == 0 {
        return 0.0;
    }
    let inv_trials = 1.0 / n_trials as f64;
    let mut itpc_sum = 0.0;

    for t in 0..n_timepoints {
        let mut sx = 0.0;
        let mut cx = 0.0;
        for trial in 0..n_trials {
            let p = phases_flat[trial * n_timepoints + t];
            sx += p.sin();
            cx += p.cos();
        }
        itpc_sum += (sx * sx + cx * cx).sqrt() * inv_trials;
    }

    itpc_sum / n_timepoints as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_perfect_sync_high_specificity() {
        // All trials at same phase → ITPC=1 at target
        let n_trials = 10;
        let n_tp = 20;
        let phases: Vec<f64> = vec![0.5; n_trials * n_tp];
        let spec = frequency_specificity(&phases, n_trials, n_tp, 10.0, 5.0);
        // Same phases, different freq ratio → control phases different → ratio > 1
        assert!(spec >= 1.0, "spec={spec}");
    }

    #[test]
    fn test_random_phases_low_specificity() {
        // Pseudorandom phases → ITPC ≈ 0 at both → ratio ≈ 1 or undefined
        let n_trials = 50;
        let n_tp = 30;
        let mut phases = vec![0.0; n_trials * n_tp];
        let mut seed = 42u64;
        for p in phases.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *p = (seed as f64 / u64::MAX as f64) * 2.0 * PI;
        }
        let spec = frequency_specificity(&phases, n_trials, n_tp, 10.0, 7.0);
        // Both ITITPs low → ratio close to 1
        assert!(spec < 3.0, "spec={spec} should be modest for random phases");
    }

    #[test]
    fn test_zero_freqs() {
        assert_eq!(frequency_specificity(&[1.0], 1, 1, 0.0, 10.0), 0.0);
        assert_eq!(frequency_specificity(&[1.0], 1, 1, 10.0, 0.0), 0.0);
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(frequency_specificity(&[], 0, 0, 10.0, 5.0), 0.0);
    }

    #[test]
    fn test_mean_itpc_synchronised() {
        let n_trials = 5;
        let n_tp = 10;
        let phases = vec![1.0; n_trials * n_tp]; // all same phase
        let itpc = mean_itpc(&phases, n_trials, n_tp);
        assert!((itpc - 1.0).abs() < 1e-10, "itpc={itpc}");
    }

    #[test]
    fn test_mean_itpc_uniform() {
        // Evenly spaced phases → ITPC ≈ 0
        let n_trials = 100;
        let n_tp = 1;
        let phases: Vec<f64> = (0..n_trials)
            .map(|i| 2.0 * PI * i as f64 / n_trials as f64)
            .collect();
        let itpc = mean_itpc(&phases, n_trials, n_tp);
        assert!(itpc < 0.05, "itpc={itpc} should be near 0");
    }
}
