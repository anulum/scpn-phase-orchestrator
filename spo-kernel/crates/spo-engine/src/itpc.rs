// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Inter-Trial Phase Coherence (Lachaux et al. 1999)

//! Inter-Trial Phase Coherence (ITPC) for phase-locked stimulus detection.
//!
//! ITPC = |mean(exp(iθ))| across trials at each time point.
//! Lachaux, Rodriguez, Martinerie & Varela 1999, Human Brain Mapping 8:194-208.

/// Compute ITPC at each time point from a (n_trials × n_timepoints) phase matrix.
///
/// ITPC_t = |1/K Σ_{k=1}^{K} exp(i θ_{k,t})|
///
/// where K = n_trials, θ_{k,t} is the phase of trial k at time t.
///
/// # Arguments
/// * `phases_flat` — row-major flattened (n_trials × n_timepoints) phase array in radians
/// * `n_trials` — number of trials (rows)
/// * `n_timepoints` — number of time points (columns)
///
/// # Returns
/// Vec of length n_timepoints with ITPC values in [0, 1].
#[must_use]
pub fn compute_itpc(phases_flat: &[f64], n_trials: usize, n_timepoints: usize) -> Vec<f64> {
    if n_trials == 0 || n_timepoints == 0 {
        return vec![];
    }
    if n_trials == 1 {
        return vec![1.0; n_timepoints];
    }

    let inv_k = 1.0 / n_trials as f64;
    let mut result = Vec::with_capacity(n_timepoints);

    for t in 0..n_timepoints {
        let mut sum_cos = 0.0_f64;
        let mut sum_sin = 0.0_f64;
        for k in 0..n_trials {
            let theta = phases_flat[k * n_timepoints + t];
            // Use sin/cos directly — more numerically stable than complex arithmetic
            sum_cos += theta.cos();
            sum_sin += theta.sin();
        }
        let mean_cos = sum_cos * inv_k;
        let mean_sin = sum_sin * inv_k;
        result.push((mean_cos * mean_cos + mean_sin * mean_sin).sqrt());
    }

    result
}

/// ITPC measured at specified pause indices (post-stimulus persistence check).
///
/// Returns the mean ITPC across the given time-point indices.
/// Indices outside `[0, n_timepoints)` are silently skipped.
///
/// # Arguments
/// * `phases_flat` — row-major flattened (n_trials × n_timepoints)
/// * `n_trials` — number of trials
/// * `n_timepoints` — number of time points
/// * `pause_indices` — time-point indices to average over
#[must_use]
pub fn itpc_persistence(
    phases_flat: &[f64],
    n_trials: usize,
    n_timepoints: usize,
    pause_indices: &[usize],
) -> f64 {
    if pause_indices.is_empty() || n_trials == 0 || n_timepoints == 0 {
        return 0.0;
    }

    let itpc = compute_itpc(phases_flat, n_trials, n_timepoints);

    let mut sum = 0.0;
    let mut count = 0usize;
    for &idx in pause_indices {
        if idx < n_timepoints {
            sum += itpc[idx];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    #[test]
    fn test_single_trial_returns_ones() {
        let phases = vec![0.0, 1.0, 2.0, 3.0];
        let itpc = compute_itpc(&phases, 1, 4);
        assert_eq!(itpc.len(), 4);
        for v in &itpc {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_identical_phases_perfect_coherence() {
        // All trials have same phase at each time → ITPC = 1.0
        let n_trials = 50;
        let n_tp = 10;
        let phase_val = 1.5;
        let phases: Vec<f64> = vec![phase_val; n_trials * n_tp];
        let itpc = compute_itpc(&phases, n_trials, n_tp);
        assert_eq!(itpc.len(), n_tp);
        for v in &itpc {
            assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_uniform_phases_near_zero_coherence() {
        // Phases uniformly spaced around the circle → ITPC ≈ 0
        let n_trials = 360;
        let n_tp = 1;
        let phases: Vec<f64> = (0..n_trials)
            .map(|k| TAU * k as f64 / n_trials as f64)
            .collect();
        let itpc = compute_itpc(&phases, n_trials, n_tp);
        assert!(itpc[0] < 0.02, "expected near 0, got {}", itpc[0]);
    }

    #[test]
    fn test_two_clusters_half_coherence() {
        // Half trials at 0, half at π → ITPC = |mean| = 0
        // (opposite phases cancel exactly)
        let n_trials = 100;
        let n_tp = 1;
        let mut phases = vec![0.0; n_trials];
        for k in n_trials / 2..n_trials {
            phases[k] = PI;
        }
        let itpc = compute_itpc(&phases, n_trials, n_tp);
        assert!(itpc[0] < 1e-12, "opposite phases should cancel, got {}", itpc[0]);
    }

    #[test]
    fn test_itpc_range_bounded() {
        // Random phases: ITPC should be in [0, 1]
        let n_trials = 200;
        let n_tp = 50;
        let mut phases = Vec::with_capacity(n_trials * n_tp);
        // Simple LCG for deterministic randomness
        let mut x: u64 = 12345;
        for _ in 0..n_trials * n_tp {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            phases.push((x >> 33) as f64 / (1u64 << 31) as f64 * TAU);
        }
        let itpc = compute_itpc(&phases, n_trials, n_tp);
        for (t, v) in itpc.iter().enumerate() {
            assert!(*v >= 0.0 && *v <= 1.0 + 1e-12, "ITPC[{t}] = {v} out of range");
        }
    }

    #[test]
    fn test_empty_returns_empty() {
        assert!(compute_itpc(&[], 0, 0).is_empty());
        assert!(compute_itpc(&[], 0, 10).is_empty());
    }

    #[test]
    fn test_persistence_basic() {
        // Perfect coherence → persistence at any index = 1.0
        let n_trials = 20;
        let n_tp = 10;
        let phases = vec![0.5; n_trials * n_tp];
        let p = itpc_persistence(&phases, n_trials, n_tp, &[2, 5, 8]);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_persistence_empty_indices() {
        let phases = vec![0.0; 100];
        assert_eq!(itpc_persistence(&phases, 10, 10, &[]), 0.0);
    }

    #[test]
    fn test_persistence_out_of_bounds_skipped() {
        let n_trials = 10;
        let n_tp = 5;
        let phases = vec![0.0; n_trials * n_tp];
        // Index 100 is out of bounds, should be skipped
        let p = itpc_persistence(&phases, n_trials, n_tp, &[0, 100]);
        assert!((p - 1.0).abs() < 1e-12, "only valid index 0 should count");
    }

    #[test]
    fn test_persistence_all_out_of_bounds() {
        let phases = vec![0.0; 50];
        assert_eq!(itpc_persistence(&phases, 5, 10, &[20, 30]), 0.0);
    }

    #[test]
    fn test_monotone_coherence_with_noise() {
        // Phases with increasing noise → ITPC should generally decrease
        let n_trials = 500;
        let n_tp = 5;
        let mut phases = Vec::with_capacity(n_trials * n_tp);
        let mut x: u64 = 42;
        for k in 0..n_trials {
            for t in 0..n_tp {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise_scale = (t + 1) as f64 * 0.5; // increasing noise
                let noise = ((x >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * noise_scale;
                phases.push(1.0 + noise);
            }
        }
        let itpc = compute_itpc(&phases, n_trials, n_tp);
        // First time point (lowest noise) should have highest ITPC
        assert!(itpc[0] > itpc[n_tp - 1], "ITPC should decrease with noise");
    }
}
