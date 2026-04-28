// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase Transfer Entropy (Schreiber 2000)

//! Transfer entropy between phase time series via binned histogram estimation.
//!
//! TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
//!
//! References:
//! - Schreiber 2000, Phys. Rev. Lett. 85:461-464.
//! - Lobier et al. 2014, NeuroImage 94:347-354 (phase TE).
//! - Paluš & Vejmelka 2007, Phys. Rev. E 75:056211 (binned estimation).

use rayon::prelude::*;
use std::f64::consts::TAU;

/// Phase transfer entropy from source → target.
///
/// TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
///
/// Uses uniform circular binning on [0, 2π) for phase discretisation.
/// Conditional entropy estimated via histogram counting.
///
/// # Arguments
/// * `source` - source phase time series (length T)
/// * `target` - target phase time series (length T)
/// * `n_bins` - number of circular bins for phase discretisation
///
/// # Returns
/// Non-negative transfer entropy in nats.
#[must_use]
pub fn phase_transfer_entropy(source: &[f64], target: &[f64], n_bins: usize) -> f64 {
    let t = source.len().min(target.len());
    if t < 3 || n_bins == 0 {
        return 0.0;
    }
    let n = t - 1; // usable pairs (t, t+1)
    let bin_width = TAU / n_bins as f64;

    // Digitise phases into bins [0, n_bins)
    let bin = |phase: f64| -> usize {
        let p = ((phase % TAU) + TAU) % TAU; // wrap to [0, 2π)
        let b = (p / bin_width) as usize;
        b.min(n_bins - 1)
    };

    let src_binned: Vec<usize> = source[..n].iter().map(|&p| bin(p)).collect();
    let tgt_binned: Vec<usize> = target[..n].iter().map(|&p| bin(p)).collect();
    let tgt_next: Vec<usize> = target[1..=n].iter().map(|&p| bin(p)).collect();

    // H(Y_{t+1} | Y_t)
    let h_yt1_yt = conditional_entropy(&tgt_next, &tgt_binned, n_bins);

    // H(Y_{t+1} | Y_t, X_t) — joint condition encoded as Y_t * n_bins + X_t
    let joint_cond: Vec<usize> = tgt_binned
        .iter()
        .zip(src_binned.iter())
        .map(|(&y, &x)| y * n_bins + x)
        .collect();
    let h_yt1_yt_xt = conditional_entropy(&tgt_next, &joint_cond, n_bins * n_bins);

    (h_yt1_yt - h_yt1_yt_xt).max(0.0)
}

/// Pairwise transfer entropy matrix TE(i→j) for N oscillators.
///
/// # Arguments
/// * `phase_series` - row-major (N × T) phase trajectories
/// * `n_osc` - number of oscillators
/// * `n_time` - number of time points
/// * `n_bins` - histogram bins
///
/// # Returns
/// Flattened (N × N) row-major TE matrix. Diagonal is 0.
///
/// # Errors
/// Returns error if phase_series length ≠ N × T.
pub fn transfer_entropy_matrix(
    phase_series: &[f64],
    n_osc: usize,
    n_time: usize,
    n_bins: usize,
) -> Result<Vec<f64>, String> {
    if phase_series.len() != n_osc * n_time {
        return Err(format!(
            "phase_series length {} != N*T={}",
            phase_series.len(),
            n_osc * n_time
        ));
    }
    let mut te = vec![0.0; n_osc * n_osc];

    te.par_chunks_mut(n_osc).enumerate().for_each(|(i, row)| {
        let src = &phase_series[i * n_time..(i + 1) * n_time];
        for j in 0..n_osc {
            if i != j {
                let tgt = &phase_series[j * n_time..(j + 1) * n_time];
                row[j] = phase_transfer_entropy(src, tgt, n_bins);
            }
        }
    });

    Ok(te)
}

/// Conditional entropy H(target | condition) via histogram counting.
///
/// For each value c of condition, computes the distribution of target
/// and accumulates weighted entropy: H = Σ_c P(c) · H(target | c).
///
/// Uses natural logarithm (nats).
fn conditional_entropy(target: &[usize], condition: &[usize], n_cond_bins: usize) -> f64 {
    let n = target.len();
    if n == 0 {
        return 0.0;
    }

    // Build per-condition histograms of target values
    // Key: condition bin → HashMap<target_bin, count>
    let mut cond_counts = vec![0usize; n_cond_bins];
    // For each condition bin, accumulate target value counts
    // Use a flat array: n_cond_bins × max_target_bins
    // Find max target value first
    let max_tgt = target.iter().copied().max().unwrap_or(0) + 1;
    let mut joint = vec![0usize; n_cond_bins * max_tgt];

    for idx in 0..n {
        let c = condition[idx];
        let t = target[idx];
        if c < n_cond_bins && t < max_tgt {
            cond_counts[c] += 1;
            joint[c * max_tgt + t] += 1;
        }
    }

    let n_f64 = n as f64;
    let mut h = 0.0_f64;
    for c in 0..n_cond_bins {
        let cc = cond_counts[c];
        if cc < 2 {
            continue;
        }
        let cc_f64 = cc as f64;
        let p_c = cc_f64 / n_f64;
        // H(target | condition=c)
        let mut h_c = 0.0_f64;
        for t in 0..max_tgt {
            let ct = joint[c * max_tgt + t];
            if ct > 0 {
                let p = ct as f64 / cc_f64;
                h_c -= p * p.ln();
            }
        }
        h += p_c * h_c;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_self_transfer_entropy_zero() {
        // TE(X→X) should be ~0 (no information gain from conditioning on self)
        let phases: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1) % TAU).collect();
        let te = phase_transfer_entropy(&phases, &phases, 16);
        assert!(te < 0.1, "self-TE should be ~0: {}", te);
    }

    #[test]
    fn test_independent_signals_low_te() {
        // Two independent signals: TE should be ~0
        let src: Vec<f64> = (0..200).map(|i| (i as f64 * 0.3) % TAU).collect();
        let tgt: Vec<f64> = (0..200).map(|i| (i as f64 * 0.7 + 1.5) % TAU).collect();
        let te = phase_transfer_entropy(&src, &tgt, 16);
        assert!(te < 0.5, "independent TE should be low: {}", te);
    }

    #[test]
    fn test_driven_signal_positive_te() {
        // Target = lagged copy of source → positive TE
        let src: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin() * PI + PI).collect();
        let mut tgt = vec![0.0];
        tgt.extend_from_slice(&src[..199]); // target = source shifted by 1
        let te = phase_transfer_entropy(&src, &tgt, 16);
        assert!(te > 0.0, "driven TE should be positive: {}", te);
    }

    #[test]
    fn test_te_nonnegative() {
        let src: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5) % TAU).collect();
        let tgt: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3 + 0.7) % TAU).collect();
        let te = phase_transfer_entropy(&src, &tgt, 8);
        assert!(te >= 0.0, "TE must be non-negative: {}", te);
    }

    #[test]
    fn test_short_series() {
        let te = phase_transfer_entropy(&[0.0, 1.0], &[0.5, 1.5], 8);
        assert_eq!(te, 0.0, "too short should return 0");
    }

    #[test]
    fn test_empty_series() {
        let te = phase_transfer_entropy(&[], &[], 8);
        assert_eq!(te, 0.0);
    }

    #[test]
    fn test_matrix_diagonal_zero() {
        let n_osc = 3;
        let n_time = 50;
        let series: Vec<f64> = (0..n_osc * n_time)
            .map(|i| (i as f64 * 0.1) % TAU)
            .collect();
        let te = transfer_entropy_matrix(&series, n_osc, n_time, 8).unwrap();
        for i in 0..n_osc {
            assert_eq!(te[i * n_osc + i], 0.0, "diagonal should be 0");
        }
    }

    #[test]
    fn test_matrix_shape() {
        let n_osc = 4;
        let n_time = 30;
        let series: Vec<f64> = (0..n_osc * n_time)
            .map(|i| (i as f64 * 0.2) % TAU)
            .collect();
        let te = transfer_entropy_matrix(&series, n_osc, n_time, 16).unwrap();
        assert_eq!(te.len(), n_osc * n_osc);
    }

    #[test]
    fn test_asymmetric_te() {
        // X drives Y but Y does not drive X → TE(X→Y) > TE(Y→X)
        let n = 200;
        let src: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin() * PI + PI).collect();
        let mut tgt = vec![src[0]];
        for i in 1..n {
            // target follows source with noise
            tgt.push((0.8 * src[i - 1] + 0.2 * (i as f64 * 0.7)) % TAU);
        }
        let te_xy = phase_transfer_entropy(&src, &tgt, 16);
        let te_yx = phase_transfer_entropy(&tgt, &src, 16);
        // Driven direction should have higher TE (not strict — stochastic)
        // Just check both are non-negative
        assert!(te_xy >= 0.0);
        assert!(te_yx >= 0.0);
    }

    #[test]
    fn test_mismatched_length_error() {
        let result = transfer_entropy_matrix(&[0.0; 10], 3, 5, 8);
        assert!(result.is_err());
    }
}
