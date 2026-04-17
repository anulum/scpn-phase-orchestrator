// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase-amplitude coupling

//!
//! Tort et al. 2010, J. Neurophysiol.: Modulation Index via KL divergence
//! from uniform distribution of phase-binned amplitudes.

use std::f64::consts::TAU;

/// Modulation Index (MI) between a low-frequency phase signal and a
/// high-frequency amplitude signal. Tort et al. 2010.
///
/// Bins amplitudes by phase, computes KL divergence from uniform,
/// normalised by ln(n_bins). Returns MI ∈ [0, 1].
#[must_use]
pub fn modulation_index(theta_low: &[f64], amp_high: &[f64], n_bins: usize) -> f64 {
    if theta_low.is_empty() || amp_high.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let len = theta_low.len().min(amp_high.len());
    let bin_width = TAU / n_bins as f64;

    let mut bin_sum = vec![0.0f64; n_bins];
    let mut bin_count = vec![0u64; n_bins];

    for idx in 0..len {
        let phase = theta_low[idx].rem_euclid(TAU);
        let mut bin = (phase / bin_width) as usize;
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        bin_sum[bin] += amp_high[idx];
        bin_count[bin] += 1;
    }

    let mut mean_amp = vec![0.0f64; n_bins];
    for k in 0..n_bins {
        if bin_count[k] > 0 {
            mean_amp[k] = bin_sum[k] / bin_count[k] as f64;
        }
    }

    let total: f64 = mean_amp.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let log_n = (n_bins as f64).ln();
    let mut kl = 0.0f64;
    for &ma in &mean_amp {
        let pk = ma / total;
        if pk > 0.0 {
            kl += pk * (pk * n_bins as f64).ln();
        }
    }

    (kl / log_n).clamp(0.0, 1.0)
}

/// N×N PAC matrix: entry [i*n+j] = MI(phase_column_i, amplitude_column_j).
///
/// Input `phases` and `amplitudes` are **row-major** flattened (T, N) arrays
/// of length T*N — element at (row, col) sits at index `row * n + col`. This
/// matches numpy's default `ravel()` order (C order). Returns a row-major
/// N×N vector (entry [i*n+j] is MI from column i phases against column j
/// amplitudes).
#[must_use]
pub fn pac_matrix(
    phases: &[f64],
    amplitudes: &[f64],
    t: usize,
    n: usize,
    n_bins: usize,
) -> Vec<f64> {
    let expected = t * n;
    if phases.len() < expected || amplitudes.len() < expected || t == 0 || n == 0 {
        return vec![0.0; n * n];
    }

    let mut result = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..n {
            // Extract column i from phases, column j from amplitudes
            // Row-major: element (row, col) at index row * n + col.
            let phase_col: Vec<f64> = (0..t).map(|row| phases[row * n + i]).collect();
            let amp_col: Vec<f64> = (0..t).map(|row| amplitudes[row * n + j]).collect();
            result[i * n + j] = modulation_index(&phase_col, &amp_col, n_bins);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_zero_for_uniform() {
        // Uniform amplitude across all phase bins → MI ≈ 0
        let n = 1000;
        let theta: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let amp = vec![1.0; n];
        let mi = modulation_index(&theta, &amp, 18);
        assert!(mi < 0.05, "MI={mi} should be near 0 for uniform");
    }

    #[test]
    fn mi_bounded_0_1() {
        let theta: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).rem_euclid(TAU)).collect();
        let amp: Vec<f64> = theta.iter().map(|&t| t.cos().abs()).collect();
        let mi = modulation_index(&theta, &amp, 18);
        assert!((0.0..=1.0).contains(&mi), "MI={mi} out of [0,1]");
    }

    #[test]
    fn mi_empty_input() {
        assert_eq!(modulation_index(&[], &[1.0], 18), 0.0);
        assert_eq!(modulation_index(&[1.0], &[], 18), 0.0);
        assert_eq!(modulation_index(&[], &[], 18), 0.0);
    }

    #[test]
    fn pac_matrix_shape() {
        let n = 3;
        let t = 100;
        let phases = vec![0.5; t * n];
        let amps = vec![1.0; t * n];
        let mat = pac_matrix(&phases, &amps, t, n, 18);
        assert_eq!(mat.len(), n * n);
    }

    #[test]
    fn pac_matrix_diagonal_entrained() {
        // Phase-locked signal: amplitude peaks at specific phase → MI > 0
        let n = 2;
        let t = 2000;
        let mut phases = vec![0.0; t * n];
        let mut amps = vec![0.0; t * n];
        for row in 0..t {
            let theta = TAU * row as f64 / t as f64;
            phases[row * n] = theta;
            phases[row * n + 1] = theta * 2.0;
            // Amplitude modulated by own phase
            amps[row * n] = 1.0 + 0.8 * theta.cos();
            amps[row * n + 1] = 1.0 + 0.8 * (theta * 2.0).cos();
        }
        let mat = pac_matrix(&phases, &amps, t, n, 18);
        // Diagonal entries should show coupling
        assert!(mat[0] > 0.01, "MI[0,0]={} too low", mat[0]);
        assert!(mat[3] > 0.01, "MI[1,1]={} too low", mat[3]);
    }
}
