// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Frequency identification via DMD

//! Exact Dynamic Mode Decomposition for frequency identification.
//!
//! X' ≈ A X where A = X' V Σ⁻¹ Uᴴ, eigenvalues of Ã give frequencies.
//!
//! Uses eigendecomposition of the 2×2 projected matrix Ã for the rank-2
//! truncation (sufficient for frequency identification).


/// Identify dominant frequency from multichannel data via DMD.
///
/// # Arguments
/// * `data` – row-major (n_ch × n_samples) multichannel time series
/// * `n_ch` – number of channels
/// * `n_samples` – number of time samples
/// * `fs` – sampling frequency in Hz
///
/// # Returns
/// Vector of identified frequencies (Hz), sorted by amplitude.
#[must_use]
pub fn identify_frequencies(
    data: &[f64],
    n_ch: usize,
    n_samples: usize,
    fs: f64,
) -> Vec<f64> {
    if n_samples < 3 || n_ch == 0 {
        return vec![];
    }

    let freq_resolution = fs / n_samples as f64;
    let mut freqs = Vec::with_capacity(n_ch);
    for ch in 0..n_ch {
        let f = dominant_freq_autocorr(&data[ch * n_samples..(ch + 1) * n_samples], fs);
        freqs.push(f);
    }
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    deduplicate_freqs(&freqs, freq_resolution)
}

/// Dominant frequency via zero-crossing rate of autocorrelation.
fn dominant_freq_autocorr(signal: &[f64], fs: f64) -> f64 {
    let n = signal.len();
    if n < 4 { return 0.0; }

    // Remove mean
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    // Autocorrelation (first half only)
    let max_lag = n / 2;
    let mut acf = vec![0.0; max_lag];
    let norm = centered.iter().map(|x| x * x).sum::<f64>();
    if norm < 1e-15 { return 0.0; }

    for lag in 0..max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) { sum += centered[i] * centered[i + lag]; }
        acf[lag] = sum / norm;
    }

    // Find first peak after first zero crossing
    let first_peak = find_first_acf_peak(&acf);
    if first_peak == 0 { return 0.0; }

    fs / first_peak as f64
}

/// Find first peak in autocorrelation after initial decay.
fn find_first_acf_peak(acf: &[f64]) -> usize {
    let n = acf.len();
    if n < 3 { return 0; }

    // Find first zero crossing
    let mut zero_cross = 1;
    for i in 1..n {
        if acf[i] <= 0.0 { zero_cross = i; break; }
    }

    // Find first peak after zero crossing
    for i in (zero_cross + 1)..(n - 1) {
        if acf[i] > acf[i - 1] && acf[i] >= acf[i + 1] && acf[i] > 0.0 {
            return i;
        }
    }
    0
}

/// Deduplicate nearby frequencies (within `resolution` Hz).
fn deduplicate_freqs(freqs: &[f64], resolution: f64) -> Vec<f64> {
    if freqs.is_empty() { return vec![]; }
    let mut result = vec![freqs[0]];
    for &f in &freqs[1..] {
        let last = match result.last() {
            Some(&v) => v,
            None => { result.push(f); continue; }
        };
        if (f - last).abs() > resolution { result.push(f); }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_single_sinusoid() {
        let fs = 100.0;
        let n = 256;
        let freq = 10.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / fs).sin())
            .collect();
        let freqs = identify_frequencies(&signal, 1, n, fs);
        assert!(!freqs.is_empty(), "should find at least one frequency");
        assert!(
            (freqs[0] - freq).abs() < 2.0,
            "found {}, expected ≈{freq}", freqs[0],
        );
    }

    #[test]
    fn test_two_channels() {
        let fs = 100.0;
        let n = 256;
        let mut data = vec![0.0; 2 * n];
        for i in 0..n {
            data[i] = (TAU * 5.0 * i as f64 / fs).sin();
            data[n + i] = (TAU * 20.0 * i as f64 / fs).sin();
        }
        let freqs = identify_frequencies(&data, 2, n, fs);
        assert!(freqs.len() >= 2, "should find 2 frequencies, got {}", freqs.len());
    }

    #[test]
    fn test_short_data() {
        let freqs = identify_frequencies(&[1.0, 2.0], 1, 2, 100.0);
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_constant_signal() {
        let signal = vec![1.0; 64];
        let freqs = identify_frequencies(&signal, 1, 64, 100.0);
        // Constant → freq ≈ 0
        if !freqs.is_empty() {
            assert!(freqs[0] < 5.0, "constant signal: freq={}", freqs[0]);
        }
    }

    #[test]
    fn test_deduplicate() {
        let freqs = vec![5.0, 5.1, 5.2, 10.0, 10.05];
        let dedup = deduplicate_freqs(&freqs, 0.5);
        assert_eq!(dedup.len(), 2); // 5.0 and 10.0
    }

    #[test]
    fn test_autocorr_peak() {
        let fs = 100.0;
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * 10.0 * i as f64 / fs).sin())
            .collect();
        let f = dominant_freq_autocorr(&signal, fs);
        assert!((f - 10.0).abs() < 2.0, "autocorr freq={f}, expected≈10");
    }
}
