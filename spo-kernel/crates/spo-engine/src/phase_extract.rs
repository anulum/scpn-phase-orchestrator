// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase extraction via Hilbert transform

//! Extract instantaneous phase, amplitude, and frequency from a real-valued
//! time series using the analytic signal (Hilbert transform via FFT).
#![allow(clippy::needless_range_loop)] // indices used as mathematical variables in DFT

use std::f64::consts::{PI, TAU};

/// Extract phases, amplitudes, instantaneous frequencies, and dominant frequency
/// from a 1-D signal via Hilbert transform.
///
/// # Arguments
/// * `signal` – real-valued time series
/// * `fs` – sampling frequency in Hz
///
/// # Returns
/// `(phases, amplitudes, inst_freq, dominant_freq)` where:
/// - `phases` – instantaneous phase in [0, 2π), length N
/// - `amplitudes` – instantaneous amplitude (envelope), length N
/// - `inst_freq` – instantaneous frequency in Hz, length N
/// - `dominant_freq` – peak frequency from FFT in Hz
#[must_use]
pub fn extract_phases(signal: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = signal.len();
    if n < 4 {
        return (vec![], vec![], vec![], 0.0);
    }

    let fft = dft(signal);
    let analytic = compute_analytic_signal(&fft, n);
    let (phases, amplitudes, unwrapped) = extract_phase_amplitude(&analytic, n);
    let inst_freq = instantaneous_frequency(&unwrapped, fs, n);
    let dominant_freq = find_dominant_freq(&fft, n, fs);

    (phases, amplitudes, inst_freq, dominant_freq)
}

/// Build analytic signal via Hilbert transform in frequency domain.
fn compute_analytic_signal(fft: &[(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    let mut a_fft = fft.to_vec();
    let half = n / 2;
    for entry in a_fft.iter_mut().take(half).skip(1) {
        entry.0 *= 2.0;
        entry.1 *= 2.0;
    }
    for entry in a_fft.iter_mut().take(n).skip(half + 1) {
        *entry = (0.0, 0.0);
    }
    idft(&a_fft)
}

/// Extract phase [0,2π), amplitude, and raw unwrapped angle from analytic signal.
fn extract_phase_amplitude(analytic: &[(f64, f64)], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut phases = vec![0.0; n];
    let mut amps = vec![0.0; n];
    let mut unwrapped = vec![0.0; n];
    for i in 0..n {
        let (re, im) = analytic[i];
        phases[i] = im.atan2(re).rem_euclid(TAU);
        amps[i] = (re * re + im * im).sqrt();
        unwrapped[i] = im.atan2(re);
    }
    // Phase unwrapping
    for i in 1..n {
        let mut d = unwrapped[i] - unwrapped[i - 1];
        while d > PI {
            d -= TAU;
        }
        while d < -PI {
            d += TAU;
        }
        unwrapped[i] = unwrapped[i - 1] + d;
    }
    (phases, amps, unwrapped)
}

/// Compute instantaneous frequency from unwrapped phase.
fn instantaneous_frequency(unwrapped: &[f64], fs: f64, n: usize) -> Vec<f64> {
    let mut inst = vec![0.0; n];
    for i in 1..n {
        inst[i] = (unwrapped[i] - unwrapped[i - 1]) * fs / TAU;
    }
    inst
}

/// Find dominant frequency from FFT magnitudes.
fn find_dominant_freq(fft: &[(f64, f64)], n: usize, fs: f64) -> f64 {
    let mut max_mag = 0.0;
    let mut max_idx = 1;
    for i in 1..(n / 2 + 1) {
        let mag = (fft[i].0 * fft[i].0 + fft[i].1 * fft[i].1).sqrt();
        if mag > max_mag {
            max_mag = mag;
            max_idx = i;
        }
    }
    max_idx as f64 * fs / n as f64
}

/// Discrete Fourier Transform (naive O(N²) for correctness).
fn dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let mut result = vec![(0.0, 0.0); n];
    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let angle = -TAU * k as f64 * t as f64 / n as f64;
            re += x[t] * angle.cos();
            im += x[t] * angle.sin();
        }
        result[k] = (re, im);
    }
    result
}

/// Inverse DFT.
fn idft(x: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = x.len();
    let inv_n = 1.0 / n as f64;
    let mut result = vec![(0.0, 0.0); n];
    for t in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..n {
            let angle = TAU * k as f64 * t as f64 / n as f64;
            let (xr, xi) = x[k];
            re += xr * angle.cos() - xi * angle.sin();
            im += xr * angle.sin() + xi * angle.cos();
        }
        result[t] = (re * inv_n, im * inv_n);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoid_dominant_freq() {
        let fs = 100.0;
        let n = 256;
        let freq = 10.0;
        let signal: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64 / fs).sin()).collect();
        let (_, _, _, dom) = extract_phases(&signal, fs);
        assert!(
            (dom - freq).abs() < fs / n as f64 * 2.0,
            "dominant={dom}, expected≈{freq}"
        );
    }

    #[test]
    fn test_phases_in_range() {
        let fs = 100.0;
        let signal: Vec<f64> = (0..128)
            .map(|i| (TAU * 5.0 * i as f64 / fs).sin())
            .collect();
        let (phases, _, _, _) = extract_phases(&signal, fs);
        for p in &phases {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} out of range");
        }
    }

    #[test]
    fn test_amplitudes_nonneg() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin() * 3.0).collect();
        let (_, amps, _, _) = extract_phases(&signal, 100.0);
        for a in &amps {
            assert!(*a >= 0.0, "amplitude {a} should be >= 0");
        }
    }

    #[test]
    fn test_short_signal() {
        let (p, _, _, _) = extract_phases(&[1.0, 2.0], 100.0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_output_lengths() {
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();
        let (phases, amps, inst, _) = extract_phases(&signal, 100.0);
        assert_eq!(phases.len(), n);
        assert_eq!(amps.len(), n);
        assert_eq!(inst.len(), n);
    }

    #[test]
    fn test_constant_signal_phases_valid() {
        let signal = vec![1.0; 32];
        let (phases, amps, _, _) = extract_phases(&signal, 100.0);
        assert_eq!(phases.len(), 32);
        // Amplitudes should be roughly constant
        for a in &amps {
            assert!(*a >= 0.0);
        }
    }

    #[test]
    fn test_dft_idft_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let fft = dft(&x);
        let back = idft(&fft);
        for (i, &val) in x.iter().enumerate() {
            assert!((back[i].0 - val).abs() < 1e-10, "roundtrip failed at {i}");
            assert!(back[i].1.abs() < 1e-10);
        }
    }
}
