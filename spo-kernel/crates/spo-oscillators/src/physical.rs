// SCPN Phase Orchestrator — Physical Phase Extraction (post-Hilbert)
//!
//! Fused extraction of theta, omega, amplitude, quality from a pre-computed
//! analytic signal. Replaces five separate NumPy passes with two Rust loops.

use std::f64::consts::TAU;

/// Extract phase, frequency, amplitude, and quality from an analytic signal.
///
/// `real` and `imag` are the real and imaginary parts of the analytic signal
/// (from scipy.signal.hilbert). For Hilbert transforms, `real == original signal`.
///
/// Returns `(theta, omega, amplitude, quality)`:
/// - theta: instantaneous phase of the last sample, in [0, TAU)
/// - omega: median instantaneous angular frequency (rad/s)
/// - amplitude: mean envelope magnitude
/// - quality: always 1.0 for Hilbert analytic signals (real == signal, so noise = 0)
pub fn extract_from_analytic(real: &[f64], imag: &[f64], sample_rate: f64) -> (f64, f64, f64, f64) {
    let n = real.len();
    if n == 0 || imag.len() != n {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Pass 1: inst_phase + amplitude accumulator
    let mut inst_phase = vec![0.0_f64; n];
    let mut amp_sum = 0.0_f64;

    for i in 0..n {
        inst_phase[i] = imag[i].atan2(real[i]);
        amp_sum += (real[i] * real[i] + imag[i] * imag[i]).sqrt();
    }

    let amplitude = amp_sum / n as f64;
    let theta = inst_phase[n - 1].rem_euclid(TAU);

    // For Hilbert analytic signals, real == original signal, so noise = 0 and quality = 1.0.
    let quality = 1.0;

    // Pass 2: unwrap + gradient → inst_freq → median → omega
    // Unwrap in-place
    for i in 1..n {
        let mut d = inst_phase[i] - inst_phase[i - 1];
        d = ((d + std::f64::consts::PI) % TAU) - std::f64::consts::PI;
        if d < -std::f64::consts::PI {
            d += TAU;
        }
        inst_phase[i] = inst_phase[i - 1] + d;
    }

    // Gradient (central differences, matching numpy.gradient)
    let omega = if n == 1 {
        0.0
    } else {
        let mut inst_freq = vec![0.0_f64; n];
        // numpy.gradient edge handling: forward/backward at boundaries, central in interior
        inst_freq[0] = (inst_phase[1] - inst_phase[0]) * sample_rate / TAU;
        inst_freq[n - 1] = (inst_phase[n - 1] - inst_phase[n - 2]) * sample_rate / TAU;
        for i in 1..(n - 1) {
            inst_freq[i] = (inst_phase[i + 1] - inst_phase[i - 1]) * 0.5 * sample_rate / TAU;
        }

        // O(n) median via select_nth_unstable
        let mid = inst_freq.len() / 2;
        if inst_freq.len() % 2 == 1 {
            inst_freq.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            inst_freq[mid] * TAU
        } else {
            inst_freq.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let upper = inst_freq[mid];
            inst_freq[..mid].select_nth_unstable_by(mid - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lower = inst_freq[mid - 1];
            (lower + upper) / 2.0 * TAU
        }
    };

    (theta, omega, amplitude, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sinusoid(freq_hz: f64, sample_rate: f64, duration: f64) -> (Vec<f64>, Vec<f64>) {
        let n = (sample_rate * duration) as usize;
        let mut real = vec![0.0; n];
        let mut imag = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / sample_rate;
            let phase = TAU * freq_hz * t;
            real[i] = phase.cos();
            imag[i] = phase.sin();
        }
        (real, imag)
    }

    #[test]
    fn extract_clean_sinusoid() {
        let (real, imag) = make_sinusoid(10.0, 1000.0, 0.5);
        let (theta, omega, _amp, quality) = extract_from_analytic(&real, &imag, 1000.0);
        assert!((0.0..TAU).contains(&theta), "theta={theta}");
        let expected_omega = TAU * 10.0;
        assert!(
            (omega - expected_omega).abs() / expected_omega < 0.05,
            "omega={omega}, expected ~{expected_omega}"
        );
        assert!(quality > 0.5, "quality={quality}");
    }

    #[test]
    fn extract_preserves_frequency() {
        for &freq in &[5.0, 20.0, 50.0] {
            let (real, imag) = make_sinusoid(freq, 1000.0, 1.0);
            let (_, omega, _, _) = extract_from_analytic(&real, &imag, 1000.0);
            let expected = TAU * freq;
            assert!(
                (omega - expected).abs() / expected < 0.05,
                "freq={freq}: omega={omega}, expected={expected}"
            );
        }
    }

    #[test]
    fn extract_zero_length() {
        let (theta, omega, amp, quality) = extract_from_analytic(&[], &[], 1000.0);
        assert_eq!(theta, 0.0);
        assert_eq!(omega, 0.0);
        assert_eq!(amp, 0.0);
        assert_eq!(quality, 0.0);
    }

    #[test]
    fn extract_mismatched_lengths() {
        let (theta, omega, amp, quality) = extract_from_analytic(&[1.0, 2.0], &[1.0], 1000.0);
        assert_eq!(theta, 0.0);
        assert_eq!(omega, 0.0);
        assert_eq!(amp, 0.0);
        assert_eq!(quality, 0.0);
    }

    #[test]
    fn nan_in_signal_no_panic() {
        let real = vec![1.0, f64::NAN, 0.5, -0.5];
        let imag = vec![0.0, 0.5, f64::NAN, 0.5];
        let (_theta, _omega, _amp, _quality) = extract_from_analytic(&real, &imag, 1000.0);
        // Must not panic; output may be NaN but that is acceptable
    }
}
