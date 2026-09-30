// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Physical oscillator

//!
//! Fused extraction of theta, omega, amplitude, quality from a pre-computed
//! analytic signal, with scaled envelope statistics.

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
/// - quality: clipped `1 - CV(envelope)`, or zero for mean envelope below 1e-15
///
/// Envelope magnitudes use hypot and scaled mean/variance arithmetic so finite,
/// representable large envelopes do not overflow intermediate squares or sums.
#[must_use]
#[allow(clippy::needless_range_loop)] // Pass 2 unwrap mutates inst_phase[i] from inst_phase[i-1]
pub fn extract_from_analytic(real: &[f64], imag: &[f64], sample_rate: f64) -> (f64, f64, f64, f64) {
    let n = real.len();
    if n == 0 || imag.len() != n {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut inst_phase = vec![0.0_f64; n];
    let mut envelope = vec![0.0_f64; n];
    let mut envelope_scale = 0.0_f64;
    for ((ip, env), (&r, &im)) in inst_phase
        .iter_mut()
        .zip(envelope.iter_mut())
        .zip(real.iter().zip(imag.iter()))
    {
        *ip = im.atan2(r);
        *env = r.hypot(im);
        envelope_scale = envelope_scale.max(*env);
    }

    let theta = inst_phase[n - 1].rem_euclid(TAU);
    let (amplitude, quality) = if envelope_scale == 0.0 {
        (0.0, 0.0)
    } else {
        let scaled_mean = envelope.iter().map(|env| env / envelope_scale).sum::<f64>() / n as f64;
        let amplitude = envelope_scale * scaled_mean;
        let quality = if amplitude < 1e-15 {
            0.0
        } else {
            let scaled_variance = envelope
                .iter()
                .map(|env| (env / envelope_scale - scaled_mean).powi(2))
                .sum::<f64>()
                / n as f64;
            (1.0 - scaled_variance.sqrt() / scaled_mean).clamp(0.0, 1.0)
        };
        (amplitude, quality)
    };

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
            inst_freq.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            inst_freq[mid] * TAU
        } else {
            inst_freq.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let upper = inst_freq[mid];
            inst_freq[..mid].select_nth_unstable_by(mid - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
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
        assert!(quality > 0.9, "quality={quality}");
    }

    #[test]
    fn noisy_signal_lower_quality() {
        let n = 500;
        let sr = 1000.0;
        let mut real = vec![0.0; n];
        let mut imag = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / sr;
            let phase = TAU * 10.0 * t;
            // Amplitude-modulate to create envelope variation
            let am = 1.0 + 0.8 * (TAU * 2.0 * t).sin();
            real[i] = am * phase.cos();
            imag[i] = am * phase.sin();
        }
        let (_, _, _, quality) = extract_from_analytic(&real, &imag, sr);
        assert!(quality < 0.9, "AM signal quality={quality} should be < 0.9");
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
    fn large_envelopes_preserve_mean_and_variation() {
        for scale in [1.0, 1e150, 1e160, 1e200, 1e290] {
            for modulation in [0.0, 0.6] {
                let n = 128;
                let mut real = Vec::with_capacity(n);
                let mut imag = Vec::with_capacity(n);
                for i in 0..n {
                    let phase = TAU * i as f64 / n as f64;
                    let amplitude = scale * (1.0 + modulation * phase.cos());
                    real.push(amplitude * (8.0 * phase).cos());
                    imag.push(amplitude * (8.0 * phase).sin());
                }
                let (_, omega, amplitude, quality) = extract_from_analytic(&real, &imag, 128.0);
                assert!((omega / (TAU * 8.0) - 1.0).abs() < 1e-12);
                assert!((amplitude / scale - 1.0).abs() < 1e-12);
                assert!((quality - (1.0 - modulation / 2.0_f64.sqrt())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn representable_envelope_mean_does_not_overflow_its_sum() {
        let (_, _, amplitude, quality) = extract_from_analytic(&[1e308; 4], &[0.0; 4], 1.0);
        assert_eq!(amplitude, 1e308);
        assert_eq!(quality, 1.0);
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
