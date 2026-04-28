// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Amplitude envelope solver

//! Sliding-window RMS envelope and modulation depth.

/// Extract sliding-window RMS envelope from a 1-D amplitude time series.
///
/// Returns a vector of the same length as `amplitudes`, padded at the front
/// with the first valid RMS value.
#[must_use]
pub fn extract_envelope(amplitudes: &[f64], window: usize) -> Vec<f64> {
    let t = amplitudes.len();
    if t == 0 || window == 0 {
        return amplitudes.to_vec();
    }
    if amplitudes.iter().any(|v| !v.is_finite()) {
        return vec![0.0; t];
    }

    // Cumulative sum of squares
    let mut cs = vec![0.0; t + 1];
    for i in 0..t {
        cs[i + 1] = cs[i] + amplitudes[i] * amplitudes[i];
    }

    // Sliding window RMS
    let n_valid = if t >= window { t - window + 1 } else { 0 };
    let mut rms = Vec::with_capacity(n_valid);
    for i in 0..n_valid {
        let sum_sq = cs[i + window] - cs[i];
        rms.push((sum_sq / window as f64).sqrt());
    }

    if rms.is_empty() {
        // Window larger than data: return zeros
        return vec![0.0; t];
    }

    // Pad front with first valid value
    let mut result = vec![rms[0]; window - 1];
    result.extend_from_slice(&rms);
    // Truncate to original length if needed
    result.truncate(t);
    result
}

/// Modulation depth: (max - min) / (max + min), in [0, 1].
///
/// Returns 0.0 for empty or constant envelopes.
#[must_use]
pub fn envelope_modulation_depth(envelope: &[f64]) -> f64 {
    if envelope.is_empty() {
        return 0.0;
    }
    let mut vmax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    for &v in envelope {
        if v > vmax {
            vmax = v;
        }
        if v < vmin {
            vmin = v;
        }
    }
    let denom = vmax + vmin;
    if denom <= 0.0 {
        return 0.0;
    }
    (vmax - vmin) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_signal() {
        let amps = vec![2.0; 20];
        let env = extract_envelope(&amps, 5);
        assert_eq!(env.len(), 20);
        for v in &env {
            assert!((*v - 2.0).abs() < 1e-10, "v={v}");
        }
    }

    #[test]
    fn test_modulation_depth_constant() {
        let env = vec![1.0; 10];
        assert!((envelope_modulation_depth(&env)).abs() < 1e-10);
    }

    #[test]
    fn test_modulation_depth_range() {
        let env = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let md = envelope_modulation_depth(&env);
        // (5-1)/(5+1) = 4/6 = 0.667
        assert!((md - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_signal() {
        let env = extract_envelope(&[], 5);
        assert!(env.is_empty());
        assert_eq!(envelope_modulation_depth(&[]), 0.0);
    }

    #[test]
    fn test_nonfinite_signal_returns_zero_envelope() {
        let env = extract_envelope(&[1.0, f64::NAN, 2.0], 2);
        assert_eq!(env, vec![0.0; 3]);
    }

    #[test]
    fn test_window_one() {
        let amps = vec![1.0, 4.0, 9.0];
        let env = extract_envelope(&amps, 1);
        assert_eq!(env.len(), 3);
        // RMS with window=1 is just abs(value)
        assert!((env[0] - 1.0).abs() < 1e-10);
        assert!((env[1] - 4.0).abs() < 1e-10);
        assert!((env[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_output_length_matches_input() {
        for t in [5, 10, 50, 100] {
            let amps: Vec<f64> = (0..t).map(|i| (i as f64 * 0.1).sin().abs()).collect();
            let env = extract_envelope(&amps, 3);
            assert_eq!(env.len(), t, "length mismatch for T={t}");
        }
    }

    #[test]
    fn test_window_equals_length() {
        let amps = vec![1.0, 2.0, 3.0];
        let env = extract_envelope(&amps, 3);
        assert_eq!(env.len(), 3);
        // Only one valid RMS value, padded
        let expected_rms = ((1.0 + 4.0 + 9.0) / 3.0_f64).sqrt();
        for v in &env {
            assert!((*v - expected_rms).abs() < 1e-10);
        }
    }

    #[test]
    fn test_window_larger_than_length() {
        let amps = vec![1.0, 2.0];
        let env = extract_envelope(&amps, 10);
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn test_sinusoidal_envelope() {
        // Sinusoid should have roughly constant RMS envelope
        let n = 200;
        let amps: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let env = extract_envelope(&amps, 63); // ~1 period
                                               // After the padding, interior values should be roughly constant
        let interior = &env[63..n];
        let mean: f64 = interior.iter().sum::<f64>() / interior.len() as f64;
        for v in interior {
            // RMS of sine ≈ 1/√2 ≈ 0.707, but depends on exact window alignment
            assert!((v - mean).abs() < 0.15, "v={v}, mean={mean}");
        }
    }

    #[test]
    fn test_modulation_depth_zeros() {
        let env = vec![0.0; 10];
        assert_eq!(envelope_modulation_depth(&env), 0.0);
    }
}
