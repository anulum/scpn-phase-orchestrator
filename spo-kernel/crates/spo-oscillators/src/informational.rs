// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Informational oscillator

//!
//! Converts event timestamps to phase via inter-event intervals.

use std::f64::consts::TAU;

/// Phase from event timing: cumulative integral of instantaneous frequency.
///
/// Returns (theta, omega_median, quality) where:
/// - theta is the final cumulative phase mod 2π
/// - omega_median is the median instantaneous angular velocity
/// - quality is 1/(1+CV) where CV is the coefficient of variation of intervals
#[must_use]
pub fn event_phase(timestamps: &[f64]) -> (f64, f64, f64) {
    if timestamps.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let intervals: Vec<f64> = timestamps
        .windows(2)
        .filter_map(|w| {
            let dt = w[1] - w[0];
            if dt > 0.0 {
                Some(dt)
            } else {
                None
            }
        })
        .collect();

    if intervals.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let inst_freq: Vec<f64> = intervals.iter().map(|&dt| 1.0 / dt).collect();

    // Median instantaneous frequency → angular velocity
    let mut sorted_freq = inst_freq.clone();
    sorted_freq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let omega_median = if sorted_freq.len() % 2 == 0 {
        let mid = sorted_freq.len() / 2;
        (sorted_freq[mid - 1] + sorted_freq[mid]) / 2.0
    } else {
        sorted_freq[sorted_freq.len() / 2]
    } * TAU;

    // Cumulative phase via median frequency × total elapsed time
    let total_time: f64 =
        timestamps.last().copied().unwrap_or(0.0) - timestamps.first().copied().unwrap_or(0.0);
    let omega_hz = omega_median / TAU;
    let cumulative = TAU * omega_hz * total_time;
    let theta = cumulative.rem_euclid(TAU);

    // Quality: inverse CV of intervals
    let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let variance = intervals
        .iter()
        .map(|&dt| (dt - mean_interval).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;
    let cv = if mean_interval > 0.0 {
        variance.sqrt() / mean_interval
    } else {
        1.0
    };
    let quality = (1.0 / (1.0 + cv)).clamp(0.0, 1.0);

    (theta, omega_median, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_timestamp() {
        let (theta, omega, quality) = event_phase(&[1.0]);
        assert_eq!(theta, 0.0);
        assert_eq!(omega, 0.0);
        assert_eq!(quality, 0.0);
    }

    #[test]
    fn empty_timestamps() {
        let (theta, omega, quality) = event_phase(&[]);
        assert_eq!(theta, 0.0);
        assert_eq!(omega, 0.0);
        assert_eq!(quality, 0.0);
    }

    #[test]
    fn regular_events_high_quality() {
        let timestamps: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let (_, _, quality) = event_phase(&timestamps);
        assert!(
            quality > 0.9,
            "regular events should have high quality, got {quality}"
        );
    }

    #[test]
    fn irregular_events_lower_quality() {
        let timestamps = vec![0.0, 0.1, 0.5, 0.55, 1.5, 1.51, 3.0];
        let (_, _, quality) = event_phase(&timestamps);
        assert!(
            quality < 0.8,
            "irregular events should have lower quality, got {quality}"
        );
    }

    #[test]
    fn theta_in_range() {
        let timestamps: Vec<f64> = (0..100).map(|i| i as f64 * 0.05).collect();
        let (theta, _, _) = event_phase(&timestamps);
        assert!((0.0..TAU).contains(&theta), "theta={theta} out of [0, 2π)");
    }

    #[test]
    fn omega_positive_for_increasing_timestamps() {
        let timestamps: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (_, omega, _) = event_phase(&timestamps);
        assert!(omega > 0.0, "omega should be positive for regular events");
    }

    #[test]
    fn nan_in_timestamps_no_panic() {
        let timestamps = vec![0.0, 1.0, f64::NAN, 3.0, 4.0];
        let (theta, omega, quality) = event_phase(&timestamps);
        assert!(theta.is_finite() || theta == 0.0);
        assert!(omega.is_finite() || omega == 0.0);
        assert!(quality.is_finite() || quality == 0.0);
    }
}
