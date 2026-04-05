// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Sleep stage classifier from Kuramoto order parameter

//! AASM sleep staging mapped to Kuramoto order parameter R.
//!
//! N3 (slow-wave): R >= 0.70 — highly synchronised cortical oscillations.
//! N2 (spindle):   R in [0.40, 0.70) — moderate synchrony with K-complex bursts.
//! N1 (drowsy):    R in [0.30, 0.40) — partial desynchronisation.
//! REM:            R in [0.20, 0.35) with functional_desync flag.
//! Wake:           R < 0.30, no functional_desync.

const N3_THRESHOLD: f64 = 0.70;
const N2_THRESHOLD: f64 = 0.40;
const N1_THRESHOLD: f64 = 0.30;
const REM_THRESHOLD: f64 = 0.20;

/// Ultradian NREM–REM cycle period (Rechtschaffen & Kales 1968).
const ULTRADIAN_PERIOD_S: f64 = 90.0 * 60.0; // 90 minutes in seconds

/// Classify sleep stage from Kuramoto order parameter R.
///
/// Returns one of: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
#[must_use]
pub fn classify_sleep_stage(r: f64, functional_desync: bool) -> u8 {
    if r >= N3_THRESHOLD {
        return 3; // N3
    }
    if r >= N2_THRESHOLD {
        return 2; // N2
    }
    if r >= N1_THRESHOLD {
        if functional_desync {
            return 4; // REM
        }
        return 1; // N1
    }
    if functional_desync && r >= REM_THRESHOLD {
        return 4; // REM
    }
    0 // Wake
}

/// Estimate position within the ~90-minute ultradian sleep cycle.
///
/// Returns phase in [0, 1) where 0 = cycle start (N3 onset).
#[must_use]
pub fn ultradian_phase(timestamps: &[f64], stages: &[u8]) -> f64 {
    if timestamps.is_empty() || stages.is_empty() {
        return 0.0;
    }
    let n = timestamps.len().min(stages.len());

    // Find last N3 epoch (stage code 3)
    let mut last_n3_idx: Option<usize> = None;
    for i in (0..n).rev() {
        if stages[i] == 3 {
            last_n3_idx = Some(i);
            break;
        }
    }

    match last_n3_idx {
        None => 0.0,
        Some(idx) => {
            let elapsed = timestamps[n - 1] - timestamps[idx];
            (elapsed % ULTRADIAN_PERIOD_S) / ULTRADIAN_PERIOD_S
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n3_classification() {
        assert_eq!(classify_sleep_stage(0.8, false), 3);
        assert_eq!(classify_sleep_stage(0.7, false), 3);
    }

    #[test]
    fn test_n2_classification() {
        assert_eq!(classify_sleep_stage(0.5, false), 2);
        assert_eq!(classify_sleep_stage(0.4, false), 2);
    }

    #[test]
    fn test_n1_classification() {
        assert_eq!(classify_sleep_stage(0.35, false), 1);
        assert_eq!(classify_sleep_stage(0.3, false), 1);
    }

    #[test]
    fn test_rem_with_desync() {
        assert_eq!(classify_sleep_stage(0.35, true), 4); // N1 range + desync → REM
        assert_eq!(classify_sleep_stage(0.25, true), 4); // below N1, above REM + desync → REM
    }

    #[test]
    fn test_wake() {
        assert_eq!(classify_sleep_stage(0.1, false), 0);
        assert_eq!(classify_sleep_stage(0.15, false), 0);
    }

    #[test]
    fn test_wake_low_desync() {
        // Below REM threshold even with desync → Wake
        assert_eq!(classify_sleep_stage(0.1, true), 0);
    }

    #[test]
    fn test_ultradian_basic() {
        let ts = vec![0.0, 60.0, 120.0, 180.0, 240.0];
        let stages = vec![3, 3, 2, 1, 0]; // N3 at t=0,60; last N3 at idx=1
        let phase = ultradian_phase(&ts, &stages);
        // elapsed = 240 - 60 = 180s, period = 5400s
        let expected = 180.0 / ULTRADIAN_PERIOD_S;
        assert!(
            (phase - expected).abs() < 1e-10,
            "phase={phase}, expected={expected}"
        );
    }

    #[test]
    fn test_ultradian_no_n3() {
        let ts = vec![0.0, 60.0, 120.0];
        let stages = vec![0, 1, 2]; // no N3
        assert_eq!(ultradian_phase(&ts, &stages), 0.0);
    }

    #[test]
    fn test_ultradian_empty() {
        assert_eq!(ultradian_phase(&[], &[]), 0.0);
    }

    #[test]
    fn test_ultradian_wrapping() {
        // Elapsed > 1 period → wraps
        let ts = vec![0.0, 6000.0]; // 6000s > 5400s period
        let stages = vec![3, 0];
        let phase = ultradian_phase(&ts, &stages);
        let expected = (6000.0 % ULTRADIAN_PERIOD_S) / ULTRADIAN_PERIOD_S;
        assert!((phase - expected).abs() < 1e-10);
    }
}
