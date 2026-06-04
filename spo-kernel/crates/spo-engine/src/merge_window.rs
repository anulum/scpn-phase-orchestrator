// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — PHA-C merge-window monitor

//! Phase-and-space merge-window monitor for PHA-C moving-frame dynamics.
//!
//! A merge lock is accepted only when wrapped phase dispersion and axial spatial
//! dispersion both remain inside their tolerances for a configured number of
//! consecutive samples.

use spo_types::{SpoError, SpoResult};
use std::f64::consts::{PI, TAU};

const DEFAULT_PHASE_TOL_RAD: f64 = 0.01;
const DEFAULT_SPATIAL_TOL_M: f64 = 0.002;

/// Resolved PHA-C tolerance profile.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeWindowToleranceProfile {
    pub name: String,
    pub phase_tol_rad: f64,
    pub spatial_tol_m: f64,
    pub multiplier: f64,
    pub baseline_phase_tol_rad: f64,
    pub baseline_spatial_tol_m: f64,
}

/// Audit-ready merge-window state for one sampled instant.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeWindowReport {
    pub t: f64,
    pub phase_dispersion_rad: f64,
    pub spatial_dispersion_m: f64,
    pub phase_locked: bool,
    pub spatial_locked: bool,
    pub lock_achieved: bool,
    pub consecutive_lock_samples: usize,
}

/// Evaluate one PHA-C merge-window sample.
///
/// # Errors
/// Returns `InvalidDimension` when vectors are empty or mismatched. Returns
/// `InvalidConfig` when tolerances, timestamps, references, or phase/position
/// samples are non-finite or invalid.
pub fn evaluate_merge_window(
    phases: &[f64],
    positions: &[f64],
    t: f64,
    reference_phase: f64,
    reference_point: f64,
    phase_tol_rad: f64,
    spatial_tol_m: f64,
    required_consecutive_samples: usize,
    prior_consecutive_lock_samples: usize,
) -> SpoResult<MergeWindowReport> {
    if phases.is_empty() {
        return Err(SpoError::InvalidDimension(
            "phases must contain at least one sample".into(),
        ));
    }
    if phases.len() != positions.len() {
        return Err(SpoError::InvalidDimension(format!(
            "positions must match phases length, got {} vs {}",
            positions.len(),
            phases.len()
        )));
    }
    if required_consecutive_samples == 0 {
        return Err(SpoError::InvalidConfig(
            "required_consecutive_samples must be at least 1".into(),
        ));
    }
    if !t.is_finite()
        || !reference_phase.is_finite()
        || !reference_point.is_finite()
        || !phase_tol_rad.is_finite()
        || !spatial_tol_m.is_finite()
    {
        return Err(SpoError::InvalidConfig(
            "merge-window scalar controls must be finite".into(),
        ));
    }
    if phase_tol_rad < 0.0 || spatial_tol_m < 0.0 {
        return Err(SpoError::InvalidConfig(
            "merge-window tolerances must be non-negative".into(),
        ));
    }

    let mut phase_dispersion = 0.0_f64;
    let mut spatial_dispersion = 0.0_f64;
    for (&phase, &position) in phases.iter().zip(positions) {
        if !phase.is_finite() || !position.is_finite() {
            return Err(SpoError::InvalidConfig(
                "merge-window samples must be finite".into(),
            ));
        }
        let phase_delta = (phase - reference_phase + PI).rem_euclid(TAU) - PI;
        phase_dispersion = phase_dispersion.max(phase_delta.abs());
        spatial_dispersion = spatial_dispersion.max((position - reference_point).abs());
    }

    let phase_locked = phase_dispersion <= phase_tol_rad;
    let spatial_locked = spatial_dispersion <= spatial_tol_m;
    let consecutive_lock_samples = if phase_locked && spatial_locked {
        prior_consecutive_lock_samples.saturating_add(1)
    } else {
        0
    };
    Ok(MergeWindowReport {
        t,
        phase_dispersion_rad: phase_dispersion,
        spatial_dispersion_m: spatial_dispersion,
        phase_locked,
        spatial_locked,
        lock_achieved: consecutive_lock_samples >= required_consecutive_samples,
        consecutive_lock_samples,
    })
}

/// Resolve a named PHA-C merge-window tolerance profile.
///
/// # Errors
/// Returns `InvalidConfig` for unknown profile names or invalid baselines.
pub fn resolve_merge_window_tolerance_profile(
    profile: &str,
    phase_baseline_rad: f64,
    spatial_baseline_m: f64,
) -> SpoResult<MergeWindowToleranceProfile> {
    if !phase_baseline_rad.is_finite() || !spatial_baseline_m.is_finite() {
        return Err(SpoError::InvalidConfig(
            "merge-window tolerance baselines must be finite".into(),
        ));
    }
    if phase_baseline_rad < 0.0 || spatial_baseline_m < 0.0 {
        return Err(SpoError::InvalidConfig(
            "merge-window tolerance baselines must be non-negative".into(),
        ));
    }
    let normalised = profile.trim().to_ascii_lowercase();
    let multiplier = match normalised.as_str() {
        "baseline_1x" => 1.0,
        "buffer_3x" => 3.0,
        "review_5x" => 5.0,
        _ => {
            return Err(SpoError::InvalidConfig(
                "unknown merge-window tolerance profile".into(),
            ));
        }
    };
    Ok(MergeWindowToleranceProfile {
        name: normalised,
        phase_tol_rad: phase_baseline_rad * multiplier,
        spatial_tol_m: spatial_baseline_m * multiplier,
        multiplier,
        baseline_phase_tol_rad: phase_baseline_rad,
        baseline_spatial_tol_m: spatial_baseline_m,
    })
}

/// Resolve a named tolerance profile against the default PHA-C baselines.
///
/// # Errors
/// Returns `InvalidConfig` for unknown profile names.
pub fn resolve_default_merge_window_tolerance_profile(
    profile: &str,
) -> SpoResult<MergeWindowToleranceProfile> {
    resolve_merge_window_tolerance_profile(profile, DEFAULT_PHASE_TOL_RAD, DEFAULT_SPATIAL_TOL_M)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_joint_phase_and_spatial_lock() {
        let report = evaluate_merge_window(
            &[0.0, 0.004, -0.005],
            &[0.0, 0.001, -0.0015],
            12.5,
            0.0,
            0.0,
            0.01,
            0.002,
            1,
            0,
        )
        .expect("valid merge-window report");
        assert!(report.phase_locked);
        assert!(report.spatial_locked);
        assert!(report.lock_achieved);
    }

    #[test]
    fn wraps_phase_near_two_pi() {
        let report = evaluate_merge_window(
            &[TAU - 0.003, 0.004],
            &[0.0, 0.0],
            0.0,
            0.0,
            0.0,
            0.005,
            0.001,
            1,
            0,
        )
        .expect("valid wrapped merge-window report");
        assert!(report.phase_locked);
        assert!(report.phase_dispersion_rad <= 0.004 + 1.0e-12);
    }

    #[test]
    fn resets_consecutive_counter_on_spatial_failure() {
        let report = evaluate_merge_window(
            &[0.0, 0.004],
            &[0.0, 0.003],
            0.0,
            0.0,
            0.0,
            0.01,
            0.002,
            3,
            2,
        )
        .expect("valid failing merge-window report");
        assert!(report.phase_locked);
        assert!(!report.spatial_locked);
        assert_eq!(report.consecutive_lock_samples, 0);
        assert!(!report.lock_achieved);
    }

    #[test]
    fn resolves_named_tolerance_profiles() {
        let baseline = resolve_default_merge_window_tolerance_profile("baseline_1x")
            .expect("valid baseline profile");
        let buffer = resolve_default_merge_window_tolerance_profile("buffer_3x")
            .expect("valid buffer profile");
        let review = resolve_default_merge_window_tolerance_profile("review_5x")
            .expect("valid review profile");
        assert_eq!(baseline.phase_tol_rad, 0.01);
        assert_eq!(baseline.spatial_tol_m, 0.002);
        assert_eq!(buffer.phase_tol_rad, 0.03);
        assert_eq!(buffer.spatial_tol_m, 0.006);
        assert_eq!(review.phase_tol_rad, 0.05);
        assert_eq!(review.spatial_tol_m, 0.01);
    }

    #[test]
    fn profile_tolerance_accepts_buffered_sample() {
        let profile = resolve_default_merge_window_tolerance_profile("buffer_3x")
            .expect("valid buffer profile");
        let strict = evaluate_merge_window(
            &[0.0, 0.024],
            &[0.0, 0.0045],
            0.0,
            0.0,
            0.0,
            DEFAULT_PHASE_TOL_RAD,
            DEFAULT_SPATIAL_TOL_M,
            1,
            0,
        )
        .expect("valid strict report");
        let profiled = evaluate_merge_window(
            &[0.0, 0.024],
            &[0.0, 0.0045],
            0.0,
            0.0,
            0.0,
            profile.phase_tol_rad,
            profile.spatial_tol_m,
            1,
            0,
        )
        .expect("valid profiled report");
        assert!(!strict.lock_achieved);
        assert!(profiled.lock_achieved);
    }

    #[test]
    fn rejects_invalid_boundaries() {
        assert!(matches!(
            evaluate_merge_window(&[], &[], 0.0, 0.0, 0.0, 0.01, 0.002, 1, 0),
            Err(SpoError::InvalidDimension(_))
        ));
        assert!(matches!(
            evaluate_merge_window(&[0.0], &[0.0, 1.0], 0.0, 0.0, 0.0, 0.01, 0.002, 1, 0,),
            Err(SpoError::InvalidDimension(_))
        ));
        assert!(matches!(
            evaluate_merge_window(&[f64::NAN], &[0.0], 0.0, 0.0, 0.0, 0.01, 0.002, 1, 0,),
            Err(SpoError::InvalidConfig(_))
        ));
        assert!(matches!(
            resolve_default_merge_window_tolerance_profile("unknown"),
            Err(SpoError::InvalidConfig(_))
        ));
    }
}
