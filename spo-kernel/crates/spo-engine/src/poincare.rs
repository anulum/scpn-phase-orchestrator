// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Poincaré section analysis

//! Poincaré section crossings and return time analysis.
//!
//! Detects hyperplane crossings, extracts crossing coordinates via
//! linear interpolation, and computes return time statistics.
//!
//! References:
//!   Poincaré 1899, "Les méthodes nouvelles de la mécanique céleste".
//!   Strogatz 2015, "Nonlinear Dynamics and Chaos", Ch. 8.

use rayon::prelude::*;
use std::f64::consts::PI;

/// Direction of crossing detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingDirection {
    Positive,
    Negative,
    Both,
}

/// Result of Poincaré section analysis.
pub struct PoincareResult {
    /// Flattened (M × d) crossing coordinates.
    pub crossings: Vec<f64>,
    /// (M,) fractional time indices of crossings.
    pub crossing_times: Vec<f64>,
    /// Number of crossings found.
    pub n_crossings: usize,
}

/// Detect crossings of a hyperplane n·x = offset.
///
/// # Arguments
/// * `traj_flat` — row-major (T × d) trajectory
/// * `t` — number of time steps
/// * `d` — state-space dimension
/// * `normal` — (d,) normal vector (will be normalised internally)
/// * `offset` — plane offset
/// * `direction` — crossing direction filter
///
/// # Errors
/// Returns `Err` if dimensions are inconsistent.
pub fn poincare_section(
    traj_flat: &[f64],
    t: usize,
    d: usize,
    normal: &[f64],
    offset: f64,
    direction: CrossingDirection,
) -> Result<PoincareResult, String> {
    if traj_flat.len() != t * d {
        return Err(format!(
            "trajectory length {} != T*d = {}",
            traj_flat.len(),
            t * d
        ));
    }
    if normal.len() != d {
        return Err(format!("normal length {} != d = {d}", normal.len()));
    }

    let norm = normal.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Err("normal vector has zero length".into());
    }
    let n_hat: Vec<f64> = normal.iter().map(|&v| v / norm).collect();

    // Parallel distance computation
    let signed_dist: Vec<f64> = (0..t)
        .into_par_iter()
        .map(|i| {
            let mut dot = 0.0;
            let row = &traj_flat[i * d..(i + 1) * d];
            for k in 0..d {
                dot += row[k] * n_hat[k];
            }
            dot - offset
        })
        .collect();

    // Parallel crossing detection and interpolation
    let results: Vec<(Vec<f64>, f64)> = (0..t - 1)
        .into_par_iter()
        .filter_map(|i| {
            let d0 = signed_dist[i];
            let d1 = signed_dist[i + 1];
            let is_crossing = match direction {
                CrossingDirection::Positive => d0 < 0.0 && d1 >= 0.0,
                CrossingDirection::Negative => d0 > 0.0 && d1 <= 0.0,
                CrossingDirection::Both => d0 * d1 < 0.0,
            };
            if is_crossing {
                let alpha = if (d1 - d0).abs() > 1e-15 {
                    -d0 / (d1 - d0)
                } else {
                    0.5
                };
                let mut cross = Vec::with_capacity(d);
                let p0 = &traj_flat[i * d..(i + 1) * d];
                let p1 = &traj_flat[(i + 1) * d..(i + 2) * d];
                for k in 0..d {
                    cross.push(p0[k] + alpha * (p1[k] - p0[k]));
                }
                Some((cross, i as f64 + alpha))
            } else {
                None
            }
        })
        .collect();

    let n_crossings = results.len();
    let mut crossings = Vec::with_capacity(n_crossings * d);
    let mut crossing_times = Vec::with_capacity(n_crossings);
    for (c, ct) in results {
        crossings.extend(c);
        crossing_times.push(ct);
    }

    Ok(PoincareResult {
        crossings,
        crossing_times,
        n_crossings,
    })
}

/// Poincaré section for phase oscillator trajectories.
///
/// Detects when oscillator `oscillator_idx` crosses `section_phase` (mod 2π).
///
/// # Arguments
/// * `phases_flat` — row-major (T × N) phase matrix
/// * `t` — number of time steps
/// * `n` — number of oscillators
/// * `oscillator_idx` — which oscillator defines the section
/// * `section_phase` — phase value for section crossing
///
/// # Errors
/// Returns `Err` if dimensions are inconsistent.
pub fn phase_poincare(
    phases_flat: &[f64],
    t: usize,
    n: usize,
    oscillator_idx: usize,
    section_phase: f64,
) -> Result<PoincareResult, String> {
    if phases_flat.len() != t * n {
        return Err(format!(
            "phases length {} != T*N = {}",
            phases_flat.len(),
            t * n
        ));
    }
    if oscillator_idx >= n {
        return Err(format!("oscillator_idx {oscillator_idx} >= N = {n}"));
    }

    let two_pi = 2.0 * PI;

    // Sequential unwrap (needed for target oscillator)
    let mut unwrapped = Vec::with_capacity(t);
    let mut last_raw = phases_flat[oscillator_idx];
    let mut last_unwrapped = last_raw;
    unwrapped.push(last_unwrapped);
    for i in 1..t {
        let curr_raw = phases_flat[i * n + oscillator_idx];
        let mut diff = curr_raw - last_raw;
        diff = diff - two_pi * (diff / two_pi).round();
        last_unwrapped += diff;
        unwrapped.push(last_unwrapped);
        last_raw = curr_raw;
    }

    // Parallel shifted phases and crossing detection
    let shifted: Vec<f64> = unwrapped
        .par_iter()
        .map(|&v| ((v - section_phase) % two_pi + two_pi) % two_pi)
        .collect();

    let results: Vec<(Vec<f64>, f64)> = (0..t - 1)
        .into_par_iter()
        .filter_map(|i| {
            if shifted[i] > PI && shifted[i + 1] < PI {
                let alpha = (shifted[i] / (shifted[i] - shifted[i + 1] + two_pi)).clamp(0.0, 1.0);
                let mut cross = Vec::with_capacity(n);
                let p0 = &phases_flat[i * n..(i + 1) * n];
                let p1 = &phases_flat[(i + 1) * n..(i + 2) * n];
                for k in 0..n {
                    cross.push(p0[k] + alpha * (p1[k] - p0[k]));
                }
                Some((cross, i as f64 + alpha))
            } else {
                None
            }
        })
        .collect();

    let n_crossings = results.len();
    let mut crossings = Vec::with_capacity(n_crossings * n);
    let mut crossing_times = Vec::with_capacity(n_crossings);
    for (c, ct) in results {
        crossings.extend(c);
        crossing_times.push(ct);
    }

    Ok(PoincareResult {
        crossings,
        crossing_times,
        n_crossings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_poincare_section_circle() {
        // 3 full revolutions, start at θ=0.1 to avoid exact-zero boundary
        // → 3 positive crossings of y=0 (interior, not at endpoints)
        let n_points = 3000;
        let mut traj = Vec::with_capacity(n_points * 2);
        for i in 0..n_points {
            let theta = 0.1 + 3.0 * TAU * i as f64 / n_points as f64;
            traj.push(theta.cos());
            traj.push(theta.sin());
        }
        let normal = vec![0.0, 1.0];
        let result = poincare_section(
            &traj,
            n_points,
            2,
            &normal,
            0.0,
            CrossingDirection::Positive,
        )
        .unwrap();
        assert_eq!(
            result.n_crossings, 3,
            "expected 3 positive crossings, got {}",
            result.n_crossings
        );
    }

    #[test]
    fn test_poincare_section_both() {
        // 3 full revolutions with offset start → 6 total crossings
        let n_points = 3000;
        let mut traj = Vec::with_capacity(n_points * 2);
        for i in 0..n_points {
            let theta = 0.1 + 3.0 * TAU * i as f64 / n_points as f64;
            traj.push(theta.cos());
            traj.push(theta.sin());
        }
        let normal = vec![0.0, 1.0];
        let result =
            poincare_section(&traj, n_points, 2, &normal, 0.0, CrossingDirection::Both).unwrap();
        assert_eq!(
            result.n_crossings, 6,
            "expected 6 both-direction crossings, got {}",
            result.n_crossings
        );
    }

    #[test]
    fn test_poincare_section_no_crossings() {
        // Constant trajectory
        let traj = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 3 points in 2D
        let normal = vec![0.0, 1.0];
        let result =
            poincare_section(&traj, 3, 2, &normal, 0.0, CrossingDirection::Positive).unwrap();
        assert_eq!(result.n_crossings, 0);
    }

    #[test]
    fn test_poincare_return_times_periodic() {
        // 4 full revolutions → crossings with equal return times
        let n_points = 4001;
        let mut traj = Vec::with_capacity(n_points * 2);
        for i in 0..n_points {
            let theta = 4.0 * TAU * i as f64 / (n_points - 1) as f64;
            traj.push(theta.cos());
            traj.push(theta.sin());
        }
        let normal = vec![0.0, 1.0];
        let result = poincare_section(
            &traj,
            n_points,
            2,
            &normal,
            0.0,
            CrossingDirection::Positive,
        )
        .unwrap();
        assert!(
            result.n_crossings >= 3,
            "expected ≥3 crossings, got {}",
            result.n_crossings
        );
        let rt: Vec<f64> = result
            .crossing_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        for (i, pair) in rt.windows(2).enumerate() {
            let diff = (pair[0] - pair[1]).abs();
            assert!(diff < 2.0, "return times {i} differ by {diff}");
        }
    }

    #[test]
    fn test_poincare_interpolation_accuracy() {
        // Simple linear crossing: (0, -1) → (0, 1), should cross at (0, 0)
        let traj = vec![0.0, -1.0, 0.0, 1.0];
        let normal = vec![0.0, 1.0];
        let result =
            poincare_section(&traj, 2, 2, &normal, 0.0, CrossingDirection::Positive).unwrap();
        assert_eq!(result.n_crossings, 1);
        assert!((result.crossings[0] - 0.0).abs() < 1e-12);
        assert!((result.crossings[1] - 0.0).abs() < 1e-12); // y should be ~0
        assert!((result.crossing_times[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_poincare_dimension_mismatch() {
        let traj = vec![1.0, 2.0, 3.0]; // 3 points in 1D
        let normal = vec![1.0, 0.0]; // 2D normal
        assert!(poincare_section(&traj, 3, 1, &normal, 0.0, CrossingDirection::Positive).is_err());
    }

    #[test]
    fn test_phase_poincare_single_oscillator() {
        // Oscillator with linearly increasing phase: crosses 0 once per 2π
        let t = 1000;
        let n = 1;
        let mut phases = Vec::with_capacity(t);
        for i in 0..t {
            phases.push(3.0 * TAU * i as f64 / t as f64); // 3 revolutions
        }
        let result = phase_poincare(&phases, t, n, 0, 0.0).unwrap();
        assert!(
            result.n_crossings >= 2,
            "expected ≥2 crossings, got {}",
            result.n_crossings
        );
    }

    #[test]
    fn test_phase_poincare_multiple_oscillators() {
        // 2 oscillators, different frequencies
        let t = 2000;
        let n = 2;
        let mut phases = Vec::with_capacity(t * n);
        for i in 0..t {
            let theta = TAU * i as f64 / t as f64;
            phases.push(5.0 * theta); // osc 0: 5 revolutions
            phases.push(3.0 * theta); // osc 1: 3 revolutions
        }
        let r0 = phase_poincare(&phases, t, n, 0, 0.0).unwrap();
        let r1 = phase_poincare(&phases, t, n, 1, 0.0).unwrap();
        assert!(
            r0.n_crossings > r1.n_crossings,
            "osc 0 ({} freq) should have more crossings than osc 1 ({})",
            r0.n_crossings,
            r1.n_crossings
        );
    }

    #[test]
    fn test_phase_poincare_section_phase() {
        // Section at π instead of 0 — should give same number of crossings
        let t = 2000;
        let n = 1;
        let mut phases = Vec::with_capacity(t);
        for i in 0..t {
            phases.push(5.0 * TAU * i as f64 / t as f64);
        }
        let r0 = phase_poincare(&phases, t, n, 0, 0.0).unwrap();
        let r_pi = phase_poincare(&phases, t, n, 0, PI).unwrap();
        // Both should have crossings; counts may differ by ±1 due to boundary
        let diff = (r0.n_crossings as i64 - r_pi.n_crossings as i64).unsigned_abs();
        assert!(
            diff <= 1,
            "crossing counts differ by {diff}: {} vs {}",
            r0.n_crossings,
            r_pi.n_crossings
        );
    }

    #[test]
    fn test_phase_poincare_empty() {
        // Constant phase — no crossings
        let phases = vec![1.0; 100];
        let result = phase_poincare(&phases, 100, 1, 0, 0.0).unwrap();
        assert_eq!(result.n_crossings, 0);
    }

    #[test]
    fn test_phase_poincare_invalid_idx() {
        let phases = vec![0.0; 10];
        assert!(phase_poincare(&phases, 10, 1, 5, 0.0).is_err());
    }
}
