// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Recurrence analysis (Eckmann et al. 1987, Marwan et al. 2007)

//! Recurrence matrix construction and Recurrence Quantification Analysis (RQA).
//!
//! References:
//! - Eckmann, Kamphorst & Ruelle 1987, Europhys. Lett. 4:973-977.
//! - Zbilut & Webber 1992, Phys. Lett. A 171:199-203.
//! - Marwan et al. 2007, Phys. Reports 438:237-329.

use std::collections::HashMap;

/// RQA result: all standard measures from Marwan et al. 2007 Table 1.
#[derive(Debug, Clone)]
pub struct RqaResult {
    pub recurrence_rate: f64,
    pub determinism: f64,
    pub avg_diagonal: f64,
    pub max_diagonal: usize,
    pub entropy_diagonal: f64,
    pub laminarity: f64,
    pub trapping_time: f64,
    pub max_vertical: usize,
}

/// Compute binary recurrence matrix R_ij = Θ(ε − ‖x_i − x_j‖).
///
/// # Arguments
/// * `trajectory` - row-major (T × d) flattened trajectory
/// * `t` - number of time points
/// * `d` - embedding dimension
/// * `epsilon` - recurrence threshold
/// * `angular` - if true, use chord distance on S¹ per dimension
///
/// # Returns
/// Flattened (T × T) boolean matrix as Vec<u8> (1 = recurrent, 0 = not).
///
/// # Errors
/// Returns error if trajectory length ≠ T × d.
pub fn recurrence_matrix(
    trajectory: &[f64],
    t: usize,
    d: usize,
    epsilon: f64,
    angular: bool,
) -> Result<Vec<u8>, String> {
    if trajectory.len() != t * d {
        return Err(format!(
            "trajectory length {} != T*d={}*{}={}",
            trajectory.len(),
            t,
            d,
            t * d
        ));
    }
    let eps_sq = epsilon * epsilon;
    let mut result = vec![0u8; t * t];

    for i in 0..t {
        // Diagonal is always recurrent (self-recurrence)
        result[i * t + i] = 1;
        for j in (i + 1)..t {
            let dist_sq = if angular {
                // Chord distance on S¹: d(θ,φ) = 2|sin((θ-φ)/2)| per dimension
                // Squared sum: Σ_k 4 sin²((x_ik - x_jk)/2)
                let mut s = 0.0_f64;
                for k in 0..d {
                    let diff = trajectory[i * d + k] - trajectory[j * d + k];
                    let half_sin = (diff * 0.5).sin();
                    s += 4.0 * half_sin * half_sin;
                }
                s
            } else {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let diff = trajectory[i * d + k] - trajectory[j * d + k];
                    s += diff * diff;
                }
                s
            };
            if dist_sq <= eps_sq {
                result[i * t + j] = 1;
                result[j * t + i] = 1;
            }
        }
    }
    Ok(result)
}

/// Cross-recurrence matrix CR_ij = Θ(ε − ‖x_i − y_j‖).
///
/// # Errors
/// Returns error if trajectory lengths ≠ T × d.
pub fn cross_recurrence_matrix(
    traj_a: &[f64],
    traj_b: &[f64],
    t: usize,
    d: usize,
    epsilon: f64,
    angular: bool,
) -> Result<Vec<u8>, String> {
    if traj_a.len() != t * d || traj_b.len() != t * d {
        return Err(format!(
            "trajectory lengths ({}, {}) != T*d={}",
            traj_a.len(),
            traj_b.len(),
            t * d
        ));
    }
    let eps_sq = epsilon * epsilon;
    let mut result = vec![0u8; t * t];

    for i in 0..t {
        for j in 0..t {
            let dist_sq = if angular {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let diff = traj_a[i * d + k] - traj_b[j * d + k];
                    let half_sin = (diff * 0.5).sin();
                    s += 4.0 * half_sin * half_sin;
                }
                s
            } else {
                let mut s = 0.0_f64;
                for k in 0..d {
                    let diff = traj_a[i * d + k] - traj_b[j * d + k];
                    s += diff * diff;
                }
                s
            };
            if dist_sq <= eps_sq {
                result[i * t + j] = 1;
            }
        }
    }
    Ok(result)
}

/// Full RQA on a flattened recurrence matrix.
///
/// Implements all standard measures from Marwan et al. 2007 Table 1:
/// RR, DET, L (avg diagonal), L_max, ENTR (diagonal entropy),
/// LAM, TT (trapping time), V_max.
///
/// The main diagonal is excluded from RR computation (self-recurrence
/// is trivial). Diagonal lines of length ≥ l_min contribute to DET.
/// Vertical lines of length ≥ v_min contribute to LAM.
///
/// # Errors
/// Returns error if recurrence length ≠ T².
#[allow(clippy::too_many_arguments)]
pub fn rqa(
    recurrence: &[u8],
    t: usize,
    l_min: usize,
    v_min: usize,
    exclude_main_diagonal: bool,
) -> Result<RqaResult, String> {
    if recurrence.len() != t * t {
        return Err(format!(
            "recurrence length {} != T²={}",
            recurrence.len(),
            t * t
        ));
    }
    if t == 0 {
        return Ok(RqaResult {
            recurrence_rate: 0.0,
            determinism: 0.0,
            avg_diagonal: 0.0,
            max_diagonal: 0,
            entropy_diagonal: 0.0,
            laminarity: 0.0,
            trapping_time: 0.0,
            max_vertical: 0,
        });
    }

    // Work on a mutable copy so we can zero the main diagonal
    let mut r_work = recurrence.to_vec();
    if exclude_main_diagonal {
        for i in 0..t {
            r_work[i * t + i] = 0;
        }
    }

    // Count recurrence points
    let n_recurrent: usize = r_work.iter().filter(|&&v| v != 0).count();
    let off_diag_total = if exclude_main_diagonal {
        t * t - t
    } else {
        t * t
    };
    let rr = if off_diag_total > 0 {
        n_recurrent as f64 / off_diag_total as f64
    } else {
        0.0
    };

    // Diagonal lines (scan all diagonals k = 1..T-1 and k = -(T-1)..=-1)
    // For symmetric R, only k > 0 needed but we scan all for cross-recurrence
    let diag_lengths = extract_diagonal_lines(&r_work, t, l_min, exclude_main_diagonal);
    let (det, avg_diag, max_diag, ent_diag) = compute_line_stats(&diag_lengths, n_recurrent);

    // Vertical lines (scan columns)
    let vert_lengths = extract_vertical_lines(&r_work, t, v_min);
    let (lam, trapping_time, max_vert, _) = compute_line_stats_vert(&vert_lengths, n_recurrent);

    Ok(RqaResult {
        recurrence_rate: rr,
        determinism: det,
        avg_diagonal: avg_diag,
        max_diagonal: max_diag,
        entropy_diagonal: ent_diag,
        laminarity: lam,
        trapping_time,
        max_vertical: max_vert,
    })
}

/// Extract diagonal line lengths from recurrence matrix.
///
/// Scans all diagonals (both above and below main diagonal).
/// Lines on the main diagonal (k=0) are excluded when exclude_main is true.
fn extract_diagonal_lines(r: &[u8], t: usize, l_min: usize, exclude_main: bool) -> Vec<usize> {
    let mut lengths = Vec::new();

    // Diagonals above main (k > 0) and below (k < 0)
    // For k > 0: positions (i, i+k) for i = 0..T-k
    // For k < 0: positions (i-k, i) for i = 0..T+k
    let start_k: isize = if exclude_main { 1 } else { 0 };
    for k in start_k..(t as isize) {
        let len = t - k as usize;
        let mut count: usize = 0;
        for idx in 0..len {
            let i = idx;
            let j = idx + k as usize;
            if r[i * t + j] != 0 {
                count += 1;
            } else {
                if count >= l_min {
                    lengths.push(count);
                }
                count = 0;
            }
        }
        if count >= l_min {
            lengths.push(count);
        }
    }
    // Diagonals below main (k < 0), equivalent to transposed upper diagonals
    for k in 1..(t as isize) {
        let len = t - k as usize;
        let mut count: usize = 0;
        for idx in 0..len {
            let i = idx + k as usize;
            let j = idx;
            if r[i * t + j] != 0 {
                count += 1;
            } else {
                if count >= l_min {
                    lengths.push(count);
                }
                count = 0;
            }
        }
        if count >= l_min {
            lengths.push(count);
        }
    }
    lengths
}

/// Extract vertical line lengths from recurrence matrix (scan columns).
fn extract_vertical_lines(r: &[u8], t: usize, v_min: usize) -> Vec<usize> {
    let mut lengths = Vec::new();
    for col in 0..t {
        let mut count: usize = 0;
        for row in 0..t {
            if r[row * t + col] != 0 {
                count += 1;
            } else {
                if count >= v_min {
                    lengths.push(count);
                }
                count = 0;
            }
        }
        if count >= v_min {
            lengths.push(count);
        }
    }
    lengths
}

/// Compute diagonal line statistics: DET, avg length, max length, entropy.
fn compute_line_stats(lengths: &[usize], n_recurrent: usize) -> (f64, f64, usize, f64) {
    if lengths.is_empty() {
        return (0.0, 0.0, 0, 0.0);
    }
    let total_points: usize = lengths.iter().sum();
    let det = if n_recurrent > 0 {
        total_points as f64 / n_recurrent as f64
    } else {
        0.0
    };
    let avg = total_points as f64 / lengths.len() as f64;
    let max_l = *lengths.iter().max().unwrap_or(&0);

    // Shannon entropy of length distribution (Marwan et al. 2007 Eq. 7)
    let mut hist: HashMap<usize, usize> = HashMap::new();
    for &l in lengths {
        *hist.entry(l).or_insert(0) += 1;
    }
    let n_total = lengths.len() as f64;
    let entropy: f64 = hist
        .values()
        .map(|&c| {
            let p = c as f64 / n_total;
            -p * p.ln()
        })
        .sum();

    (det, avg, max_l, entropy)
}

/// Compute vertical line statistics: LAM, trapping time, max vertical.
fn compute_line_stats_vert(lengths: &[usize], n_recurrent: usize) -> (f64, f64, usize, f64) {
    if lengths.is_empty() {
        return (0.0, 0.0, 0, 0.0);
    }
    let total_points: usize = lengths.iter().sum();
    let lam = if n_recurrent > 0 {
        total_points as f64 / n_recurrent as f64
    } else {
        0.0
    };
    let trapping = total_points as f64 / lengths.len() as f64;
    let max_v = *lengths.iter().max().unwrap_or(&0);
    (lam, trapping, max_v, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_recurrence_matrix_identity() {
        // Identical points → full recurrence
        let traj = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // T=3, d=2, all same
        let r = recurrence_matrix(&traj, 3, 2, 0.01, false).unwrap();
        assert_eq!(r.len(), 9);
        assert!(r.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_recurrence_matrix_distant() {
        // Far-apart points → no recurrence except self
        let traj = vec![0.0, 100.0, 200.0]; // T=3, d=1
        let r = recurrence_matrix(&traj, 3, 1, 1.0, false).unwrap();
        // Only diagonal should be 1
        assert_eq!(r[0 * 3 + 0], 1);
        assert_eq!(r[1 * 3 + 1], 1);
        assert_eq!(r[2 * 3 + 2], 1);
        assert_eq!(r[0 * 3 + 1], 0);
        assert_eq!(r[0 * 3 + 2], 0);
        assert_eq!(r[1 * 3 + 2], 0);
    }

    #[test]
    fn test_recurrence_matrix_symmetric() {
        let traj = vec![0.0, 0.5, 1.5, 0.1, 2.0]; // T=5, d=1
        let r = recurrence_matrix(&traj, 5, 1, 0.6, false).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    r[i * 5 + j],
                    r[j * 5 + i],
                    "R not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_angular_metric() {
        // Phases 0 and 2π should be close on the circle
        let traj = vec![0.0, TAU - 0.01]; // T=2, d=1
        let r = recurrence_matrix(&traj, 2, 1, 0.1, true).unwrap();
        assert_eq!(
            r[0 * 2 + 1],
            1,
            "0 and 2π-0.01 should be recurrent with angular metric"
        );
    }

    #[test]
    fn test_rqa_periodic_trajectory() {
        // Periodic: 0, 1, 0, 1, 0, 1 (period 2)
        let traj = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // T=6, d=1
        let r = recurrence_matrix(&traj, 6, 1, 0.1, false).unwrap();
        let result = rqa(&r, 6, 2, 2, true).unwrap();
        // Periodic trajectory should have high determinism
        assert!(
            result.determinism > 0.5,
            "periodic should be deterministic: DET={}",
            result.determinism
        );
        assert!(result.max_diagonal >= 2);
    }

    #[test]
    fn test_rqa_random_trajectory() {
        // Random (pseudo) trajectory: low determinism expected
        let mut traj = Vec::with_capacity(100);
        let mut x = 0.1_f64;
        for _ in 0..100 {
            x = (x * 3.9 * (1.0 - x)).max(0.0).min(1.0); // logistic map, chaotic
            traj.push(x);
        }
        let r = recurrence_matrix(&traj, 100, 1, 0.1, false).unwrap();
        let result = rqa(&r, 100, 2, 2, true).unwrap();
        assert!(result.recurrence_rate >= 0.0 && result.recurrence_rate <= 1.0);
        assert!(result.determinism >= 0.0 && result.determinism <= 1.0);
        assert!(result.laminarity >= 0.0 && result.laminarity <= 1.0);
    }

    #[test]
    fn test_rqa_empty() {
        let result = rqa(&[], 0, 2, 2, true).unwrap();
        assert_eq!(result.recurrence_rate, 0.0);
        assert_eq!(result.max_diagonal, 0);
    }

    #[test]
    fn test_cross_recurrence_matrix() {
        // Identical trajectories → same as recurrence
        let traj = vec![0.0, 1.0, 0.0, 1.0]; // T=4, d=1
        let cr = cross_recurrence_matrix(&traj, &traj, 4, 1, 0.1, false).unwrap();
        let r = recurrence_matrix(&traj, 4, 1, 0.1, false).unwrap();
        assert_eq!(cr, r);
    }

    #[test]
    fn test_rqa_laminar_trajectory() {
        // Laminar: stays near 0 for a while, then near 1
        let mut traj = vec![0.0; 20];
        traj.extend(vec![1.0; 20]);
        traj.extend(vec![0.0; 20]);
        let r = recurrence_matrix(&traj, 60, 1, 0.1, false).unwrap();
        let result = rqa(&r, 60, 2, 2, true).unwrap();
        // Laminar trajectory should have high laminarity
        assert!(
            result.laminarity > 0.3,
            "laminar should have high LAM: {}",
            result.laminarity
        );
        assert!(result.trapping_time > 2.0);
    }

    #[test]
    fn test_diagonal_entropy_nonneg() {
        let traj = vec![0.0, 0.5, 0.1, 0.6, 0.2, 0.7, 0.3, 0.8]; // T=8, d=1
        let r = recurrence_matrix(&traj, 8, 1, 0.5, false).unwrap();
        let result = rqa(&r, 8, 2, 2, true).unwrap();
        assert!(
            result.entropy_diagonal >= 0.0,
            "entropy should be non-negative: {}",
            result.entropy_diagonal
        );
    }

    #[test]
    fn test_mismatched_length_error() {
        let result = recurrence_matrix(&[0.0, 1.0, 2.0], 2, 2, 1.0, false);
        assert!(result.is_err());
    }
}
