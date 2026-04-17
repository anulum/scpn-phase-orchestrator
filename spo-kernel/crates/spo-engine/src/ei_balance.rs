// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Excitatory/Inhibitory balance

//! E/I balance computation for Kuramoto coupling matrices.
//!
//! Kuroki & Mizuseki 2025, Neural Computation — E/I balance is the
//! critical parameter for synchronisation, not K or D.

/// E/I balance result.
pub struct EIBalanceResult {
    pub ratio: f64,
    pub excitatory_strength: f64,
    pub inhibitory_strength: f64,
    pub is_balanced: bool,
}

/// Compute E/I balance from coupling matrix and layer typing.
///
/// ratio > 1: excitation-dominated (hypersynchrony risk)
/// ratio < 1: inhibition-dominated (desynchronisation risk)
/// ratio ≈ 1: balanced (optimal for metastability)
///
/// # Arguments
/// * `knm_flat` — (N×N) row-major coupling matrix
/// * `n` — number of oscillators
/// * `excitatory_indices` — indices of excitatory oscillators
/// * `inhibitory_indices` — indices of inhibitory oscillators
#[must_use]
pub fn compute_ei_balance(
    knm_flat: &[f64],
    n: usize,
    excitatory_indices: &[usize],
    inhibitory_indices: &[usize],
) -> EIBalanceResult {
    let e_strength = if excitatory_indices.is_empty() {
        0.0
    } else {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &i in excitatory_indices {
            if i < n {
                for j in 0..n {
                    sum += knm_flat[i * n + j];
                    count += 1;
                }
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    };

    let i_strength = if inhibitory_indices.is_empty() {
        0.0
    } else {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &i in inhibitory_indices {
            if i < n {
                for j in 0..n {
                    sum += knm_flat[i * n + j];
                    count += 1;
                }
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    };

    let ratio = if i_strength.abs() < 1e-15 {
        if e_strength > 0.0 {
            f64::INFINITY
        } else {
            1.0
        }
    } else {
        e_strength / i_strength
    };

    EIBalanceResult {
        ratio,
        excitatory_strength: e_strength,
        inhibitory_strength: i_strength,
        is_balanced: (0.8..=1.2).contains(&ratio),
    }
}

/// Scale inhibitory coupling to achieve target E/I ratio.
///
/// Returns modified knm (flat) with inhibitory rows scaled.
#[must_use]
pub fn adjust_ei_ratio(
    knm_flat: &[f64],
    n: usize,
    excitatory_indices: &[usize],
    inhibitory_indices: &[usize],
    target_ratio: f64,
) -> Vec<f64> {
    let balance = compute_ei_balance(knm_flat, n, excitatory_indices, inhibitory_indices);

    if balance.inhibitory_strength.abs() < 1e-15 || balance.excitatory_strength.abs() < 1e-15 {
        return knm_flat.to_vec();
    }
    if (balance.ratio - target_ratio).abs() < 1e-10 {
        return knm_flat.to_vec();
    }

    let scale = balance.ratio / target_ratio;
    let mut result = knm_flat.to_vec();
    for &idx in inhibitory_indices {
        if idx < n {
            for j in 0..n {
                result[idx * n + j] *= scale;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_ratio() {
        // Equal E and I strength → ratio = 1.0
        let n = 4;
        let knm = vec![1.0; n * n];
        let e_idx = vec![0, 1];
        let i_idx = vec![2, 3];
        let result = compute_ei_balance(&knm, n, &e_idx, &i_idx);
        assert!((result.ratio - 1.0).abs() < 1e-10);
        assert!(result.is_balanced);
    }

    #[test]
    fn test_excitation_dominated() {
        let n = 4;
        let mut knm = vec![1.0; n * n];
        // Make excitatory rows stronger
        for j in 0..n {
            knm[0 * n + j] = 3.0;
            knm[1 * n + j] = 3.0;
        }
        let result = compute_ei_balance(&knm, n, &[0, 1], &[2, 3]);
        assert!(
            result.ratio > 1.0,
            "should be excitation-dominated, got {}",
            result.ratio
        );
        assert!(!result.is_balanced);
    }

    #[test]
    fn test_no_inhibitory() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let result = compute_ei_balance(&knm, n, &[0, 1, 2], &[]);
        assert_eq!(result.inhibitory_strength, 0.0);
        assert_eq!(result.ratio, f64::INFINITY);
    }

    #[test]
    fn test_no_excitatory() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let result = compute_ei_balance(&knm, n, &[], &[0, 1, 2]);
        assert_eq!(result.excitatory_strength, 0.0);
        // e_strength = 0, i_strength > 0 → ratio = 0/i = 0
        assert_eq!(result.ratio, 0.0);
    }

    #[test]
    fn test_adjust_ratio_to_target() {
        let n = 4;
        let mut knm = vec![1.0; n * n];
        // E rows twice as strong
        for j in 0..n {
            knm[0 * n + j] = 2.0;
            knm[1 * n + j] = 2.0;
        }
        let adjusted = adjust_ei_ratio(&knm, n, &[0, 1], &[2, 3], 1.0);
        let new_balance = compute_ei_balance(&adjusted, n, &[0, 1], &[2, 3]);
        assert!(
            (new_balance.ratio - 1.0).abs() < 0.1,
            "should be near 1.0 after adjustment, got {}",
            new_balance.ratio
        );
    }

    #[test]
    fn test_adjust_no_change_when_balanced() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let adjusted = adjust_ei_ratio(&knm, n, &[0], &[1, 2], 1.0);
        assert_eq!(adjusted, knm);
    }

    #[test]
    fn test_empty_coupling() {
        let result = compute_ei_balance(&[], 0, &[], &[]);
        assert_eq!(result.ratio, 1.0);
        assert_eq!(result.excitatory_strength, 0.0);
    }

    #[test]
    fn test_out_of_bounds_indices_ignored() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let result = compute_ei_balance(&knm, n, &[0, 100], &[1]);
        // Index 100 is out of bounds, should be skipped
        assert!(result.excitatory_strength > 0.0);
    }
}
