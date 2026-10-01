// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Excitatory/Inhibitory balance

//! Signed arithmetic E/I summaries and copy-preserving row adjustment.

use spo_types::{SpoError, SpoResult};

/// E/I balance result.
///
/// `excitatory_strength` / `inhibitory_strength` aggregate the mean coupling
/// from each source group over all targets. The four `*_to_*` block means
/// resolve directed source-to-target interaction strengths, including diagonal
/// entries. The balance flag is a numerical interval, not a physiological
/// validation or an empirical regime classification.
pub struct EIBalanceResult {
    pub ratio: f64,
    pub excitatory_strength: f64,
    pub inhibitory_strength: f64,
    pub is_balanced: bool,
    pub e_to_e: f64,
    pub e_to_i: f64,
    pub i_to_e: f64,
    pub i_to_i: f64,
}

/// Validate row-major cardinality and finite coupling before indexing.
fn validate_matrix(knm: &[f64], n: usize) -> SpoResult<()> {
    if n.checked_mul(n) != Some(knm.len()) {
        return Err(SpoError::InvalidDimension(
            "knm must have exactly n * n values".into(),
        ));
    }
    if knm.iter().any(|v| !v.is_finite()) {
        return Err(SpoError::InvalidConfig(
            "knm must contain only finite values".into(),
        ));
    }
    Ok(())
}

/// Canonicalise a source set, retaining the historical out-of-range policy.
fn index_set(indices: &[usize], n: usize) -> Vec<usize> {
    let mut indices: Vec<_> = indices.iter().copied().filter(|&i| i < n).collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Compute a scaled compensated mean; empty or zero groups have zero mean.
fn mean<I: Iterator<Item = f64> + Clone>(values: I) -> f64 {
    let scale = values.clone().fold(0.0_f64, |s, v| s.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut correction = 0.0;
    let mut count = 0usize;
    for value in values {
        let value = value / scale;
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
        count += 1;
    }
    (sum + correction) / count as f64 * scale
}

/// Mean over canonical source rows and target columns, including the diagonal.
fn block_mean(knm: &[f64], n: usize, rows: &[usize], cols: &[usize]) -> f64 {
    mean(
        rows.iter()
            .flat_map(|&i| cols.iter().map(move |&j| knm[i * n + j])),
    )
}

/// Compute signed mean outgoing strengths and four directed block means.
///
/// Each group is a set: repeated indices count once, indices at least n are
/// ignored, and empty groups have zero mean. A denominator with magnitude
/// below 1e-15 is silent: ratio is infinity for positive excitation and one
/// otherwise. Other ratios are signed quotients, with balance in [0.8, 1.2].
///
/// # Errors
/// Returns an error for incorrect n-by-n cardinality, count overflow or
/// non-finite coupling. Empty zero-by-zero matrices remain valid.
pub fn compute_ei_balance(
    knm_flat: &[f64],
    n: usize,
    excitatory_indices: &[usize],
    inhibitory_indices: &[usize],
) -> SpoResult<EIBalanceResult> {
    validate_matrix(knm_flat, n)?;
    let e = index_set(excitatory_indices, n);
    let i = index_set(inhibitory_indices, n);
    let e_strength = mean(
        e.iter()
            .flat_map(|&row| (0..n).map(move |col| knm_flat[row * n + col])),
    );
    let i_strength = mean(
        i.iter()
            .flat_map(|&row| (0..n).map(move |col| knm_flat[row * n + col])),
    );
    let ratio = if i_strength.abs() < 1e-15 {
        if e_strength > 0.0 {
            f64::INFINITY
        } else {
            1.0
        }
    } else {
        e_strength / i_strength
    };
    Ok(EIBalanceResult {
        ratio,
        excitatory_strength: e_strength,
        inhibitory_strength: i_strength,
        is_balanced: (0.8..=1.2).contains(&ratio),
        e_to_e: block_mean(knm_flat, n, &e, &e),
        e_to_i: block_mean(knm_flat, n, &e, &i),
        i_to_e: block_mean(knm_flat, n, &i, &e),
        i_to_i: block_mean(knm_flat, n, &i, &i),
    })
}

/// Scale each inhibitory source row once and return an independent matrix.
///
/// Signed strengths are retained. Silent source groups or a ratio already
/// within 1e-10 of target return an unchanged copy. Achieving the target by
/// this single scaling requires disjoint source groups, adequate f64 precision
/// and an adjusted inhibitory mean with magnitude at least 1e-15; otherwise the
/// summary uses its silent convention. A scale rounded to signed zero refuses.
///
/// # Errors
/// Returns an error for malformed/non-finite coupling, a non-positive or
/// non-finite target, a scale that underflows to zero, or a non-finite scale or
/// adjusted element.
pub fn adjust_ei_ratio(
    knm_flat: &[f64],
    n: usize,
    excitatory_indices: &[usize],
    inhibitory_indices: &[usize],
    target_ratio: f64,
) -> SpoResult<Vec<f64>> {
    if !target_ratio.is_finite() || target_ratio <= 0.0 {
        return Err(SpoError::InvalidConfig(
            "target_ratio must be a finite positive real".into(),
        ));
    }
    let balance = compute_ei_balance(knm_flat, n, excitatory_indices, inhibitory_indices)?;
    if balance.inhibitory_strength.abs() < 1e-15
        || balance.excitatory_strength.abs() < 1e-15
        || (balance.ratio - target_ratio).abs() < 1e-10
    {
        return Ok(knm_flat.to_vec());
    }
    let scale = balance.ratio / target_ratio;
    if !scale.is_finite() || scale == 0.0 {
        return Err(SpoError::InvalidConfig(
            "E/I adjustment must remain finite with a non-zero scale".into(),
        ));
    }
    let mut result = knm_flat.to_vec();
    for idx in index_set(inhibitory_indices, n) {
        for j in 0..n {
            result[idx * n + j] *= scale;
        }
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(SpoError::InvalidConfig(
            "E/I adjustment must remain finite".into(),
        ));
    }
    Ok(result)
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
        let result = compute_ei_balance(&knm, n, &e_idx, &i_idx).expect("valid E/I input");
        assert!((result.ratio - 1.0).abs() < 1e-10);
        assert!(result.is_balanced);
    }

    #[test]
    fn test_interaction_type_breakdown() {
        // 4 oscillators, E = {0, 1}, I = {2, 3}. Each block has a distinct
        // constant value so the directed means are exactly recoverable.
        let n = 4;
        let mut knm = vec![0.0; n * n];
        for &i in &[0usize, 1] {
            for &j in &[0usize, 1] {
                knm[i * n + j] = 2.0; // E→E
            }
            for &j in &[2usize, 3] {
                knm[i * n + j] = 0.5; // E→I
            }
        }
        for &i in &[2usize, 3] {
            for &j in &[0usize, 1] {
                knm[i * n + j] = 1.5; // I→E
            }
            for &j in &[2usize, 3] {
                knm[i * n + j] = 3.0; // I→I
            }
        }
        let r = compute_ei_balance(&knm, n, &[0, 1], &[2, 3]).expect("valid E/I input");
        assert!((r.e_to_e - 2.0).abs() < 1e-12);
        assert!((r.e_to_i - 0.5).abs() < 1e-12);
        assert!((r.i_to_e - 1.5).abs() < 1e-12);
        assert!((r.i_to_i - 3.0).abs() < 1e-12);
        // Aggregate excitatory strength = mean over E rows (all targets) =
        // (2.0 + 0.5) / 2 = 1.25; inhibitory = (1.5 + 3.0) / 2 = 2.25.
        assert!((r.excitatory_strength - 1.25).abs() < 1e-12);
        assert!((r.inhibitory_strength - 2.25).abs() < 1e-12);
    }

    #[test]
    fn test_excitation_dominated() {
        let n = 4;
        let mut knm = vec![1.0; n * n];
        // Make excitatory rows stronger
        for j in 0..n {
            knm[j] = 3.0;
            knm[n + j] = 3.0;
        }
        let result = compute_ei_balance(&knm, n, &[0, 1], &[2, 3]).expect("valid E/I input");
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
        let result = compute_ei_balance(&knm, n, &[0, 1, 2], &[]).expect("valid E/I input");
        assert_eq!(result.inhibitory_strength, 0.0);
        assert_eq!(result.ratio, f64::INFINITY);
    }

    #[test]
    fn test_no_excitatory() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let result = compute_ei_balance(&knm, n, &[], &[0, 1, 2]).expect("valid E/I input");
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
            knm[j] = 2.0;
            knm[n + j] = 2.0;
        }
        let adjusted = adjust_ei_ratio(&knm, n, &[0, 1], &[2, 3], 1.0).expect("valid E/I input");
        let new_balance =
            compute_ei_balance(&adjusted, n, &[0, 1], &[2, 3]).expect("valid E/I input");
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
        let adjusted = adjust_ei_ratio(&knm, n, &[0], &[1, 2], 1.0).expect("valid E/I input");
        assert_eq!(adjusted, knm);
    }

    #[test]
    fn test_empty_coupling() {
        let result = compute_ei_balance(&[], 0, &[], &[]).expect("valid E/I input");
        assert_eq!(result.ratio, 1.0);
        assert_eq!(result.excitatory_strength, 0.0);
    }

    #[test]
    fn test_out_of_bounds_indices_ignored() {
        let n = 3;
        let knm = vec![1.0; n * n];
        let result = compute_ei_balance(&knm, n, &[0, 100], &[1]).expect("valid E/I input");
        // Index 100 is out of bounds, should be skipped
        assert!(result.excitatory_strength > 0.0);
    }
    #[test]
    fn duplicate_sets_scale_once() {
        let k = [0., 2., 4., 6., 0., 8., 10., 12., 0.];
        let b = compute_ei_balance(&k, 3, &[0, 0, 1, 99], &[2, 2]).expect("valid E/I input");
        assert!((b.ratio - 5. / 11.).abs() < 1e-15);
        assert_eq!(b.e_to_e, 2.);
        assert_eq!(b.e_to_i, 6.);
        assert_eq!(b.i_to_e, 11.);
        let a = adjust_ei_ratio(&k, 3, &[0, 0, 1], &[2, 2, 99], 1.).expect("valid E/I input");
        assert!((a[6] - 50. / 11.).abs() < 1e-14);
        assert_eq!(&a[..6], &k[..6]);
    }

    #[test]
    fn finite_scaled_and_cancelled_means() {
        for value in [0., 1e308, -1e308] {
            let k = [value; 4];
            let b = compute_ei_balance(&k, 2, &[0], &[1]).expect("valid E/I input");
            assert_eq!(b.ratio, 1.);
            assert_eq!(b.excitatory_strength, value);
            assert_eq!(b.inhibitory_strength, value);
            assert_eq!(
                adjust_ei_ratio(&k, 2, &[0], &[1], 1.).expect("valid E/I input"),
                k
            );
        }
        let k = [1e308, 1e308, -1e308, -1e308].repeat(4);
        let b = compute_ei_balance(&k, 4, &[0, 1], &[2, 3]).expect("valid E/I input");
        assert_eq!(b.excitatory_strength, 0.);
        assert_eq!(b.inhibitory_strength, 0.);
        assert_eq!(b.e_to_e, 1e308);
        assert_eq!(b.e_to_i, -1e308);
    }

    #[test]
    fn signed_quotients_and_silent_sources() {
        for k in [[0., 2., -1., 0.], [0., -2., -1., 0.]] {
            let b = compute_ei_balance(&k, 2, &[0], &[1]).expect("valid E/I input");
            assert_eq!(b.ratio, -k[1]);
            let a = adjust_ei_ratio(&k, 2, &[0], &[1], 1.).expect("valid E/I input");
            assert_eq!(a[2], k[1]);
        }
        let k = [-1., -1., 0., 0.];
        assert_eq!(
            compute_ei_balance(&k, 2, &[0], &[1])
                .expect("valid E/I input")
                .ratio,
            1.
        );
        assert_eq!(
            adjust_ei_ratio(&k, 2, &[0], &[1], 1.).expect("valid E/I input"),
            k
        );
        assert_eq!(
            adjust_ei_ratio(&k, 2, &[], &[1], 1.).expect("valid E/I input"),
            k
        );
    }

    #[test]
    fn invalid_matrix_and_target_refuse() {
        for (k, n) in [(vec![1.], 2), (vec![], usize::MAX), (vec![f64::NAN; 4], 2)] {
            assert!(compute_ei_balance(&k, n, &[0], &[1]).is_err());
            assert!(adjust_ei_ratio(&k, n, &[0], &[1], 1.).is_err());
        }
        let k = [0., 2., 1., 0.];
        for target in [0., -1., f64::NAN, f64::INFINITY, 1e-310] {
            assert!(adjust_ei_ratio(&k, 2, &[0], &[1], target).is_err());
        }
        let k = [0., 1e308, 1e308, -1e308 + 1e294];
        assert!(adjust_ei_ratio(&k, 2, &[0], &[1], 1.).is_err());
    }

    #[test]
    fn underflowing_scale_refuses_and_recovers_with_signed_sources() {
        for excitation in [-2., 2.] {
            for inhibition in [-1e308, 1e308] {
                let k = [0., excitation, inhibition, 0.];
                let before = k;
                assert!(adjust_ei_ratio(&k, 2, &[0, 0], &[1, 1], 1e308).is_err());
                assert_eq!(k, before);
                let a = adjust_ei_ratio(&k, 2, &[0, 0], &[1, 1], 1.)
                    .expect("representable recovery scale");
                assert!((a[2] - excitation).abs() < 1e-14);
                let b = compute_ei_balance(&a, 2, &[0], &[1]).expect("valid recovered input");
                assert!((b.ratio - 1.).abs() < 1e-14);
                assert_eq!(k, before);
            }
        }
    }

    #[test]
    fn representable_scale_retains_silent_summary() {
        let k = [0., 2., 1., 0.];
        let a = adjust_ei_ratio(&k, 2, &[0], &[1], 1e20).expect("representable scale");
        assert!((a[2] / 2e-20 - 1.).abs() < 1e-14);
        let b = compute_ei_balance(&a, 2, &[0], &[1]).expect("valid silent input");
        assert_eq!(b.ratio, f64::INFINITY);
        assert_eq!(k, [0., 2., 1., 0.]);
    }
}
