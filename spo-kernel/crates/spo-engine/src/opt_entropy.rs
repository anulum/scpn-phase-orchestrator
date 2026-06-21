// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ordinal-Pattern Transition Entropy (OPT-entropy)

//! Ordinal-pattern transition entropy of a scalar series.
//!
//! Bandt–Pompe ordinal patterns (Lehmer-encoded stable argsort) feed a
//! consecutive-pattern transition distribution whose normalised Shannon
//! entropy collapses ahead of an explosive (first-order) synchronisation
//! onset. Algorithm matches the NumPy / Go / Julia / Mojo references
//! bit-for-bit.

/// Factorial of a small embedding dimension.
#[must_use]
fn factorial(value: usize) -> i64 {
    let mut result: i64 = 1;
    for factor in 2..=value {
        result *= factor as i64;
    }
    result
}

/// Stable ascending argsort of a window with index tie-breaking.
fn stable_argsort(window: &[f64]) -> Vec<usize> {
    let dimension = window.len();
    let mut used = vec![false; dimension];
    let mut perm = vec![0usize; dimension];
    for slot in perm.iter_mut() {
        let mut best: isize = -1;
        for idx in 0..dimension {
            if used[idx] {
                continue;
            }
            let take = best == -1
                || window[idx] < window[best as usize]
                || (window[idx] == window[best as usize] && idx < best as usize);
            if take {
                best = idx as isize;
            }
        }
        *slot = best as usize;
        used[best as usize] = true;
    }
    perm
}

/// Lehmer code of a permutation in `[0, D! - 1]`.
fn lehmer_code(perm: &[usize], fact: &[i64]) -> i64 {
    let dimension = perm.len();
    let mut code: i64 = 0;
    for i in 0..dimension {
        let mut smaller: i64 = 0;
        for j in (i + 1)..dimension {
            if perm[j] < perm[i] {
                smaller += 1;
            }
        }
        code += smaller * fact[dimension - 1 - i];
    }
    code
}

/// Number of ordinal windows `M = T - (D - 1) * τ` (saturating at zero).
#[must_use]
fn window_count(length: usize, dimension: usize, delay: usize) -> usize {
    length.saturating_sub((dimension - 1) * delay)
}

/// Lehmer-encoded ordinal-pattern sequence of `series`.
#[must_use]
pub fn ordinal_pattern_sequence(series: &[f64], dimension: usize, delay: usize) -> Vec<i64> {
    let count = window_count(series.len(), dimension, delay);
    let fact: Vec<i64> = (0..dimension).map(factorial).collect();
    let mut codes = Vec::with_capacity(count);
    let mut window = vec![0.0f64; dimension];
    for m in 0..count {
        for k in 0..dimension {
            window[k] = series[m + k * delay];
        }
        let perm = stable_argsort(&window);
        codes.push(lehmer_code(&perm, &fact));
    }
    codes
}

/// Normalised ordinal-pattern transition entropy in `[0, 1]`.
#[must_use]
pub fn transition_entropy(series: &[f64], dimension: usize, delay: usize) -> f64 {
    let codes = ordinal_pattern_sequence(series, dimension, delay);
    let n_codes = codes.len();
    if n_codes < 2 {
        return 0.0;
    }
    let fact_d = factorial(dimension);
    let total = n_codes - 1;
    let mut keys: Vec<i64> = Vec::with_capacity(total);
    for m in 0..total {
        keys.push(codes[m] * fact_d + codes[m + 1]);
    }
    keys.sort_unstable();

    // Run-length count of distinct transitions in ascending key order.
    let mut counts: Vec<i64> = Vec::new();
    let mut run: i64 = 1;
    for idx in 1..keys.len() {
        if keys[idx] == keys[idx - 1] {
            run += 1;
        } else {
            counts.push(run);
            run = 1;
        }
    }
    counts.push(run);

    let distinct = counts.len();
    if distinct < 2 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0;
    for &count in &counts {
        let probability = count as f64 / total_f;
        entropy -= probability * probability.ln();
    }
    let max_entropy = (distinct as f64).ln();
    if max_entropy < 1e-15 {
        return 0.0;
    }
    (entropy / max_entropy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increasing_series_is_identity_pattern() {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0];
        let codes = ordinal_pattern_sequence(&series, 3, 1);
        assert_eq!(codes, vec![0, 0, 0]);
    }

    #[test]
    fn decreasing_series_is_max_pattern() {
        let series = [5.0, 4.0, 3.0, 2.0, 1.0];
        let codes = ordinal_pattern_sequence(&series, 3, 1);
        assert_eq!(codes, vec![5, 5, 5]);
    }

    #[test]
    fn constant_series_has_zero_entropy() {
        let series = vec![2.0; 64];
        assert_eq!(transition_entropy(&series, 3, 1), 0.0);
    }

    #[test]
    fn short_series_has_zero_entropy() {
        assert_eq!(transition_entropy(&[1.0, 2.0], 3, 1), 0.0);
        assert!(ordinal_pattern_sequence(&[1.0, 2.0], 3, 1).is_empty());
    }

    #[test]
    fn entropy_in_unit_interval() {
        let series: Vec<f64> = (0..500).map(|i| ((i as f64) * 0.7).sin()).collect();
        let value = transition_entropy(&series, 3, 1);
        assert!(
            (0.0..=1.0).contains(&value),
            "entropy {value} out of [0, 1]"
        );
    }
}
