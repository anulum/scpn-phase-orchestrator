// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Partial Information Decomposition (PID)

//! Time-series Williams & Beer 2010 partial information decomposition of two
//! oscillator groups about the global synchronisation state.
//!
//! The input is a phase history `(T, N)` (row-major). Each timestep is reduced
//! to three circular observables — the global order-parameter phase (target Y)
//! and the two group order-parameter phases (sources A, B) — binned into
//! `n_bins` equal-width phase bins. From the joint distribution over the `T`
//! samples:
//!
//! * `redundancy I_red = Σ_y p(y)·min(I_spec(Y=y; A), I_spec(Y=y; B))`,
//! * `synergy   I_syn = MI(A,B; Y) − MI(A; Y) − MI(B; Y) + I_red`,
//!
//! with the specific information `I_spec(Y=y; S) = Σ_s p(s|y)·log[p(y|s)/p(y)]`.

use std::f64::consts::TAU;

/// Bin a circular angle into `[0, n_bins)`.
fn bin_angle(angle: f64, n_bins: usize) -> usize {
    let wrapped = angle.rem_euclid(TAU);
    let bin = (wrapped / (TAU / n_bins as f64)).floor() as usize;
    bin.min(n_bins - 1)
}

/// Order-parameter phase (`atan2` of the mean unit vector) over `members` at
/// row `row` of a row-major `(T, N)` history.
fn group_phase(history: &[f64], n: usize, row: usize, members: &[usize]) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for &j in members {
        let theta = history[row * n + j];
        sin_sum += theta.sin();
        cos_sum += theta.cos();
    }
    let count = members.len() as f64;
    (sin_sum / count).atan2(cos_sum / count)
}

/// Global order-parameter phase over all `n` oscillators at `row`.
fn global_phase(history: &[f64], n: usize, row: usize) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for j in 0..n {
        let theta = history[row * n + j];
        sin_sum += theta.sin();
        cos_sum += theta.cos();
    }
    let count = n as f64;
    (sin_sum / count).atan2(cos_sum / count)
}

/// `MI(X; Y)` from a joint count matrix `joint[x * n_bins + y]` and marginals.
fn mutual_information(
    joint: &[f64],
    marg_x: &[f64],
    marg_y: &[f64],
    n_x: usize,
    n_bins: usize,
    total: f64,
) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    let mut mi = 0.0;
    for x in 0..n_x {
        if marg_x[x] <= 0.0 {
            continue;
        }
        for y in 0..n_bins {
            let cxy = joint[x * n_bins + y];
            if cxy <= 0.0 || marg_y[y] <= 0.0 {
                continue;
            }
            let p_xy = cxy / total;
            mi += p_xy * (p_xy / ((marg_x[x] / total) * (marg_y[y] / total))).ln();
        }
    }
    mi.max(0.0)
}

/// Williams & Beer `I_min` redundancy from joint/marginal counts.
fn i_min_redundancy(
    cay: &[f64],
    cby: &[f64],
    ca: &[f64],
    cb: &[f64],
    cy: &[f64],
    n_bins: usize,
    total: f64,
) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    let mut i_red = 0.0;
    for y in 0..n_bins {
        if cy[y] <= 0.0 {
            continue;
        }
        let p_y = cy[y] / total;
        let mut ispec_a = 0.0;
        for x in 0..n_bins {
            if cay[x * n_bins + y] <= 0.0 || ca[x] <= 0.0 {
                continue;
            }
            let p_a_given_y = cay[x * n_bins + y] / cy[y];
            let p_y_given_a = cay[x * n_bins + y] / ca[x];
            ispec_a += p_a_given_y * (p_y_given_a / p_y).ln();
        }
        let mut ispec_b = 0.0;
        for x in 0..n_bins {
            if cby[x * n_bins + y] <= 0.0 || cb[x] <= 0.0 {
                continue;
            }
            let p_b_given_y = cby[x * n_bins + y] / cy[y];
            let p_y_given_b = cby[x * n_bins + y] / cb[x];
            ispec_b += p_b_given_y * (p_y_given_b / p_y).ln();
        }
        i_red += p_y * ispec_a.min(ispec_b);
    }
    i_red.max(0.0)
}

/// Time-series PID returning `(redundancy, synergy)`.
///
/// `history` is the row-major `(t, n)` phase history; `group_a` / `group_b` are
/// oscillator index sets; `n_bins` is the phase-bin count.
#[must_use]
pub fn pid_decomposition(
    history: &[f64],
    t: usize,
    n: usize,
    group_a: &[usize],
    group_b: &[usize],
    n_bins: usize,
) -> (f64, f64) {
    if t == 0 || n == 0 || group_a.is_empty() || group_b.is_empty() || n_bins == 0 {
        return (0.0, 0.0);
    }

    let mut cy = vec![0.0; n_bins];
    let mut ca = vec![0.0; n_bins];
    let mut cb = vec![0.0; n_bins];
    let mut cay = vec![0.0; n_bins * n_bins];
    let mut cby = vec![0.0; n_bins * n_bins];
    let mut cab = vec![0.0; n_bins * n_bins];
    let mut caby = vec![0.0; n_bins * n_bins * n_bins];

    for row in 0..t {
        let y = bin_angle(global_phase(history, n, row), n_bins);
        let a = bin_angle(group_phase(history, n, row, group_a), n_bins);
        let b = bin_angle(group_phase(history, n, row, group_b), n_bins);
        cy[y] += 1.0;
        ca[a] += 1.0;
        cb[b] += 1.0;
        cay[a * n_bins + y] += 1.0;
        cby[b * n_bins + y] += 1.0;
        let ab = a * n_bins + b;
        cab[ab] += 1.0;
        caby[ab * n_bins + y] += 1.0;
    }

    let total = t as f64;
    let mi_a = mutual_information(&cay, &ca, &cy, n_bins, n_bins, total);
    let mi_b = mutual_information(&cby, &cb, &cy, n_bins, n_bins, total);
    let mi_ab = mutual_information(&caby, &cab, &cy, n_bins * n_bins, n_bins, total);
    let i_red = i_min_redundancy(&cay, &cby, &ca, &cb, &cy, n_bins, total);
    let synergy = (mi_ab - mi_a - mi_b + i_red).max(0.0);
    (i_red, synergy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coherent_groups(t: usize) -> (Vec<f64>, usize) {
        // Group A (0..4) coherent at a_row, group B (4..8) at b_row; the two
        // series sweep the circle so Y depends on both → genuine synergy.
        let n = 8;
        let mut history = vec![0.0; t * n];
        for row in 0..t {
            let a_phase = TAU * (row as f64) / (t as f64);
            let b_phase = TAU * (2.0 * row as f64) / (t as f64);
            for j in 0..4 {
                history[row * n + j] = a_phase;
            }
            for j in 4..8 {
                history[row * n + j] = b_phase;
            }
        }
        (history, n)
    }

    #[test]
    fn empty_inputs_return_zero() {
        assert_eq!(pid_decomposition(&[], 0, 0, &[0], &[1], 8), (0.0, 0.0));
        let (h, n) = coherent_groups(10);
        assert_eq!(pid_decomposition(&h, 10, n, &[], &[4, 5], 8), (0.0, 0.0));
    }

    #[test]
    fn non_negative_and_bounded() {
        let (h, n) = coherent_groups(2000);
        let (red, syn) = pid_decomposition(&h, 2000, n, &[0, 1, 2, 3], &[4, 5, 6, 7], 8);
        assert!(red >= 0.0, "redundancy {red}");
        assert!(syn >= 0.0, "synergy {syn}");
    }

    #[test]
    fn fully_redundant_has_zero_synergy() {
        // All oscillators share one sweeping phase → A = B = Y, so the entire
        // shared information is redundant and synergy vanishes.
        let n = 8;
        let t = 2000;
        let mut history = vec![0.0; t * n];
        for row in 0..t {
            let phase = TAU * (row as f64) / (t as f64);
            for j in 0..n {
                history[row * n + j] = phase;
            }
        }
        let (red, syn) = pid_decomposition(&history, t, n, &[0, 1, 2, 3], &[4, 5, 6, 7], 8);
        assert!(
            red > 0.0,
            "fully-redundant redundancy should be positive: {red}"
        );
        assert!(syn < 1e-9, "fully-redundant synergy should vanish: {syn}");
    }

    #[test]
    fn single_snapshot_is_zero() {
        let n = 8;
        let history: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert_eq!(
            pid_decomposition(&history, 1, n, &[0, 1, 2, 3], &[4, 5, 6, 7], 8),
            (0.0, 0.0)
        );
    }
}
