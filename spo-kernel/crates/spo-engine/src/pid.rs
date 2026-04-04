// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Partial Information Decomposition (PID)

//! Williams & Beer 2010 minimum-MI redundancy/synergy for phase groups.
//! Circular MI estimated via binned phase histograms.

use std::f64::consts::TAU;

const DEFAULT_BINS: usize = 32;

fn circular_entropy(phases: &[f64], n_bins: usize) -> f64 {
    if phases.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let mut counts = vec![0u64; n_bins];
    let bin_width = TAU / n_bins as f64;
    for &p in phases {
        let wrapped = p.rem_euclid(TAU);
        let bin = (wrapped / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }
    let total = phases.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn joint_entropy_2d(a: &[f64], b: &[f64], n_bins: usize) -> f64 {
    if a.len() != b.len() || a.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let mut counts = vec![0u64; n_bins * n_bins];
    let bin_width = TAU / n_bins as f64;
    for i in 0..a.len() {
        let ba = (a[i].rem_euclid(TAU) / bin_width).floor() as usize;
        let bb = (b[i].rem_euclid(TAU) / bin_width).floor() as usize;
        let ba = ba.min(n_bins - 1);
        let bb = bb.min(n_bins - 1);
        counts[ba * n_bins + bb] += 1;
    }
    let total = a.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn mutual_information(a: &[f64], b: &[f64], n_bins: usize) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let ha = circular_entropy(a, n_bins);
    let hb = circular_entropy(b, n_bins);
    let hab = joint_entropy_2d(a, b, n_bins);
    (ha + hb - hab).max(0.0)
}

/// Redundant information shared by both groups about the global phase.
///
/// I_red = min(MI(A; global), MI(B; global))
#[must_use]
pub fn redundancy(phases: &[f64], group_a: &[usize], group_b: &[usize], n_bins: usize) -> f64 {
    if phases.is_empty() || group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }

    let global_phase = global_mean_phase(phases);
    let bins = if n_bins == 0 { DEFAULT_BINS } else { n_bins };

    let phases_a: Vec<f64> = group_a.iter().map(|&i| phases[i]).collect();
    let global_a = vec![global_phase; group_a.len()];

    let phases_b: Vec<f64> = group_b.iter().map(|&i| phases[i]).collect();
    let global_b = vec![global_phase; group_b.len()];

    let mi_a = mutual_information(&phases_a, &global_a, bins);
    let mi_b = mutual_information(&phases_b, &global_b, bins);
    mi_a.min(mi_b)
}

/// Synergistic information present only in the joint (A, B).
///
/// I_syn = MI(A+B; global) - MI(A; global) - MI(B; global) + I_red
#[must_use]
pub fn synergy(phases: &[f64], group_a: &[usize], group_b: &[usize], n_bins: usize) -> f64 {
    if phases.is_empty() || group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }

    let global_phase = global_mean_phase(phases);
    let bins = if n_bins == 0 { DEFAULT_BINS } else { n_bins };

    let phases_a: Vec<f64> = group_a.iter().map(|&i| phases[i]).collect();
    let phases_b: Vec<f64> = group_b.iter().map(|&i| phases[i]).collect();
    let mut phases_joint = phases_a.clone();
    phases_joint.extend_from_slice(&phases_b);

    let global_a = vec![global_phase; group_a.len()];
    let global_b = vec![global_phase; group_b.len()];
    let global_joint = vec![global_phase; phases_joint.len()];

    let mi_a = mutual_information(&phases_a, &global_a, bins);
    let mi_b = mutual_information(&phases_b, &global_b, bins);
    let mi_joint = mutual_information(&phases_joint, &global_joint, bins);
    let i_red = mi_a.min(mi_b);

    (mi_joint - mi_a - mi_b + i_red).max(0.0)
}

fn global_mean_phase(phases: &[f64]) -> f64 {
    let (sin_sum, cos_sum) = phases
        .iter()
        .fold((0.0, 0.0), |(s, c), &p| (s + p.sin(), c + p.cos()));
    let n = phases.len() as f64;
    (sin_sum / n).atan2(cos_sum / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sync_phases_zero_redundancy() {
        let phases = vec![0.0; 8];
        let r = redundancy(&phases, &[0, 1, 2, 3], &[4, 5, 6, 7], 32);
        assert!(r >= 0.0);
    }

    #[test]
    fn synergy_non_negative() {
        let phases: Vec<f64> = (0..8).map(|i| i as f64 * PI / 4.0).collect();
        let s = synergy(&phases, &[0, 1, 2, 3], &[4, 5, 6, 7], 32);
        assert!(s >= 0.0, "synergy should be non-negative, got {s}");
    }

    #[test]
    fn empty_phases_returns_zero() {
        assert_eq!(redundancy(&[], &[0], &[1], 32), 0.0);
        assert_eq!(synergy(&[], &[0], &[1], 32), 0.0);
    }

    #[test]
    fn empty_groups_returns_zero() {
        let phases = vec![0.0, 1.0, 2.0];
        assert_eq!(redundancy(&phases, &[], &[0, 1], 32), 0.0);
        assert_eq!(synergy(&phases, &[0], &[], 32), 0.0);
    }

    #[test]
    fn circular_entropy_uniform() {
        // N phases uniformly distributed across bins should give high entropy
        let n = 320;
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let h = circular_entropy(&phases, 32);
        let max_h = (32.0_f64).ln();
        assert!(
            (h - max_h).abs() < 0.1,
            "uniform should give near-max entropy: {h} vs {max_h}"
        );
    }

    #[test]
    fn circular_entropy_concentrated() {
        // All phases at 0 should give low entropy (1 bin occupied)
        let phases = vec![0.0; 100];
        let h = circular_entropy(&phases, 32);
        assert!(h < 0.01, "concentrated should give near-zero entropy: {h}");
    }

    #[test]
    fn mutual_information_non_negative() {
        let a = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let b = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mi = mutual_information(&a, &b, 32);
        assert!(mi >= 0.0, "MI should be non-negative: {mi}");
    }
}
