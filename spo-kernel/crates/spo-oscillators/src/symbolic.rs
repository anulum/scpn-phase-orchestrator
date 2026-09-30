// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Symbolic oscillator

//!
//! Maps discrete state indices to phases on the unit circle.

use std::f64::consts::TAU;

/// Ring-phase: θ = 2π·s/N, mapping state index to unit circle.
#[must_use]
pub fn ring_phase(state_index: usize, n_states: usize) -> f64 {
    if n_states == 0 {
        return 0.0;
    }
    (TAU * ((state_index % n_states) as f64 / n_states as f64)) % TAU
}

/// Graph-walk phase: normalise sequential position to [0, 2π).
#[must_use]
pub fn graph_walk_phase(position: usize, walk_length: usize) -> f64 {
    if walk_length == 0 {
        return 0.0;
    }
    (TAU * position as f64 / walk_length as f64) % TAU
}

/// Transition quality for symbolic sequences.
/// step=0 → stalled (0.2), step=1 → ideal (1.0), larger → penalised.
#[must_use]
pub fn transition_quality(step_size: usize, n_states: usize) -> f64 {
    transition_quality_for_distance(step_size as u128, n_states)
}

/// Score a full-width distance without truncating it to the target pointer width.
fn transition_quality_for_distance(step_size: u128, n_states: usize) -> f64 {
    if step_size == 0 {
        return 0.2;
    }
    if step_size == 1 {
        return 1.0;
    }
    if n_states == 0 {
        return 0.1;
    }
    (1.0 - (step_size - 1) as f64 / n_states as f64).max(0.1)
}

/// Vectorised ring-phase mapping for a full symbolic state sequence.
#[must_use]
pub fn ring_phases(state_indices: &[i64], n_states: usize) -> Vec<f64> {
    ring_phases_from_indices(state_indices, n_states)
}

/// Vectorised ring mapping retaining the full unsigned 64-bit label domain.
#[must_use]
pub fn ring_phases_unsigned(state_indices: &[u64], n_states: usize) -> Vec<f64> {
    ring_phases_from_indices(state_indices, n_states)
}

/// Reduce signed or unsigned labels before floating conversion, including large vocabularies.
fn ring_phases_from_indices<StateIndex: Copy + Into<i128>>(
    state_indices: &[StateIndex],
    n_states: usize,
) -> Vec<f64> {
    if n_states == 0 {
        return vec![0.0; state_indices.len()];
    }
    state_indices
        .iter()
        .map(|&index| {
            let wrapped = index.into().rem_euclid(n_states as i128);
            ring_phase(wrapped as usize, n_states)
        })
        .collect()
}

/// Normalise exact linear walk distances, converting prefixes to f64 only at the end.
///
/// Each i64 transition fits u64; the sum over an addressable slice fits u128.
#[must_use]
pub fn graph_walk_phases(state_indices: &[i64], n_states: usize) -> Vec<f64> {
    graph_walk_phases_from_indices(state_indices, n_states)
}

/// Graph-walk mapping retaining unsigned labels instead of reinterpreting their sign bit.
#[must_use]
pub fn graph_walk_phases_unsigned(state_indices: &[u64], n_states: usize) -> Vec<f64> {
    graph_walk_phases_from_indices(state_indices, n_states)
}

/// Accumulate exact distances for the signed and unsigned 64-bit entry points.
fn graph_walk_phases_from_indices<StateIndex: Copy + Into<i128>>(
    state_indices: &[StateIndex],
    n_states: usize,
) -> Vec<f64> {
    if state_indices.len() < 2 {
        return ring_phases_from_indices(state_indices, n_states);
    }
    let mut cumulative: Vec<u128> = Vec::with_capacity(state_indices.len());
    cumulative.push(0);
    let mut running: u128 = 0;
    for window in state_indices.windows(2) {
        let step = window[1].into().abs_diff(window[0].into());
        running += step;
        cumulative.push(running);
    }
    let walk_length = if running > 0 { running } else { 1 };
    cumulative
        .into_iter()
        .map(|position| (TAU * position as f64 / walk_length as f64) % TAU)
        .collect()
}

/// Vectorised transition-quality mapping for a full symbolic state sequence.
#[must_use]
pub fn transition_qualities(
    state_indices: &[i64],
    n_states: usize,
    initial_quality: f64,
) -> Vec<f64> {
    transition_qualities_from_indices(state_indices, n_states, initial_quality)
}

/// Linear transition qualities for the full unsigned label domain.
#[must_use]
pub fn transition_qualities_unsigned(
    state_indices: &[u64],
    n_states: usize,
    initial_quality: f64,
) -> Vec<f64> {
    transition_qualities_from_indices(state_indices, n_states, initial_quality)
}

/// Preserve full-width distances before applying the shared linear quality law.
fn transition_qualities_from_indices<StateIndex: Copy + Into<i128>>(
    state_indices: &[StateIndex],
    n_states: usize,
    initial_quality: f64,
) -> Vec<f64> {
    if state_indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(state_indices.len());
    out.push(initial_quality);
    for window in state_indices.windows(2) {
        let step = window[1].into().abs_diff(window[0].into());
        out.push(transition_quality_for_distance(step, n_states));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_phase_zero() {
        assert_eq!(ring_phase(0, 4), 0.0);
    }

    #[test]
    fn ring_phase_quarter() {
        let p = ring_phase(1, 4);
        assert!((p - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn ring_phase_wraps() {
        let p = ring_phase(4, 4);
        assert!(p < 1e-12); // 2π mod 2π = 0
    }

    #[test]
    fn ring_phase_n_zero() {
        assert_eq!(ring_phase(0, 0), 0.0);
    }

    #[test]
    fn graph_walk_midpoint() {
        let p = graph_walk_phase(5, 10);
        assert!((p - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn graph_walk_zero_length() {
        assert_eq!(graph_walk_phase(0, 0), 0.0);
    }

    #[test]
    fn quality_stalled() {
        assert_eq!(transition_quality(0, 10), 0.2);
    }

    #[test]
    fn quality_ideal() {
        assert_eq!(transition_quality(1, 10), 1.0);
    }

    #[test]
    fn quality_large_jump() {
        let q = transition_quality(5, 10);
        assert!(q < 1.0);
        assert!(q >= 0.1);
    }

    #[test]
    fn quality_max_jump() {
        let q = transition_quality(100, 10);
        assert_eq!(q, 0.1);
    }

    #[test]
    fn ring_phases_vector_len_matches() {
        let phases = ring_phases(&[0, 1, 2, 3], 4);
        assert_eq!(phases.len(), 4);
    }

    #[test]
    fn graph_walk_phases_vector_len_matches() {
        let phases = graph_walk_phases(&[0, 2, 5], 8);
        assert_eq!(phases.len(), 3);
    }

    #[test]
    fn graph_walk_preserves_full_signed_distance() {
        let phases = graph_walk_phases(&[-(1_i64 << 62), 1_i64 << 62, 0], 4);
        assert_eq!(phases[0], 0.0);
        assert!((phases[1] - 2.0 * TAU / 3.0).abs() < 1e-12);
        assert_eq!(phases[2], 0.0);
    }

    #[test]
    fn graph_walk_accumulates_beyond_u64_without_saturation() {
        let labels: Vec<i64> = (0..257)
            .map(|index| if index % 2 == 0 { i64::MIN } else { i64::MAX })
            .collect();
        let phases = graph_walk_phases(&labels, 4);
        for (index, phase) in phases.iter().enumerate() {
            let expected = (TAU * index as f64 / 256.0) % TAU;
            assert!((phase - expected).abs() < 1e-12);
        }
        assert_eq!(phases[128], TAU / 2.0);
    }

    #[test]
    fn graph_walk_stationary_signed_extreme_has_zero_distance() {
        assert_eq!(graph_walk_phases(&[i64::MIN; 3], 4), vec![0.0; 3]);
    }

    #[test]
    fn unsigned_full_span_phases_and_qualities_preserve_labels() {
        let labels = [0, u64::MAX, 0];
        let ring = ring_phases_unsigned(&labels, 4);
        let graph = graph_walk_phases_unsigned(&labels, 4);
        assert_eq!(ring, vec![0.0, 3.0 * TAU / 4.0, 0.0]);
        assert_eq!(graph, vec![0.0, TAU / 2.0, 0.0]);
        assert_eq!(
            transition_qualities_unsigned(&labels, 4, 0.5),
            vec![0.5, 0.1, 0.1]
        );
    }

    #[test]
    fn ring_reduces_before_float_and_preserves_full_usize_vocabulary() {
        assert_eq!(ring_phase(usize::MAX, 4), 3.0 * TAU / 4.0);
        let expected = TAU * (1.0 / usize::MAX as f64);
        assert_eq!(
            ring_phases(&[0, 1, 0], usize::MAX),
            vec![0.0, expected, 0.0]
        );
    }

    #[test]
    fn transition_quality_does_not_truncate_a_64_bit_distance() {
        assert_eq!(
            transition_qualities(&[0, (1_i64 << 32) + 1], 4, 0.5),
            vec![0.5, 0.1]
        );
    }

    #[test]
    fn unsigned_empty_singleton_and_stationary_observations() {
        assert!(graph_walk_phases_unsigned(&[], 4).is_empty());
        assert!(transition_qualities_unsigned(&[], 4, 0.5).is_empty());
        assert_eq!(
            graph_walk_phases_unsigned(&[u64::MAX], 4),
            vec![3.0 * TAU / 4.0]
        );
        assert_eq!(graph_walk_phases_unsigned(&[u64::MAX; 3], 4), vec![0.0; 3]);
        assert_eq!(
            transition_qualities_unsigned(&[u64::MAX; 3], 4, 0.5),
            vec![0.5, 0.2, 0.2]
        );
        assert_eq!(ring_phases_unsigned(&[u64::MAX], 0), vec![0.0]);
    }

    #[test]
    fn transition_qualities_vector_len_matches() {
        let qualities = transition_qualities(&[0, 1, 4], 8, 0.5);
        assert_eq!(qualities.len(), 3);
        assert_eq!(qualities[0], 0.5);
    }
}
