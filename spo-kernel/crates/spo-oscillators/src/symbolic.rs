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
    (TAU * state_index as f64 / n_states as f64) % TAU
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
    if step_size == 0 {
        return 0.2;
    }
    if step_size == 1 {
        return 1.0;
    }
    if n_states == 0 {
        return 0.1;
    }
    (1.0 - (step_size as f64 - 1.0) / n_states as f64).max(0.1)
}

/// Vectorised ring-phase mapping for a full symbolic state sequence.
#[must_use]
pub fn ring_phases(state_indices: &[i64], n_states: usize) -> Vec<f64> {
    if n_states == 0 {
        return vec![0.0; state_indices.len()];
    }
    state_indices
        .iter()
        .map(|&index| {
            let wrapped = index.rem_euclid(n_states as i64);
            ring_phase(wrapped as usize, n_states)
        })
        .collect()
}

/// Vectorised graph-walk phase mapping mirroring Python fallback semantics.
#[must_use]
pub fn graph_walk_phases(state_indices: &[i64], n_states: usize) -> Vec<f64> {
    if state_indices.len() < 2 {
        return ring_phases(state_indices, n_states);
    }
    let mut cumulative: Vec<usize> = Vec::with_capacity(state_indices.len());
    cumulative.push(0);
    let mut running: usize = 0;
    for window in state_indices.windows(2) {
        let step = window[1].abs_diff(window[0]) as usize;
        running = running.saturating_add(step);
        cumulative.push(running);
    }
    let walk_length = if running > 0 { running } else { 1 };
    cumulative
        .into_iter()
        .map(|position| graph_walk_phase(position, walk_length))
        .collect()
}

/// Vectorised transition-quality mapping for a full symbolic state sequence.
#[must_use]
pub fn transition_qualities(
    state_indices: &[i64],
    n_states: usize,
    initial_quality: f64,
) -> Vec<f64> {
    if state_indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(state_indices.len());
    out.push(initial_quality);
    for window in state_indices.windows(2) {
        let step = window[1].abs_diff(window[0]) as usize;
        out.push(transition_quality(step, n_states));
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
    fn transition_qualities_vector_len_matches() {
        let qualities = transition_qualities(&[0, 1, 4], 8, 0.5);
        assert_eq!(qualities.len(), 3);
        assert_eq!(qualities[0], 0.5);
    }
}
