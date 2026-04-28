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
}
