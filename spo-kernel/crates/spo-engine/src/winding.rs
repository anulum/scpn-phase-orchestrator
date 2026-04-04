// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase winding number computation

//! Winding number computation for oscillator phase trajectories.
//!
//! Counts how many full 2π rotations each oscillator completes
//! over a trajectory, handling wrap-around via incremental unwrapping.

use std::f64::consts::{PI, TAU};

/// Compute cumulative winding numbers for each oscillator.
///
/// # Arguments
/// * `phases_history` — flat row-major array of shape (T, N)
/// * `t` — number of timesteps
/// * `n` — number of oscillators
///
/// # Returns
/// Vec of N winding numbers (signed integers as i64).
///
/// # Errors
/// Returns empty vec if t < 2 or n == 0.
#[must_use]
pub fn winding_numbers(phases_history: &[f64], t: usize, n: usize) -> Vec<i64> {
    if t < 2 || n == 0 || phases_history.len() != t * n {
        return vec![0i64; n];
    }

    let mut cumulative = vec![0.0f64; n];

    for step in 1..t {
        let prev_row = &phases_history[(step - 1) * n..step * n];
        let curr_row = &phases_history[step * n..(step + 1) * n];
        for j in 0..n {
            let dtheta = curr_row[j] - prev_row[j];
            // Wrap to [-π, π]
            let wrapped = (dtheta + PI).rem_euclid(TAU) - PI;
            cumulative[j] += wrapped;
        }
    }

    cumulative
        .iter()
        .map(|&c| (c / TAU).floor() as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn no_rotation_gives_zero() {
        // 5 timesteps, 3 oscillators, phases constant
        let phases = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let w = winding_numbers(&phases, 5, 3);
        assert_eq!(w, vec![0, 0, 0]);
    }

    #[test]
    fn one_full_rotation() {
        let n = 1;
        let steps = 100;
        let mut phases = Vec::with_capacity(steps * n);
        for i in 0..steps {
            // One full rotation over 100 steps
            phases.push((i as f64 / steps as f64) * TAU % TAU);
        }
        let w = winding_numbers(&phases, steps, n);
        // Almost one full rotation (0 to ~2π), floor gives 0
        // Need to go beyond 2π for winding=1
        assert_eq!(w[0], 0);
    }

    #[test]
    fn two_full_rotations() {
        let n = 1;
        let steps = 200;
        let dt = TAU * 2.1 / (steps as f64 - 1.0);
        let mut phases = Vec::with_capacity(steps * n);
        for i in 0..steps {
            phases.push((i as f64 * dt) % TAU);
        }
        let w = winding_numbers(&phases, steps, n);
        assert_eq!(w[0], 2);
    }

    #[test]
    fn negative_rotation() {
        let n = 1;
        let steps = 200;
        // ~1.1 negative rotations → floor(-1.1) = -2
        let dt = -TAU * 1.1 / (steps as f64 - 1.0);
        let mut phases = Vec::with_capacity(steps * n);
        let mut phase: f64 = 3.0;
        for _ in 0..steps {
            phases.push(phase.rem_euclid(TAU));
            phase += dt;
        }
        let w = winding_numbers(&phases, steps, n);
        assert!(w[0] < 0, "should be negative: {}", w[0]);
    }

    #[test]
    fn multiple_oscillators() {
        let n = 2;
        let steps = 100;
        let mut phases = Vec::with_capacity(steps * n);
        for i in 0..steps {
            // osc 0: no rotation, osc 1: ~1.5 rotations
            let t = i as f64 / (steps as f64 - 1.0);
            phases.push(1.0); // constant
            phases.push((t * TAU * 1.5) % TAU);
        }
        let w = winding_numbers(&phases, steps, n);
        assert_eq!(w[0], 0);
        assert_eq!(w[1], 1);
    }

    #[test]
    fn too_few_timesteps() {
        let w = winding_numbers(&[1.0, 2.0], 1, 2);
        assert_eq!(w, vec![0, 0]);
    }

    #[test]
    fn empty_input() {
        let w = winding_numbers(&[], 0, 0);
        assert!(w.is_empty());
    }

    #[test]
    fn size_mismatch_gives_zeros() {
        let w = winding_numbers(&[1.0, 2.0], 3, 2);
        assert_eq!(w, vec![0, 0]);
    }
}
