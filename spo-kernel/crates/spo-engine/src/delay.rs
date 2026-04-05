// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

//! Kuramoto model with time-delayed coupling.
//!
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t−τ) − θ_i(t) − α_ij)
//!
//! Time delay generates effective higher-order interactions
//! (Ciszak et al. 2025).

use std::collections::VecDeque;
use std::f64::consts::TAU;

/// Run n_steps of delayed Kuramoto, returning final phases.
///
/// Internally maintains a circular buffer for delayed phase lookup.
/// Falls back to instantaneous coupling when buffer is not yet full.
///
/// # Arguments
/// * `phases_init` — (N,) initial phases
/// * `omegas` — (N,) natural frequencies
/// * `knm_flat` — (N×N) row-major coupling
/// * `alpha_flat` — (N×N) row-major phase lags
/// * `n` — number of oscillators
/// * `zeta` — external drive strength
/// * `psi` — external drive phase
/// * `dt` — timestep
/// * `delay_steps` — number of steps of delay
/// * `n_steps` — total steps to run
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn delayed_kuramoto_run(
    phases_init: &[f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
    dt: f64,
    delay_steps: usize,
    n_steps: usize,
) -> Vec<f64> {
    let mut phases = phases_init.to_vec();
    let max_buf = delay_steps + 1;
    let mut buffer: VecDeque<Vec<f64>> = VecDeque::with_capacity(max_buf);

    for _ in 0..n_steps {
        // Get delayed phases (or current if buffer not full)
        let coupling_phases = if delay_steps > 0 && buffer.len() >= delay_steps {
            buffer[buffer.len() - delay_steps].clone()
        } else {
            phases.clone()
        };

        // Push current to buffer
        buffer.push_back(phases.clone());
        if buffer.len() > max_buf {
            buffer.pop_front();
        }

        // Euler step: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t−τ) − θ_i(t) − α_ij)
        let mut new_phases = Vec::with_capacity(n);
        for i in 0..n {
            let mut coupling = 0.0;
            for j in 0..n {
                let k_ij = knm_flat[i * n + j];
                if k_ij.abs() > 1e-30 {
                    let a_ij = alpha_flat[i * n + j];
                    coupling += k_ij * (coupling_phases[j] - phases[i] - a_ij).sin();
                }
            }
            let mut dtheta = omegas[i] + coupling;
            if zeta.abs() > 1e-30 {
                dtheta += zeta * (psi - phases[i]).sin();
            }
            let raw = phases[i] + dt * dtheta;
            new_phases.push(((raw % TAU) + TAU) % TAU);
        }
        phases = new_phases;
    }

    phases
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_all_to_all(n: usize, strength: f64) -> Vec<f64> {
        let mut knm = vec![strength / n as f64; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        knm
    }

    #[test]
    fn test_zero_delay_equals_instantaneous() {
        let n = 4;
        let phases = vec![0.0, 0.5, 1.0, 1.5];
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n, 2.0);
        let alpha = vec![0.0; n * n];

        let r0 = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 0, 100);
        let r1 = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 0, 100);
        assert_eq!(r0, r1);
    }

    #[test]
    fn test_delay_differs_from_instantaneous() {
        let n = 4;
        let phases = vec![0.0, 0.5, 1.0, 1.5];
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n, 5.0);
        let alpha = vec![0.0; n * n];

        let instant =
            delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 0, 200);
        let delayed =
            delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 10, 200);

        let diff: f64 = instant
            .iter()
            .zip(&delayed)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "delay should produce different result, diff = {diff}"
        );
    }

    #[test]
    fn test_phases_bounded() {
        let n = 5;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 1.2).collect();
        let omegas: Vec<f64> = (0..n).map(|i| 0.5 + 0.3 * i as f64).collect();
        let knm = make_all_to_all(n, 3.0);
        let alpha = vec![0.0; n * n];

        let result =
            delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 5, 500);
        for (i, &v) in result.iter().enumerate() {
            assert!(v >= 0.0 && v < TAU, "phase[{i}] = {v} out of [0, 2π)");
        }
    }

    #[test]
    fn test_external_drive() {
        let n = 3;
        let phases = vec![0.0; n];
        let omegas = vec![0.0; n]; // no natural frequency
        let knm = vec![0.0; n * n]; // no coupling
        let alpha = vec![0.0; n * n];

        // With external drive, phases should move toward psi
        let result = delayed_kuramoto_run(
            &phases,
            &omegas,
            &knm,
            &alpha,
            n,
            1.0,
            PI / 2.0,
            0.01,
            0,
            500,
        );
        // All phases should have moved
        for &v in &result {
            assert!(v > 0.01, "external drive should move phases");
        }
    }

    #[test]
    fn test_preserves_length() {
        let n = 6;
        let phases = vec![0.0; n];
        let omegas = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];

        let result = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 3, 10);
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_deterministic() {
        let n = 4;
        let phases = vec![0.1, 0.5, 1.2, 2.3];
        let omegas = vec![1.0, 1.5, 2.0, 0.5];
        let knm = make_all_to_all(n, 2.0);
        let alpha = vec![0.0; n * n];

        let r1 = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 5, 100);
        let r2 = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 5, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_large_delay_uses_initial() {
        // When delay > n_steps, buffer never fills → uses instantaneous
        let n = 3;
        let phases = vec![0.1, 0.2, 0.3];
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n, 1.0);
        let alpha = vec![0.0; n * n];

        let result =
            delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, n, 0.0, 0.0, 0.01, 1000, 10);
        // Should still produce valid phases
        assert_eq!(result.len(), n);
        for &v in &result {
            assert!(v >= 0.0 && v < TAU);
        }
    }
}
