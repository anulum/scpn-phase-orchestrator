// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Thermodynamic entropy production rate

//! Entropy production rate for overdamped Kuramoto systems.
//!
//! Measures total dissipation: Sigma_i (dtheta_i/dt)^2 * dt.
//! Zero at frequency-locked fixed points; positive otherwise.

/// Compute entropy production rate.
///
/// # Arguments
/// * `phases` — (N,) oscillator phases
/// * `omegas` — (N,) natural frequencies
/// * `knm` — (N*N,) row-major coupling matrix
/// * `alpha` — global coupling scale factor
/// * `dt` — timestep
///
/// # Returns
/// Scalar entropy production rate (non-negative).
#[must_use]
pub fn entropy_production_rate(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: f64,
    dt: f64,
) -> f64 {
    let n = phases.len();
    if n == 0 || dt <= 0.0 || omegas.len() != n || knm.len() != n * n {
        return 0.0;
    }

    let inv_n = alpha / n as f64;
    let mut total = 0.0;

    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            coupling += knm[i * n + j] * (phases[j] - phases[i]).sin();
        }
        let dtheta = omegas[i] + inv_n * coupling;
        total += dtheta * dtheta;
    }

    total * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn zero_at_locked_state() {
        // All oscillators at same phase, same frequency, zero coupling
        let n = 4;
        let phases = vec![1.0; n];
        let omegas = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let ep = entropy_production_rate(&phases, &omegas, &knm, 1.0, 0.01);
        assert!(ep.abs() < 1e-15, "expected zero, got {ep}");
    }

    #[test]
    fn positive_for_free_running() {
        let n = 4;
        let phases = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let omegas = vec![1.0, 1.5, 0.8, 1.2];
        let knm = vec![0.0; n * n];
        let ep = entropy_production_rate(&phases, &omegas, &knm, 0.0, 0.01);
        assert!(ep > 0.0, "free-running should have positive EP");
    }

    #[test]
    fn scales_with_dt() {
        let n = 2;
        let phases = vec![0.0, 1.0];
        let omegas = vec![1.0, 2.0];
        let knm = vec![0.0; 4];
        let ep1 = entropy_production_rate(&phases, &omegas, &knm, 0.0, 0.01);
        let ep2 = entropy_production_rate(&phases, &omegas, &knm, 0.0, 0.02);
        assert!(
            (ep2 / ep1 - 2.0).abs() < 1e-10,
            "EP should scale linearly with dt"
        );
    }

    #[test]
    fn coupling_reduces_ep_at_sync() {
        let n = 4;
        // Near-sync phases
        let phases = vec![0.0, 0.01, 0.02, 0.03];
        let omegas = vec![1.0; n];
        let mut knm = vec![5.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }

        let ep_coupled = entropy_production_rate(&phases, &omegas, &knm, 1.0, 0.01);
        let ep_uncoupled = entropy_production_rate(&phases, &omegas, &vec![0.0; n * n], 1.0, 0.01);
        // Both should be positive, but the uncoupled should differ
        assert!(ep_coupled > 0.0);
        assert!(ep_uncoupled > 0.0);
    }

    #[test]
    fn empty_returns_zero() {
        assert_eq!(entropy_production_rate(&[], &[], &[], 1.0, 0.01), 0.0);
    }

    #[test]
    fn negative_dt_returns_zero() {
        let ep = entropy_production_rate(&[0.0], &[1.0], &[0.0], 1.0, -0.01);
        assert_eq!(ep, 0.0);
    }

    #[test]
    fn dimension_mismatch_returns_zero() {
        let ep = entropy_production_rate(&[0.0, 1.0], &[1.0], &[0.0; 4], 1.0, 0.01);
        assert_eq!(ep, 0.0);
    }
}
