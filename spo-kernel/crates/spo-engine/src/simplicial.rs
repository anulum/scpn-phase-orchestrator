// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Simplicial (higher-order) Kuramoto coupling

//! Kuramoto with pairwise + 3-body simplicial coupling.
//!
//! dθ_i/dt = ω_i
//!           + Σ_j K_ij sin(θ_j - θ_i - α_ij)
//!           + (σ₂/N²) Σ_{j,k} sin(θ_j + θ_k - 2θ_i)
//!
//! The 3-body term uses the identity:
//!   Σ_{j,k} sin((θ_j - θ_i) + (θ_k - θ_i))
//!     = 2 · S_i · C_i
//! where S_i = Σ_j sin(θ_j - θ_i), C_i = Σ_j cos(θ_j - θ_i).
//!
//! Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.

use std::f64::consts::TAU;

/// Run simplicial Kuramoto integration (Euler) for `n_steps`.
///
/// # Arguments
/// * `phases` – initial phases, length `n`
/// * `omegas` – natural frequencies, length `n`
/// * `knm` – row-major coupling matrix, length `n*n`
/// * `alpha` – row-major phase-lag matrix, length `n*n`
/// * `zeta` – external drive strength
/// * `psi` – external drive phase
/// * `sigma2` – 3-body simplicial coupling strength
/// * `dt` – time step
/// * `n_steps` – number of Euler steps
///
/// # Returns
/// Final phases, length `n`.
#[must_use]
pub fn simplicial_run(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    sigma2: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let n = phases.len();
    let mut p = phases.to_vec();

    for _ in 0..n_steps {
        let deriv = simplicial_derivative(&p, omegas, knm, alpha, n, zeta, psi, sigma2);
        for i in 0..n {
            p[i] = (p[i] + dt * deriv[i]).rem_euclid(TAU);
        }
    }

    p
}

/// Compute the full derivative for one time step.
fn simplicial_derivative(
    theta: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
    sigma2: f64,
) -> Vec<f64> {
    let mut result = vec![0.0; n];

    for i in 0..n {
        let mut pairwise = 0.0;
        for j in 0..n {
            let diff = theta[j] - theta[i] - alpha[i * n + j];
            pairwise += knm[i * n + j] * diff.sin();
        }
        result[i] = omegas[i] + pairwise;
    }

    // 3-body simplicial term
    if sigma2 != 0.0 && n >= 3 {
        let inv_n2 = sigma2 / (n as f64 * n as f64);
        for i in 0..n {
            let mut s_sum = 0.0;
            let mut c_sum = 0.0;
            for j in 0..n {
                let d = theta[j] - theta[i];
                s_sum += d.sin();
                c_sum += d.cos();
            }
            // Σ_{j,k} sin((θ_j - θ_i) + (θ_k - θ_i))
            // = (Σ sin)(Σ cos) + (Σ cos)(Σ sin) = 2·S·C
            result[i] += 2.0 * s_sum * c_sum * inv_n2;
        }
    }

    // External drive
    if zeta != 0.0 {
        for i in 0..n {
            result[i] += zeta * (psi - theta[i]).sin();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_identical_phases_no_drift() {
        // All oscillators at same phase, same ω → pairwise coupling = 0
        let n = 5;
        let phases = vec![1.0; n];
        let omegas = vec![1.0; n];
        let knm = vec![0.5; n * n]; // uniform coupling
        let alpha = vec![0.0; n * n];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.01, 100);
        // All should advance by ω·dt·n_steps = 1.0 (mod 2π)
        let expected = (1.0_f64 + 1.0 * 0.01 * 100.0).rem_euclid(TAU);
        for p in &result {
            assert!((p - expected).abs() < 1e-6, "p={p}, expected={expected}");
        }
    }

    #[test]
    fn test_zero_coupling_free_rotation() {
        let n = 3;
        let phases = vec![0.0, PI / 2.0, PI];
        let omegas = vec![1.0, 2.0, 3.0];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.01, 100);
        // Free rotation: θ_i = θ_i0 + ω_i·dt·n_steps
        for i in 0..n {
            let expected = (phases[i] + omegas[i] * 0.01 * 100.0).rem_euclid(TAU);
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "i={i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_simplicial_term_effect() {
        // With σ₂ > 0, 3-body coupling should change evolution
        let n = 4;
        let phases = vec![0.0, 0.5, 1.0, 1.5];
        let omegas = vec![1.0; n];
        let knm = vec![0.0; n * n]; // no pairwise
        let alpha = vec![0.0; n * n];

        let without = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.01, 50);
        let with = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 2.0, 0.01, 50);

        // Results should differ
        let mut diff_sum = 0.0;
        for i in 0..n {
            diff_sum += (without[i] - with[i]).abs();
        }
        assert!(
            diff_sum > 1e-4,
            "simplicial term had no effect: diff_sum={diff_sum}"
        );
    }

    #[test]
    fn test_external_drive() {
        let n = 3;
        let phases = vec![0.0; n];
        let omegas = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];

        let result = simplicial_run(
            &phases,
            &omegas,
            &knm,
            &alpha,
            1.0,
            PI / 2.0,
            0.0,
            0.01,
            100,
        );
        // Drive toward psi=π/2 with zeta=1
        for p in &result {
            assert!(*p > 0.0, "should move toward psi");
        }
    }

    #[test]
    fn test_synchronisation_with_pairwise() {
        // Strong pairwise coupling should synchronise
        let n = 5;
        let phases = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let omegas = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 2.0;
                }
            }
        }
        let alpha = vec![0.0; n * n];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.01, 1000);

        // Check phase coherence: R should be high
        let mut sx = 0.0;
        let mut cx = 0.0;
        for p in &result {
            sx += p.sin();
            cx += p.cos();
        }
        let r = (sx * sx + cx * cx).sqrt() / n as f64;
        assert!(r > 0.8, "R={r} should be > 0.8 with strong coupling");
    }

    #[test]
    fn test_phases_in_range() {
        let n = 4;
        let phases = vec![0.1, 2.0, 4.0, 5.5];
        let omegas = vec![3.0, -1.0, 2.0, 7.0];
        let knm = vec![0.3; n * n];
        let alpha = vec![0.0; n * n];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.5, 1.0, 1.0, 0.01, 500);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} out of [0, 2π)");
        }
    }

    #[test]
    fn test_single_step() {
        // Verify one step matches manual calculation
        let phases = vec![0.0, PI];
        let omegas = vec![0.0, 0.0];
        let knm = vec![0.0, 1.0, 1.0, 0.0];
        let alpha = vec![0.0; 4];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.1, 1);
        // θ₀: coupling = sin(π - 0) = 0 → θ₀ stays 0
        // θ₁: coupling = sin(0 - π) = 0 → θ₁ stays π
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - PI).abs() < 1e-10);
    }

    #[test]
    fn test_simplicial_with_pairwise_combined() {
        let n = 4;
        let phases = vec![0.0, 0.3, 0.6, 0.9];
        let omegas = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 1.0;
                }
            }
        }
        let alpha = vec![0.0; n * n];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 1.5, 0.005, 500);
        // Both pairwise and 3-body active → phases should be valid
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU);
        }
    }

    #[test]
    fn test_zero_steps_returns_input() {
        let phases = vec![1.0, 2.0, 3.0];
        let omegas = vec![1.0; 3];
        let knm = vec![0.5; 9];
        let alpha = vec![0.0; 9];
        let result = simplicial_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.0, 0.01, 0);
        for (a, b) in result.iter().zip(phases.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_alpha_phase_lag() {
        // Phase lag α should shift the coupling null point
        let n = 2;
        let phases = vec![0.0, 0.5];
        let omegas = vec![0.0; n];
        let knm = vec![0.0, 1.0, 1.0, 0.0];
        let alpha_zero = vec![0.0; 4];
        let alpha_shift = vec![0.0, 0.3, 0.3, 0.0];

        let r1 = simplicial_run(&phases, &omegas, &knm, &alpha_zero, 0.0, 0.0, 0.0, 0.01, 50);
        let r2 = simplicial_run(
            &phases,
            &omegas,
            &knm,
            &alpha_shift,
            0.0,
            0.0,
            0.0,
            0.01,
            50,
        );

        let diff = (r1[0] - r2[0]).abs() + (r1[1] - r2[1]).abs();
        assert!(diff > 1e-4, "alpha should shift evolution: diff={diff}");
    }
}
