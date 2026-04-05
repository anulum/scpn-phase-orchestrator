// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Torus-preserving geometric integrator

//! Geometric integrator on the N-torus T^N = (S¹)^N.
//!
//! Represents each phase as a unit complex number z_i = exp(iθ_i),
//! computes the derivative in the tangent space, and maps back via
//! the exponential map. This avoids mod 2π discontinuity artefacts.
//!
//! Symplectic Euler on T^N:
//!   z_i(t+dt) = z_i(t) · exp(i · ω_eff_i · dt)

use std::f64::consts::TAU;

/// Run torus geometric integration for `n_steps`.
///
/// Uses the exponential map on S¹: z → z · exp(i·ω_eff·dt).
/// This is mathematically equivalent to Euler but avoids
/// numerical artefacts at the 0/2π boundary.
#[must_use]
pub fn torus_run(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let n = phases.len();
    let mut z_re = vec![0.0; n];
    let mut z_im = vec![0.0; n];
    for i in 0..n {
        z_re[i] = phases[i].cos();
        z_im[i] = phases[i].sin();
    }

    for _ in 0..n_steps {
        let theta: Vec<f64> = (0..n).map(|i| z_im[i].atan2(z_re[i])).collect();
        let omega_eff = kuramoto_derivative(&theta, omegas, knm, alpha, n, zeta, psi);
        exp_map_step(&mut z_re, &mut z_im, &omega_eff, dt, n);
    }

    (0..n).map(|i| z_im[i].atan2(z_re[i]).rem_euclid(TAU)).collect()
}

/// Exponential map on S¹: z_i → z_i · exp(i·ω_i·dt), then renormalise.
fn exp_map_step(z_re: &mut [f64], z_im: &mut [f64], omega: &[f64], dt: f64, n: usize) {
    for i in 0..n {
        let angle = omega[i] * dt;
        let (sin_a, cos_a) = angle.sin_cos();
        let re = z_re[i];
        let im = z_im[i];
        z_re[i] = re * cos_a - im * sin_a;
        z_im[i] = re * sin_a + im * cos_a;
        let norm = (z_re[i] * z_re[i] + z_im[i] * z_im[i]).sqrt();
        if norm > 0.0 { z_re[i] /= norm; z_im[i] /= norm; }
    }
}

/// Standard Kuramoto derivative: ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(ψ - θ_i).
fn kuramoto_derivative(
    theta: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            let diff = theta[j] - theta[i] - alpha[i * n + j];
            coupling += knm[i * n + j] * diff.sin();
        }
        result[i] = omegas[i] + coupling;
        if zeta != 0.0 {
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
    fn test_free_rotation() {
        let n = 3;
        let phases = vec![0.0, PI / 2.0, PI];
        let omegas = vec![1.0, 2.0, 3.0];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = torus_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100);
        for i in 0..n {
            let expected = (phases[i] + omegas[i]).rem_euclid(TAU);
            assert!(
                (result[i] - expected).abs() < 0.01,
                "i={i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_phases_in_range() {
        let n = 4;
        let phases = vec![0.1, 2.0, 4.0, 5.5];
        let omegas = vec![3.0, -1.0, 2.0, 7.0];
        let knm = vec![0.3; n * n];
        let alpha = vec![0.0; n * n];
        let result = torus_run(&phases, &omegas, &knm, &alpha, 0.5, 1.0, 0.01, 500);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} out of [0, 2π)");
        }
    }

    #[test]
    fn test_synchronisation() {
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
        let result = torus_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 1000);
        let mut sx = 0.0;
        let mut cx = 0.0;
        for p in &result {
            sx += p.sin();
            cx += p.cos();
        }
        let r = (sx * sx + cx * cx).sqrt() / n as f64;
        assert!(r > 0.8, "R={r} should be > 0.8");
    }

    #[test]
    fn test_zero_steps() {
        let phases = vec![1.0, 2.0, 3.0];
        let knm = vec![0.0; 9];
        let alpha = vec![0.0; 9];
        let result = torus_run(&phases, &[0.0; 3], &knm, &alpha, 0.0, 0.0, 0.01, 0);
        for (a, b) in result.iter().zip(phases.iter()) {
            let expected = b.rem_euclid(TAU);
            assert!((a - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_unit_circle_preservation() {
        // After many steps, phases should still be valid angles
        let n = 3;
        let phases = vec![0.1, 3.0, 5.9];
        let omegas = vec![10.0, -5.0, 20.0];
        let knm = vec![1.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = torus_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 10000);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} escaped [0, 2π)");
        }
    }

    #[test]
    fn test_external_drive() {
        let n = 2;
        let phases = vec![0.0, 0.0];
        let omegas = vec![0.0, 0.0];
        let knm = vec![0.0; 4];
        let alpha = vec![0.0; 4];
        let result = torus_run(&phases, &omegas, &knm, &alpha, 1.0, PI / 2.0, 0.01, 100);
        for p in &result {
            assert!(*p > 0.0, "should move toward psi=π/2");
        }
    }

    #[test]
    fn test_identical_phases_same_omega() {
        let n = 4;
        let phases = vec![1.0; n];
        let omegas = vec![2.0; n];
        let knm = vec![1.0; n * n];
        let alpha = vec![0.0; n * n];
        let result = torus_run(&phases, &omegas, &knm, &alpha, 0.0, 0.0, 0.01, 100);
        // All should remain identical
        let spread = result.iter().map(|p| *p).fold(f64::NEG_INFINITY, f64::max)
            - result.iter().map(|p| *p).fold(f64::INFINITY, f64::min);
        assert!(spread < 1e-6, "spread={spread}, phases should stay equal");
    }
}
