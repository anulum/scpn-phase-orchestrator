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

use rayon::prelude::*;
use std::f64::consts::TAU;

/// Run torus geometric integration for `n_steps`.
///
/// Uses the exponential map on S¹: z → z · exp(i·ω_eff·dt).
/// This is mathematically equivalent to Euler but avoids
/// numerical artefacts at the 0/2π boundary.
#[must_use]
#[allow(clippy::too_many_arguments)]
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
        let (s, c) = phases[i].sin_cos();
        z_re[i] = c;
        z_im[i] = s;
    }

    let alpha_zero = alpha.iter().all(|&a| a == 0.0);
    let (zs_psi, zc_psi) = if zeta != 0.0 {
        let (s, c) = psi.sin_cos();
        (zeta * s, zeta * c)
    } else {
        (0.0, 0.0)
    };

    for _ in 0..n_steps {
        let re_slice = &*z_re;
        let im_slice = &*z_im;

        let res: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut coupling = 0.0;
                let offset = i * n;
                let k_row = &knm[offset..offset + n];

                if alpha_zero {
                    let mut k_iter = k_row.chunks_exact(8);
                    let mut re_iter = re_slice.chunks_exact(8);
                    let mut im_iter = im_slice.chunks_exact(8);
                    let mut acc = 0.0;
                    for ((kc, rec), imc) in
                        k_iter.by_ref().zip(re_iter.by_ref()).zip(im_iter.by_ref())
                    {
                        // sin(tj - ti) = sj*ci - cj*si
                        acc += kc[0] * (imc[0] * re_slice[i] - rec[0] * im_slice[i]);
                        acc += kc[1] * (imc[1] * re_slice[i] - rec[1] * im_slice[i]);
                        acc += kc[2] * (imc[2] * re_slice[i] - rec[2] * im_slice[i]);
                        acc += kc[3] * (imc[3] * re_slice[i] - rec[3] * im_slice[i]);
                        acc += kc[4] * (imc[4] * re_slice[i] - rec[4] * im_slice[i]);
                        acc += kc[5] * (imc[5] * re_slice[i] - rec[5] * im_slice[i]);
                        acc += kc[6] * (imc[6] * re_slice[i] - rec[6] * im_slice[i]);
                        acc += kc[7] * (imc[7] * re_slice[i] - rec[7] * im_slice[i]);
                    }
                    coupling = acc;
                    for ((&kj, &rej), &imj) in k_iter
                        .remainder()
                        .iter()
                        .zip(re_iter.remainder())
                        .zip(im_iter.remainder())
                    {
                        coupling += kj * (imj * re_slice[i] - rej * im_slice[i]);
                    }
                } else {
                    for j in 0..n {
                        let tj = im_slice[j].atan2(re_slice[j]);
                        let ti = im_slice[i].atan2(re_slice[i]);
                        coupling += knm[offset + j] * (tj - ti - alpha[offset + j]).sin();
                    }
                }

                let mut omega_eff = omegas[i] + coupling;
                if zeta != 0.0 {
                    omega_eff += zs_psi * re_slice[i] - zc_psi * im_slice[i];
                }

                let angle = omega_eff * dt;
                let (sin_a, cos_a) = angle.sin_cos();
                let next_re = re_slice[i] * cos_a - im_slice[i] * sin_a;
                let next_im = re_slice[i] * sin_a + im_slice[i] * cos_a;
                let norm = (next_re * next_re + next_im * next_im).sqrt();
                if norm > 0.0 {
                    (next_re / norm, next_im / norm)
                } else {
                    (next_re, next_im)
                }
            })
            .collect();

        for i in 0..n {
            z_re[i] = res[i].0;
            z_im[i] = res[i].1;
        }
    }

    (0..n)
        .map(|i| z_im[i].atan2(z_re[i]).rem_euclid(TAU))
        .collect()
}

/// Exponential map on S¹: z_i → z_i · exp(i·ω_i·dt), then renormalise.

/// Standard Kuramoto derivative: ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(ψ - θ_i).

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
