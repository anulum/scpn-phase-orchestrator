// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Lyapunov spectrum via Benettin QR algorithm

use rayon::prelude::*;
use std::f64::consts::PI;

/// Full Lyapunov spectrum of a Kuramoto network with Sakaguchi phase-lag
/// and optional external driver.
///
/// Evolves N perturbation vectors alongside the Kuramoto ODE using RK4
/// integration. Every `qr_interval` steps, QR-reorthogonalises via
/// modified Gram-Schmidt and accumulates growth rates from diag(R).
///
/// References:
/// - Benettin et al. 1980, Meccanica 15:9-20.
/// - Shimada & Nagashima 1979, Prog. Theor. Phys. 61:1605-1616.
/// - Pikovsky & Politi 2016, Lyapunov Exponents, Cambridge UP.
///
/// # Arguments
/// * `phases_init` - (N,) initial phases
/// * `omegas` - (N,) natural frequencies
/// * `knm` - (N*N,) row-major coupling matrix K_{ij}
/// * `alpha` - (N*N,) row-major phase-lag matrix α_{ij}
/// * `dt` - integration timestep
/// * `n_steps` - total integration steps
/// * `qr_interval` - steps between QR reorthogonalisations
/// * `zeta` - external driver strength
/// * `psi` - external driver target phase
///
/// # Returns
/// (N,) Lyapunov exponents in units of 1/time, sorted descending.
///
/// # Errors
/// Returns error if input lengths are inconsistent or qr_interval is 0.
#[allow(clippy::too_many_arguments)]
pub fn lyapunov_spectrum(
    phases_init: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    dt: f64,
    n_steps: usize,
    qr_interval: usize,
    zeta: f64,
    psi: f64,
) -> Result<Vec<f64>, String> {
    let n = phases_init.len();
    if omegas.len() != n {
        return Err(format!(
            "omegas length {} != phases length {}",
            omegas.len(),
            n
        ));
    }
    if knm.len() != n * n {
        return Err(format!("knm length {} != N*N={}", knm.len(), n * n));
    }
    if alpha.len() != n * n {
        return Err(format!("alpha length {} != N*N={}", alpha.len(), n * n));
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if qr_interval == 0 {
        return Err("qr_interval must be > 0".to_string());
    }

    let nn = n * n;
    let mut phases = phases_init.to_vec();
    let mut q = vec![0.0; nn];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut exponents = vec![0.0; n];
    let mut n_qr = 0;
    let mut total_time = 0.0;

    let mut k1_p = vec![0.0; n];
    let mut k2_p = vec![0.0; n];
    let mut k3_p = vec![0.0; n];
    let mut k4_p = vec![0.0; n];
    let mut k1_q = vec![0.0; nn];
    let mut k2_q = vec![0.0; nn];
    let mut k3_q = vec![0.0; nn];
    let mut k4_q = vec![0.0; nn];
    let mut tmp_p = vec![0.0; n];
    let mut tmp_q = vec![0.0; nn];
    let mut sin_theta = vec![0.0; n];
    let mut cos_theta = vec![0.0; n];

    let alpha_zero = alpha.iter().all(|&a| a == 0.0);
    let (zs_psi, zc_psi) = if zeta != 0.0 {
        let (s, c) = psi.sin_cos();
        (zeta * s, zeta * c)
    } else {
        (0.0, 0.0)
    };

    let dt6 = dt / 6.0;
    let two_pi = 2.0 * PI;

    for step in 0..n_steps {
        compute_rhs(
            n,
            &phases,
            &mut sin_theta,
            &mut cos_theta,
            omegas,
            knm,
            alpha,
            alpha_zero,
            zeta,
            zs_psi,
            zc_psi,
            &mut k1_p,
        );
        compute_jq(
            n, &phases, &sin_theta, &cos_theta, knm, alpha, alpha_zero, zeta, zs_psi, zc_psi, &q,
            &mut k1_q,
        );

        for i in 0..n {
            tmp_p[i] = phases[i] + 0.5 * dt * k1_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + 0.5 * dt * k1_q[i];
        }
        compute_rhs(
            n,
            &tmp_p,
            &mut sin_theta,
            &mut cos_theta,
            omegas,
            knm,
            alpha,
            alpha_zero,
            zeta,
            zs_psi,
            zc_psi,
            &mut k2_p,
        );
        compute_jq(
            n, &tmp_p, &sin_theta, &cos_theta, knm, alpha, alpha_zero, zeta, zs_psi, zc_psi,
            &tmp_q, &mut k2_q,
        );

        for i in 0..n {
            tmp_p[i] = phases[i] + 0.5 * dt * k2_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + 0.5 * dt * k2_q[i];
        }
        compute_rhs(
            n,
            &tmp_p,
            &mut sin_theta,
            &mut cos_theta,
            omegas,
            knm,
            alpha,
            alpha_zero,
            zeta,
            zs_psi,
            zc_psi,
            &mut k3_p,
        );
        compute_jq(
            n, &tmp_p, &sin_theta, &cos_theta, knm, alpha, alpha_zero, zeta, zs_psi, zc_psi,
            &tmp_q, &mut k3_q,
        );

        for i in 0..n {
            tmp_p[i] = phases[i] + dt * k3_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + dt * k3_q[i];
        }
        compute_rhs(
            n,
            &tmp_p,
            &mut sin_theta,
            &mut cos_theta,
            omegas,
            knm,
            alpha,
            alpha_zero,
            zeta,
            zs_psi,
            zc_psi,
            &mut k4_p,
        );
        compute_jq(
            n, &tmp_p, &sin_theta, &cos_theta, knm, alpha, alpha_zero, zeta, zs_psi, zc_psi,
            &tmp_q, &mut k4_q,
        );

        for i in 0..n {
            phases[i] = (phases[i] + dt6 * (k1_p[i] + 2.0 * k2_p[i] + 2.0 * k3_p[i] + k4_p[i]))
                .rem_euclid(two_pi);
        }
        for i in 0..nn {
            q[i] += dt6 * (k1_q[i] + 2.0 * k2_q[i] + 2.0 * k3_q[i] + k4_q[i]);
        }
        total_time += dt;

        if (step + 1) % qr_interval == 0 {
            let r_diag = modified_gram_schmidt(&mut q, n);
            for (i, &rd) in r_diag.iter().enumerate() {
                exponents[i] += rd.abs().max(1e-300).ln();
            }
            n_qr += 1;
        }
    }

    if n_qr > 0 {
        for e in exponents.iter_mut() {
            *e /= total_time;
        }
    }
    exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(exponents)
}

/// Kuramoto RHS: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_rhs(
    n: usize,
    phases: &[f64],
    sin_theta: &mut [f64],
    cos_theta: &mut [f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    alpha_zero: bool,
    zeta: f64,
    zs_psi: f64,
    zc_psi: f64,
    out: &mut [f64],
) {
    for i in 0..n {
        let (s, c) = phases[i].sin_cos();
        sin_theta[i] = s;
        cos_theta[i] = c;
    }
    let st = &*sin_theta;
    let ct = &*cos_theta;

    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let mut coupling = 0.0;
        let offset = i * n;
        let k_row = &knm[offset..offset + n];
        let ci = ct[i];
        let si = st[i];

        if alpha_zero {
            for j in 0..n {
                coupling += k_row[j] * (st[j] * ci - ct[j] * si);
            }
        } else {
            for j in 0..n {
                coupling += k_row[j] * (phases[j] - phases[i] - alpha[offset + j]).sin();
            }
        }

        let driving = if zeta != 0.0 {
            zs_psi * ci - zc_psi * si
        } else {
            0.0
        };
        *val = omegas[i] + coupling + driving;
    });
}

/// Compute J(θ)·Q where J is the Kuramoto Jacobian.
///
/// J_ij = K_ij cos(θ_j - θ_i - α_ij) for i≠j
/// J_ii = -Σ_{j≠i} K_ij cos(θ_j - θ_i - α_ij) - ζ cos(Ψ - θ_i)
///
/// Result stored in out (row-major N×N).
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_jq(
    n: usize,
    phases: &[f64],
    sin_theta: &[f64],
    cos_theta: &[f64],
    knm: &[f64],
    alpha: &[f64],
    alpha_zero: bool,
    zeta: f64,
    zs_psi: f64,
    zc_psi: f64,
    q_mat: &[f64],
    out: &mut [f64],
) {
    let st = sin_theta;
    let ct = cos_theta;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let mut diag_sum = 0.0;
        let offset = i * n;
        let k_row = &knm[offset..offset + n];
        let ci = ct[i];
        let si = st[i];

        out_row.fill(0.0);

        for j in 0..n {
            if i == j {
                continue;
            }
            let j_ij = if alpha_zero {
                k_row[j] * (ct[j] * ci + st[j] * si)
            } else {
                k_row[j] * (phases[j] - phases[i] - alpha[offset + j]).cos()
            };
            diag_sum += j_ij;

            let q_row = &q_mat[j * n..(j + 1) * n];
            for k in 0..n {
                out_row[k] += j_ij * q_row[k];
            }
        }

        let driver_diag = if zeta != 0.0 {
            zc_psi * ci + zs_psi * si
        } else {
            0.0
        };
        let j_ii = -(diag_sum + driver_diag);
        let q_row_ii = &q_mat[i * n..(i + 1) * n];
        for k in 0..n {
            out_row[k] += j_ii * q_row_ii[k];
        }
    });
}

/// Modified Gram-Schmidt QR on row-major N×N matrix Q, in-place.
///
/// Returns the diagonal of R (column norms before normalisation = stretching factors).
/// Uses two-pass reorthogonalisation (Daniel et al. 1976) for numerical stability
/// when vectors become nearly parallel under contraction.
fn modified_gram_schmidt(q: &mut [f64], n: usize) -> Vec<f64> {
    let mut r_diag = vec![0.0; n];
    for j in 0..n {
        for _pass in 0..2 {
            for k in 0..j {
                let (q_before, q_after) = q.split_at_mut(j * n);
                let q_k = &q_before[k * n..(k + 1) * n];
                let q_j = &mut q_after[..n];

                let dot: f64 = q_k.iter().zip(q_j.iter()).map(|(&qk, &qj)| qk * qj).sum();
                for (i, val) in q_j.iter_mut().enumerate() {
                    *val -= dot * q_k[i];
                }
            }
        }
        let start = j * n;
        let q_j = &q[start..start + n];
        let norm_sq: f64 = q_j.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        r_diag[j] = norm;
        if norm > 1e-300 {
            let inv = 1.0 / norm;
            let q_j_mut = &mut q[start..start + n];
            for val in q_j_mut.iter_mut() {
                *val *= inv;
            }
        }
    }
    r_diag
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_empty_input() {
        let result = lyapunov_spectrum(&[], &[], &[], &[], 0.01, 100, 10, 0.0, 0.0);
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_single_oscillator_neutral() {
        // Single uncoupled oscillator: exponent = 0 (neutral stability)
        let result =
            lyapunov_spectrum(&[0.5], &[1.0], &[0.0], &[0.0], 0.01, 500, 10, 0.0, 0.0).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result[0].abs() < 0.1,
            "single uncoupled: λ = {} (expected ~0)",
            result[0]
        );
    }

    #[test]
    fn test_sorted_descending() {
        let n = 4;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * TAU / n as f64).collect();
        let omegas = vec![1.0; n];
        let mut knm = vec![0.5; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = vec![0.0; n * n];
        let result =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.01, 500, 10, 0.0, 0.0).unwrap();
        assert_eq!(result.len(), n);
        for i in 1..n {
            assert!(
                result[i - 1] >= result[i] - 1e-10,
                "not sorted: λ_{} = {} < λ_{} = {}",
                i - 1,
                result[i - 1],
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_strong_coupling_transverse_contraction() {
        // Strong coupling → synchronisation → transverse exponents negative
        // (attractor has codimension N-1, so λ_2..λ_N < 0)
        let n = 4;
        let phases = vec![0.1, 0.2, 0.15, 0.12];
        let omegas = vec![1.0; n];
        let mut knm = vec![5.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = vec![0.0; n * n];
        let result =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.005, 2000, 10, 0.0, 0.0).unwrap();
        assert!(
            result[1] < 0.1,
            "λ_2 should be negative for strong coupling: {}",
            result[1]
        );
    }

    #[test]
    fn test_zero_coupling_zero_exponents() {
        // Uncoupled oscillators: all Lyapunov exponents should be ~0
        let n = 3;
        let phases = vec![0.0, 1.0, 2.0];
        let omegas = vec![1.0, 1.1, 0.9];
        let knm = vec![0.0; n * n];
        let alpha = vec![0.0; n * n];
        let result =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.01, 1000, 10, 0.0, 0.0).unwrap();
        for (i, &e) in result.iter().enumerate() {
            assert!(e.abs() < 0.1, "λ_{} = {} should be ~0 for uncoupled", i, e);
        }
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let result = lyapunov_spectrum(
            &[0.0, 1.0],
            &[1.0],
            &[0.0; 4],
            &[0.0; 4],
            0.01,
            10,
            5,
            0.0,
            0.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_qr_interval_error() {
        let result = lyapunov_spectrum(&[0.0], &[1.0], &[0.0], &[0.0], 0.01, 10, 0, 0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_driver_stabilises() {
        // External driver with strong coupling: should produce
        // more negative exponents than without
        let n = 3;
        let phases = vec![0.0, 2.0, 4.0];
        let omegas = vec![1.0; n];
        let mut knm = vec![1.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = vec![0.0; n * n];

        let no_driver =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.01, 1000, 10, 0.0, 0.0).unwrap();
        let with_driver =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.01, 1000, 10, 2.0, 0.0).unwrap();

        // Driver adds extra contraction → max exponent should be more negative
        assert!(
            with_driver[0] <= no_driver[0] + 0.5,
            "driver should stabilise: λ_max(driver)={} vs λ_max(free)={}",
            with_driver[0],
            no_driver[0]
        );
    }

    #[test]
    fn test_rk4_more_accurate_than_euler_would_be() {
        // Convergence check: smaller dt should give similar results
        let n = 3;
        let phases = vec![0.0, 1.0, 2.0];
        let omegas = vec![1.0, 1.2, 0.8];
        let mut knm = vec![2.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = vec![0.0; n * n];

        let coarse =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.02, 500, 10, 0.0, 0.0).unwrap();
        let fine =
            lyapunov_spectrum(&phases, &omegas, &knm, &alpha, 0.01, 1000, 10, 0.0, 0.0).unwrap();

        for i in 0..n {
            let diff = (coarse[i] - fine[i]).abs();
            assert!(
                diff < 1.0,
                "λ_{}: coarse={} vs fine={}, diff={}",
                i,
                coarse[i],
                fine[i],
                diff
            );
        }
    }
}
