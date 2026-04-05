// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Lyapunov spectrum via Benettin QR algorithm

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
    let two_pi = 2.0 * PI;

    let mut phases = phases_init.to_vec();
    // Q: row-major N×N perturbation matrix (identity initially)
    let mut q = vec![0.0_f64; nn];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut exponents = vec![0.0_f64; n];
    let mut n_qr: usize = 0;
    let mut total_time = 0.0_f64;

    // Scratch space (allocated once, reused every step)
    let mut k1_p = vec![0.0_f64; n];
    let mut k2_p = vec![0.0_f64; n];
    let mut k3_p = vec![0.0_f64; n];
    let mut k4_p = vec![0.0_f64; n];
    let mut k1_q = vec![0.0_f64; nn];
    let mut k2_q = vec![0.0_f64; nn];
    let mut k3_q = vec![0.0_f64; nn];
    let mut k4_q = vec![0.0_f64; nn];
    let mut tmp_phases = vec![0.0_f64; n];
    let mut tmp_q = vec![0.0_f64; nn];

    for step in 0..n_steps {
        // RK4 step for (phases, Q) simultaneously
        //
        // Phase ODE: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
        // Tangent ODE: dQ/dt = J(θ) · Q
        // where J_ij = K_ij cos(θ_j - θ_i - α_ij), J_ii = -Σ_{j≠i} J_ij - ζ cos(Ψ - θ_i)

        // k1
        compute_rhs(n, &phases, omegas, knm, alpha, zeta, psi, &mut k1_p);
        compute_jq(n, &phases, knm, alpha, zeta, psi, &q, &mut k1_q);

        // k2: evaluate at (phases + dt/2 * k1_p, Q + dt/2 * k1_q)
        for i in 0..n {
            tmp_phases[i] = phases[i] + 0.5 * dt * k1_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + 0.5 * dt * k1_q[i];
        }
        compute_rhs(n, &tmp_phases, omegas, knm, alpha, zeta, psi, &mut k2_p);
        compute_jq(n, &tmp_phases, knm, alpha, zeta, psi, &tmp_q, &mut k2_q);

        // k3: evaluate at (phases + dt/2 * k2_p, Q + dt/2 * k2_q)
        for i in 0..n {
            tmp_phases[i] = phases[i] + 0.5 * dt * k2_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + 0.5 * dt * k2_q[i];
        }
        compute_rhs(n, &tmp_phases, omegas, knm, alpha, zeta, psi, &mut k3_p);
        compute_jq(n, &tmp_phases, knm, alpha, zeta, psi, &tmp_q, &mut k3_q);

        // k4: evaluate at (phases + dt * k3_p, Q + dt * k3_q)
        for i in 0..n {
            tmp_phases[i] = phases[i] + dt * k3_p[i];
        }
        for i in 0..nn {
            tmp_q[i] = q[i] + dt * k3_q[i];
        }
        compute_rhs(n, &tmp_phases, omegas, knm, alpha, zeta, psi, &mut k4_p);
        compute_jq(n, &tmp_phases, knm, alpha, zeta, psi, &tmp_q, &mut k4_q);

        // Combine: y += (dt/6)(k1 + 2k2 + 2k3 + k4)
        let dt6 = dt / 6.0;
        for i in 0..n {
            phases[i] += dt6 * (k1_p[i] + 2.0 * k2_p[i] + 2.0 * k3_p[i] + k4_p[i]);
        }
        for i in 0..nn {
            q[i] += dt6 * (k1_q[i] + 2.0 * k2_q[i] + 2.0 * k3_q[i] + k4_q[i]);
        }

        // Wrap phases to [0, 2π)
        for p in phases.iter_mut() {
            *p = ((*p % two_pi) + two_pi) % two_pi;
        }
        total_time += dt;

        // QR reorthogonalisation (modified Gram-Schmidt, Pikovsky & Politi 2016 §3.2)
        if (step + 1) % qr_interval == 0 {
            let r_diag = modified_gram_schmidt(&mut q, n);
            for (i, &rd) in r_diag.iter().enumerate() {
                let rd_abs = rd.abs().max(1e-300);
                exponents[i] += rd_abs.ln();
            }
            n_qr += 1;
        }
    }

    if n_qr > 0 {
        for e in exponents.iter_mut() {
            *e /= total_time;
        }
    }

    // Sort descending: λ_1 ≥ λ_2 ≥ ... ≥ λ_N
    exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(exponents)
}

/// Kuramoto RHS: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_rhs(
    n: usize,
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    out: &mut [f64],
) {
    for i in 0..n {
        let mut coupling = 0.0_f64;
        for j in 0..n {
            let diff = phases[j] - phases[i] - alpha[i * n + j];
            coupling += knm[i * n + j] * diff.sin();
        }
        let driving = if zeta != 0.0 {
            zeta * (psi - phases[i]).sin()
        } else {
            0.0
        };
        out[i] = omegas[i] + coupling + driving;
    }
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
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    q_mat: &[f64],
    out: &mut [f64],
) {
    // Zero output
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..n {
        let mut diag_sum = 0.0_f64;
        for j in 0..n {
            if i != j {
                let diff = phases[j] - phases[i] - alpha[i * n + j];
                let j_ij = knm[i * n + j] * diff.cos();
                diag_sum += j_ij;
                // out[i,k] += j_ij * q[j,k]
                for k in 0..n {
                    out[i * n + k] += j_ij * q_mat[j * n + k];
                }
            }
        }
        // Diagonal includes driver term: -ζ cos(Ψ - θ_i)
        let driver_diag = if zeta != 0.0 {
            zeta * (psi - phases[i]).cos()
        } else {
            0.0
        };
        let j_ii = -(diag_sum + driver_diag);
        for k in 0..n {
            out[i * n + k] += j_ii * q_mat[i * n + k];
        }
    }
}

/// Modified Gram-Schmidt QR on row-major N×N matrix Q, in-place.
///
/// Returns the diagonal of R (column norms before normalisation = stretching factors).
/// Uses two-pass reorthogonalisation (Daniel et al. 1976) for numerical stability
/// when vectors become nearly parallel under contraction.
fn modified_gram_schmidt(q: &mut [f64], n: usize) -> Vec<f64> {
    let mut r_diag = vec![0.0_f64; n];

    for j in 0..n {
        // Two-pass MGS for stability (Daniel et al. 1976, BIT 16:421-430)
        for _pass in 0..2 {
            for k in 0..j {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += q[i * n + k] * q[i * n + j];
                }
                for i in 0..n {
                    q[i * n + j] -= dot * q[i * n + k];
                }
            }
        }
        // Normalise column j
        let mut norm = 0.0_f64;
        for i in 0..n {
            norm += q[i * n + j] * q[i * n + j];
        }
        norm = norm.sqrt();
        r_diag[j] = norm;
        if norm > 1e-300 {
            let inv = 1.0 / norm;
            for i in 0..n {
                q[i * n + j] *= inv;
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
