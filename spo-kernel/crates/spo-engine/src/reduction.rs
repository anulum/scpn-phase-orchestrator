// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ott-Antonsen mean-field reduction

//! Exact mean-field reduction for globally-coupled Kuramoto with
//! Lorentzian g(ω) distribution.
//!
//! dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)
//!
//! Critical coupling K_c = 2Δ.
//! Steady-state: R_ss = √(1 - 2Δ/K) for K > K_c.
//!
//! Ott & Antonsen 2008, Chaos 18(3):037113.

/// Run Ott-Antonsen reduction (RK4) for `n_steps`.
///
/// The complex ODE on the OA manifold:
///   dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)
///
/// # Arguments
/// * `z_re`, `z_im` – initial mean-field z = z_re + i·z_im
/// * `omega_0` – centre frequency of Lorentzian g(ω)
/// * `delta` – half-width of Lorentzian
/// * `k_coupling` – coupling strength K
/// * `dt` – time step
/// * `n_steps` – number of RK4 steps
///
/// # Returns
/// `(z_re, z_im, R, psi)` where R = |z|, psi = arg(z).
#[must_use]
pub fn oa_run(
    z_re: f64,
    z_im: f64,
    omega_0: f64,
    delta: f64,
    k_coupling: f64,
    dt: f64,
    n_steps: usize,
) -> (f64, f64, f64, f64) {
    let mut re = z_re;
    let mut im = z_im;
    let half_k = k_coupling / 2.0;

    for _ in 0..n_steps {
        // RK4 on complex ODE
        let (k1r, k1i) = oa_deriv(re, im, omega_0, delta, half_k);
        let (k2r, k2i) = oa_deriv(
            re + 0.5 * dt * k1r,
            im + 0.5 * dt * k1i,
            omega_0,
            delta,
            half_k,
        );
        let (k3r, k3i) = oa_deriv(
            re + 0.5 * dt * k2r,
            im + 0.5 * dt * k2i,
            omega_0,
            delta,
            half_k,
        );
        let (k4r, k4i) = oa_deriv(re + dt * k3r, im + dt * k3i, omega_0, delta, half_k);
        re += (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        im += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i);
    }

    let r = (re * re + im * im).sqrt();
    let psi = im.atan2(re);
    (re, im, r, psi)
}

/// OA manifold ODE derivative (real, imaginary parts).
///
/// dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)
fn oa_deriv(re: f64, im: f64, omega_0: f64, delta: f64, half_k: f64) -> (f64, f64) {
    let abs_sq = re * re + im * im;
    // -(Δ + iω₀)(re + i·im) = -Δ·re + ω₀·im + i(-Δ·im - ω₀·re)
    let lin_re = -delta * re + omega_0 * im;
    let lin_im = -delta * im - omega_0 * re;
    // (K/2)(z - |z|²z) = (K/2)(1 - |z|²)z
    let cubic_factor = half_k * (1.0 - abs_sq);
    let cub_re = cubic_factor * re;
    let cub_im = cubic_factor * im;
    (lin_re + cub_re, lin_im + cub_im)
}

/// Analytical steady-state R_ss = √(1 - 2Δ/K) for K > K_c = 2Δ.
#[must_use]
pub fn steady_state_r_oa(delta: f64, k_coupling: f64) -> f64 {
    let k_c = 2.0 * delta;
    if k_coupling <= k_c {
        return 0.0;
    }
    (1.0 - 2.0 * delta / k_coupling).sqrt()
}

/// Estimate Lorentzian parameters from frequency array.
///
/// Uses median as ω₀ and IQR/2 as Δ (the IQR of a Lorentzian equals 2Δ).
///
/// Returns (omega_0, delta).
#[must_use]
pub fn fit_lorentzian(omegas: &[f64]) -> (f64, f64) {
    if omegas.is_empty() {
        return (0.0, 0.01);
    }
    let mut sorted = omegas.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let omega_0 = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let q25 = sorted[n / 4];
    let q75 = sorted[3 * n / 4];
    let delta = if q75 > q25 { (q75 - q25) / 2.0 } else { 0.01 };
    (omega_0, delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_steady_state_subcritical() {
        // K < K_c = 2Δ → R = 0
        assert_eq!(steady_state_r_oa(1.0, 1.5), 0.0);
    }

    #[test]
    fn test_steady_state_supercritical() {
        // K=4, Δ=1 → R = √(1-2/4) = √0.5 ≈ 0.707
        let r = steady_state_r_oa(1.0, 4.0);
        assert!((r - 0.5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_oa_run_converges_supercritical() {
        // K=4 >> K_c=2, should converge to R ≈ 0.707
        let (_, _, r, _) = oa_run(0.01, 0.0, 0.0, 1.0, 4.0, 0.01, 2000);
        let expected = steady_state_r_oa(1.0, 4.0);
        assert!((r - expected).abs() < 0.05, "R={r}, expected≈{expected}");
    }

    #[test]
    fn test_oa_run_subcritical_decays() {
        // K=1 < K_c=2 → z should decay to 0
        let (_, _, r, _) = oa_run(0.5, 0.0, 0.0, 1.0, 1.0, 0.01, 2000);
        assert!(r < 0.1, "R={r} should decay below 0.1");
    }

    #[test]
    fn test_oa_run_with_rotation() {
        // ω₀ ≠ 0 → mean field rotates
        let (_, _, r, _) = oa_run(0.01, 0.0, 5.0, 0.5, 4.0, 0.01, 2000);
        let expected = steady_state_r_oa(0.5, 4.0);
        assert!((r - expected).abs() < 0.1, "R={r}, expected≈{expected}");
    }

    #[test]
    fn test_fit_lorentzian_basic() {
        let omegas: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let (omega_0, delta) = fit_lorentzian(&omegas);
        // Median of 0..9.9 ≈ 4.95
        assert!((omega_0 - 4.95).abs() < 0.1, "ω₀={omega_0}");
        assert!(delta > 0.0);
    }

    #[test]
    fn test_fit_lorentzian_empty() {
        let (omega_0, delta) = fit_lorentzian(&[]);
        assert_eq!(omega_0, 0.0);
        assert_eq!(delta, 0.01);
    }

    #[test]
    fn test_zero_steps() {
        let (re, im, r, _) = oa_run(0.5, 0.3, 0.0, 1.0, 2.0, 0.01, 0);
        assert!((re - 0.5).abs() < 1e-15);
        assert!((im - 0.3).abs() < 1e-15);
        let expected_r = (0.5_f64 * 0.5 + 0.3 * 0.3).sqrt();
        assert!((r - expected_r).abs() < 1e-10);
    }

    #[test]
    fn test_psi_angle() {
        let (_, _, _, psi) = oa_run(0.0, 0.01, 0.0, 0.5, 4.0, 0.01, 1);
        // Starting at (0, 0.01) → angle ≈ π/2
        assert!((psi - PI / 2.0).abs() < 0.1, "psi={psi}");
    }

    #[test]
    fn test_critical_coupling() {
        // At exactly K_c, R_ss = 0
        assert_eq!(steady_state_r_oa(1.0, 2.0), 0.0);
    }
}
