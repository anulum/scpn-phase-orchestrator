// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Bifurcation analysis (Keller 1977, Kuramoto 1975)

//! Bifurcation continuation for Kuramoto synchronisation transitions.
//!
//! Traces steady-state order parameter R as a function of coupling strength K.
//! Detects critical coupling K_c where R bifurcates from 0 to partial sync.
//!
//! References:
//!   Kuramoto 1975, International Symposium on Mathematical Problems
//!     in Theoretical Physics, Lecture Notes in Physics 39:420-422.
//!   Strogatz 2000, Physica D 143:1-20.
//!   Keller 1977, "Numerical Solution of Bifurcation and Nonlinear
//!     Eigenvalue Problems".

/// Run Kuramoto ODE to steady state and return time-averaged R.
///
/// Euler integration of dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij)
///
/// # Arguments
/// * `phases_init` — (N,) initial phases
/// * `omegas` — (N,) natural frequencies
/// * `knm_flat` — (N×N) row-major coupling template (will be scaled by k_scale)
/// * `alpha_flat` — (N×N) row-major phase lag matrix
/// * `n` — number of oscillators
/// * `k_scale` — coupling strength multiplier
/// * `dt` — integration timestep
/// * `n_transient` — steps to discard
/// * `n_measure` — steps to average R over
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn steady_state_r(
    phases_init: &[f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    k_scale: f64,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> f64 {
    let mut phases = phases_init.to_vec();

    // Transient: integrate to steady state
    for _ in 0..n_transient {
        kuramoto_step(&mut phases, omegas, knm_flat, alpha_flat, n, k_scale, dt);
    }

    // Measure: time-averaged R
    let mut r_sum = 0.0;
    for _ in 0..n_measure {
        kuramoto_step(&mut phases, omegas, knm_flat, alpha_flat, n, k_scale, dt);
        r_sum += order_parameter(&phases);
    }

    r_sum / n_measure as f64
}

/// Single Euler step for Kuramoto model.
fn kuramoto_step(
    phases: &mut [f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    k_scale: f64,
    dt: f64,
) {
    // Compute coupling: c_i = Σ_j K_ij * sin(θ_j − θ_i − α_ij)
    // Then θ_i += dt * (ω_i + c_i)
    let old = phases.to_vec();
    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            let k_ij = knm_flat[i * n + j] * k_scale;
            if k_ij.abs() < 1e-30 {
                continue;
            }
            let a_ij = alpha_flat[i * n + j];
            coupling += k_ij * (old[j] - old[i] - a_ij).sin();
        }
        phases[i] = old[i] + dt * (omegas[i] + coupling);
    }
}

/// Kuramoto order parameter R = |<exp(iθ)>|.
fn order_parameter(phases: &[f64]) -> f64 {
    if phases.is_empty() {
        return 0.0;
    }
    let n = phases.len() as f64;
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    for &theta in phases {
        sum_cos += theta.cos();
        sum_sin += theta.sin();
    }
    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}

/// Trace R(K) bifurcation diagram.
///
/// Returns (K_values, R_values, K_critical).
/// K_critical is NaN if no transition found.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn trace_sync_transition(
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    phases_init: &[f64],
    k_min: f64,
    k_max: f64,
    n_points: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut k_values = Vec::with_capacity(n_points);
    let mut r_values = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let k = if n_points > 1 {
            k_min + (k_max - k_min) * i as f64 / (n_points - 1) as f64
        } else {
            k_min
        };
        k_values.push(k);
        let r = steady_state_r(
            phases_init,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            k,
            dt,
            n_transient,
            n_measure,
        );
        r_values.push(r);
    }

    // Find K_critical: first crossing of R = 0.1
    let threshold = 0.1;
    let mut k_critical = f64::NAN;
    for i in 0..r_values.len() - 1 {
        if r_values[i] < threshold && r_values[i + 1] >= threshold {
            let r_lo = r_values[i];
            let r_hi = r_values[i + 1];
            if r_hi > r_lo {
                let frac = (threshold - r_lo) / (r_hi - r_lo);
                k_critical = k_values[i] + frac * (k_values[i + 1] - k_values[i]);
            } else {
                k_critical = k_values[i + 1];
            }
            break;
        }
    }

    (k_values, r_values, k_critical)
}

/// Binary search for critical coupling K_c.
///
/// Returns K_c or NaN if no transition in [0, 20].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn find_critical_coupling(
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    phases_init: &[f64],
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    tol: f64,
) -> f64 {
    let threshold = 0.1;

    let mut k_lo = 0.0;
    let mut k_hi = 20.0;

    let r_hi = steady_state_r(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        k_hi,
        dt,
        n_transient,
        n_measure,
    );
    if r_hi < threshold {
        return f64::NAN;
    }

    for _ in 0..30 {
        let k_mid = (k_lo + k_hi) / 2.0;
        let r_mid = steady_state_r(
            phases_init,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            k_mid,
            dt,
            n_transient,
            n_measure,
        );
        if r_mid < threshold {
            k_lo = k_mid;
        } else {
            k_hi = k_mid;
        }
        if k_hi - k_lo < tol {
            break;
        }
    }

    (k_lo + k_hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn make_all_to_all(n: usize) -> Vec<f64> {
        let mut knm = vec![1.0 / n as f64; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        knm
    }

    #[test]
    fn test_order_parameter_sync() {
        let phases = vec![0.0, 0.0, 0.0, 0.0];
        let r = order_parameter(&phases);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_order_parameter_antisync() {
        let phases = vec![0.0, std::f64::consts::PI];
        let r = order_parameter(&phases);
        assert!(r < 1e-12);
    }

    #[test]
    fn test_order_parameter_empty() {
        assert_eq!(order_parameter(&[]), 0.0);
    }

    #[test]
    fn test_steady_state_r_zero_coupling() {
        // Zero coupling → oscillators drift → R ≈ 0 (for non-identical ω)
        let n = 8;
        let omegas: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();

        let r = steady_state_r(&phases, &omegas, &knm, &alpha, n, 0.0, 0.01, 500, 200);
        assert!(r < 0.5, "zero coupling should give low R, got {r}");
    }

    #[test]
    fn test_steady_state_r_strong_coupling() {
        // Strong coupling → R close to 1
        let n = 8;
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.05 * i as f64).collect();
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();

        let r = steady_state_r(&phases, &omegas, &knm, &alpha, n, 10.0, 0.01, 2000, 500);
        assert!(r > 0.8, "strong coupling should give high R, got {r}");
    }

    #[test]
    fn test_trace_monotone_r() {
        // R should generally increase with K
        let n = 6;
        let omegas: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();

        let (_, r_values, _) = trace_sync_transition(
            &omegas, &knm, &alpha, n, &phases, 0.0, 8.0, 5, 0.01, 500, 200,
        );
        // R at K=8 should be greater than R at K=0
        assert!(
            r_values[4] > r_values[0],
            "R should increase with K: R(0)={}, R(8)={}",
            r_values[0],
            r_values[4]
        );
    }

    #[test]
    fn test_trace_returns_correct_length() {
        let n = 4;
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases = vec![0.0; n];
        let (k_vals, r_vals, _) = trace_sync_transition(
            &omegas, &knm, &alpha, n, &phases, 0.0, 5.0, 10, 0.01, 100, 50,
        );
        assert_eq!(k_vals.len(), 10);
        assert_eq!(r_vals.len(), 10);
    }

    #[test]
    fn test_find_critical_coupling_exists() {
        // Identical frequencies → K_c = 0 (already synchronised at any K)
        let n = 4;
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases = vec![0.0; n];

        let kc = find_critical_coupling(&omegas, &knm, &alpha, n, &phases, 0.01, 500, 200, 0.1);
        // Identical ω: even tiny coupling syncs → K_c should be small
        assert!(
            kc < 5.0,
            "identical frequencies should have low K_c, got {kc}"
        );
    }

    #[test]
    fn test_find_critical_coupling_spread_frequencies() {
        // Spread frequencies → should find a finite K_c
        let n = 6;
        let omegas: Vec<f64> = (0..n).map(|i| 0.5 + 1.0 * i as f64).collect();
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();

        let kc = find_critical_coupling(&omegas, &knm, &alpha, n, &phases, 0.01, 2000, 500, 0.1);
        assert!(!kc.is_nan(), "should find K_c for spread frequencies");
        // K_c > 0 (some coupling needed)
        assert!(
            kc > 0.0,
            "spread frequencies need nonzero coupling, got {kc}"
        );
    }

    #[test]
    fn test_kuramoto_step_preserves_count() {
        let n = 5;
        let mut phases = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let omegas = vec![1.0; n];
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        kuramoto_step(&mut phases, &omegas, &knm, &alpha, n, 1.0, 0.01);
        assert_eq!(phases.len(), n);
        // All phases should have changed
        assert!((phases[0] - 0.0).abs() > 1e-10);
    }

    #[test]
    fn test_r_values_bounded() {
        let n = 4;
        let omegas = vec![1.0, 2.0, 3.0, 4.0];
        let knm = make_all_to_all(n);
        let alpha = vec![0.0; n * n];
        let phases = vec![0.0, 1.0, 2.0, 3.0];

        let (_, r_vals, _) = trace_sync_transition(
            &omegas, &knm, &alpha, n, &phases, 0.0, 10.0, 5, 0.01, 200, 100,
        );
        for (i, &r) in r_vals.iter().enumerate() {
            assert!(r >= 0.0 && r <= 1.0 + 1e-10, "R[{i}] = {r} out of [0,1]");
        }
    }
}
