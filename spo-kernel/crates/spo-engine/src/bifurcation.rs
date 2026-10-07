// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Bifurcation analysis (Keller 1977, Kuramoto 1975)

//! Finite-horizon Kuramoto measurements over independent coupling trials.
//!
//! Uses full-snapshot explicit Euler, with row-major target/source coupling
//! and no implicit population normalization. A crossing of R=0.1 is a
//! numerical classification, not a certified bifurcation or stability proof.

/// Check dimensions and finite inputs before allocating or indexing.
///
/// # Errors
/// Returns an error for dimensions, nonfinite values, or invalid timestep.
#[allow(clippy::too_many_arguments)]
pub fn validate_trial_inputs(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    k_scale: f64,
    dt: f64,
) -> Result<(), &'static str> {
    if n == 0 {
        return Err("n must be positive");
    }
    let square = n.checked_mul(n).ok_or("n squared overflows usize")?;
    if phases.len() != n || omegas.len() != n || knm.len() != square || alpha.len() != square {
        return Err("trial array lengths must match n and n squared");
    }
    if !phases
        .iter()
        .chain(omegas)
        .chain(knm)
        .chain(alpha)
        .all(|v| v.is_finite())
    {
        return Err("trial arrays must contain only finite values");
    }
    if !k_scale.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return Err("k_scale must be finite and dt finite and positive");
    }
    Ok(())
}

/// Return the post-step mean R, or an explicit input/arithmetic error.
///
/// A zero measurement window returns zero after input validation, without
/// executing transient steps. No convergence or stability certificate is made.
///
/// # Errors
/// Returns an error for malformed inputs or unrepresentable arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn try_steady_state_r(
    phases_init: &[f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    k_scale: f64,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> Result<f64, &'static str> {
    validate_trial_inputs(phases_init, omegas, knm_flat, alpha_flat, n, k_scale, dt)?;
    if n_measure == 0 {
        return Ok(0.0);
    }
    let mut phases = phases_init.to_vec();
    for _ in 0..n_transient {
        try_kuramoto_step(&mut phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)?;
    }
    let mut r_sum = 0.0;
    for _ in 0..n_measure {
        try_kuramoto_step(&mut phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)?;
        r_sum += order_parameter(&phases);
    }
    let r = r_sum / n_measure as f64;
    if !r.is_finite() || !(0.0..=1.0 + 1e-12).contains(&r) {
        return Err("steady-state R must be finite and lie in [0, 1]");
    }
    Ok(r.min(1.0))
}

/// Compatible scalar API; invalid inputs/arithmetic produce NaN.
///
/// Use `try_steady_state_r` to obtain the error reason.
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
    try_steady_state_r(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        k_scale,
        dt,
        n_transient,
        n_measure,
    )
    .unwrap_or(f64::NAN)
}

fn try_kuramoto_step(
    phases: &mut [f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    k_scale: f64,
    dt: f64,
) -> Result<(), &'static str> {
    let old = phases.to_vec();
    for i in 0..n {
        let mut coupling = 0.0;
        for j in 0..n {
            let k_ij = knm_flat[i * n + j] * k_scale;
            if !k_ij.is_finite() {
                return Err("scaled coupling overflow");
            }
            if k_ij == 0.0 {
                continue;
            }
            let angle = old[j] - old[i] - alpha_flat[i * n + j];
            if !angle.is_finite() {
                return Err("phase difference overflow");
            }
            coupling += k_ij * angle.sin();
        }
        let velocity = omegas[i] + coupling;
        let next = old[i] + dt * velocity;
        if !velocity.is_finite() || !next.is_finite() {
            return Err("Euler step overflow");
        }
        phases[i] = next;
    }
    Ok(())
}

#[cfg(test)]
fn kuramoto_step(
    phases: &mut [f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    scale: f64,
    dt: f64,
) {
    try_kuramoto_step(phases, omegas, knm, alpha, n, scale, dt).expect("valid unit-test trial");
}

fn order_parameter(phases: &[f64]) -> f64 {
    // The checked exported trial requires a nonempty population.
    let n = phases.len() as f64;
    let mut c = 0.0;
    let mut s = 0.0;
    for &theta in phases {
        c += theta.cos();
        s += theta.sin();
    }
    ((c / n).powi(2) + (s / n).powi(2)).sqrt()
}

/// Independently measure every grid point and interpolate the first upcrossing.
///
/// The optional crossing is represented by NaN when absent. Sampling the grid
/// is not pseudo-arclength continuation; each trial uses the same initial phases.
/// The direct native API retains singleton grids, equal endpoints and negative
/// coupling. The public Python sweep separately requires an increasing nonnegative
/// range and at least two points.
///
/// # Errors
/// Returns an error for an invalid grid or any invalid or overflowing trial.
#[allow(clippy::too_many_arguments)]
pub fn try_trace_sync_transition(
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    phases: &[f64],
    k_min: f64,
    k_max: f64,
    n_points: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64), &'static str> {
    use rayon::prelude::*;
    validate_trial_inputs(phases, omegas, knm, alpha, n, 1.0, dt)?;
    if !k_min.is_finite() || !k_max.is_finite() || k_max < k_min {
        return Err("native coupling range must be finite and nondecreasing");
    }
    if n_points == 0 {
        return Err("n_points must be positive");
    }
    let k_values: Vec<f64> = (0..n_points)
        .map(|i| {
            if n_points == 1 {
                return k_min;
            }
            let fraction = i as f64 / (n_points - 1) as f64;
            (1.0 - fraction) * k_min + fraction * k_max
        })
        .collect();
    let r_values: Vec<f64> = k_values
        .par_iter()
        .map(|&k| try_steady_state_r(phases, omegas, knm, alpha, n, k, dt, n_transient, n_measure))
        .collect::<Result<_, _>>()?;
    let mut critical = f64::NAN;
    for i in 0..n_points - 1 {
        if r_values[i] < 0.1 && r_values[i + 1] >= 0.1 {
            let fraction = (0.1 - r_values[i]) / (r_values[i + 1] - r_values[i]);
            // Convex interpolation keeps finite opposite-sign endpoints finite.
            critical = (1.0 - fraction) * k_values[i] + fraction * k_values[i + 1];
            break;
        }
    }
    Ok((k_values, r_values, critical))
}

/// Compatible sweep API; errors produce empty arrays and NaN.
///
/// Use `try_trace_sync_transition` to obtain the error reason.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn trace_sync_transition(
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    phases: &[f64],
    k_min: f64,
    k_max: f64,
    n_points: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    try_trace_sync_transition(
        omegas,
        knm,
        alpha,
        n,
        phases,
        k_min,
        k_max,
        n_points,
        dt,
        n_transient,
        n_measure,
    )
    .unwrap_or_else(|_| (vec![], vec![], f64::NAN))
}

/// Search for the R=0.1 classification boundary on `[0,20]`.
///
/// Assumes a monotone response; performs at most 30 iterations. Returns NaN
/// when the upper endpoint is below the threshold, including an empty window.
/// The lower endpoint is not measured. An already-superthreshold lower bracket
/// can therefore return a small positive interval midpoint without a transition.
///
/// # Errors
/// Returns an error for an invalid tolerance or any invalid or overflowing trial.
#[allow(clippy::too_many_arguments)]
pub fn try_find_critical_coupling(
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    phases: &[f64],
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    tol: f64,
) -> Result<f64, &'static str> {
    validate_trial_inputs(phases, omegas, knm, alpha, n, 1.0, dt)?;
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and positive");
    }
    let mut lo = 0.0;
    let mut hi = 20.0;
    if try_steady_state_r(
        phases,
        omegas,
        knm,
        alpha,
        n,
        hi,
        dt,
        n_transient,
        n_measure,
    )? < 0.1
    {
        return Ok(f64::NAN);
    }
    for _ in 0..30 {
        let mid = (lo + hi) / 2.0;
        if try_steady_state_r(
            phases,
            omegas,
            knm,
            alpha,
            n,
            mid,
            dt,
            n_transient,
            n_measure,
        )? < 0.1
        {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < tol {
            break;
        }
    }
    Ok((lo + hi) / 2.0)
}

/// Compatible search API; subthreshold upper endpoint or invalid computation is NaN.
///
/// The untested lower-bracket midpoint convention is retained. Use
/// `try_find_critical_coupling` to distinguish input/arithmetic errors.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn find_critical_coupling(
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    n: usize,
    phases: &[f64],
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    tol: f64,
) -> f64 {
    try_find_critical_coupling(
        omegas,
        knm,
        alpha,
        n,
        phases,
        dt,
        n_transient,
        n_measure,
        tol,
    )
    .unwrap_or(f64::NAN)
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
    fn test_empty_trial_is_refused_at_the_exported_boundary() {
        assert!(try_steady_state_r(&[], &[], &[], &[], 0, 1.0, 0.01, 0, 1).is_err());
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
