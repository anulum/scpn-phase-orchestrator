// SCPN Phase Orchestrator — Order Parameters
//!
//! R = |⟨e^{iθ}⟩|  (Kuramoto global order parameter)
//! PLV = |⟨e^{i(φ_a - φ_b)}⟩|  (Phase-Locking Value)

use std::f64::consts::TAU;

/// Kuramoto global order parameter: (R, psi_mean).
///
/// R = |mean(exp(i*theta))|, psi_mean = arg(mean(exp(i*theta))) mod 2π.
pub fn compute_order_parameter(phases: &[f64]) -> (f64, f64) {
    let n = phases.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let (sum_sin, sum_cos) = phases
        .iter()
        .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
    let mean_sin = sum_sin / n;
    let mean_cos = sum_cos / n;
    let r = (mean_sin * mean_sin + mean_cos * mean_cos)
        .sqrt()
        .clamp(0.0, 1.0);
    let psi = mean_sin.atan2(mean_cos).rem_euclid(TAU);
    (r, psi)
}

/// Phase-locking value between two equal-length phase arrays.
///
/// PLV = |mean(exp(i*(φ_a - φ_b)))| over samples.
pub fn compute_plv(phases_a: &[f64], phases_b: &[f64]) -> f64 {
    let n = phases_a.len().min(phases_b.len()) as f64;
    if n < 1.0 {
        return 0.0;
    }
    let (sum_sin, sum_cos) =
        phases_a
            .iter()
            .zip(phases_b.iter())
            .fold((0.0, 0.0), |(s, c), (&a, &b)| {
                let diff = a - b;
                (s + diff.sin(), c + diff.cos())
            });
    let mean_sin = sum_sin / n;
    let mean_cos = sum_cos / n;
    (mean_sin * mean_sin + mean_cos * mean_cos)
        .sqrt()
        .clamp(0.0, 1.0)
}

/// Order parameter R for oscillators selected by mask indices.
pub fn compute_layer_coherence(phases: &[f64], indices: &[usize]) -> f64 {
    let valid: Vec<f64> = indices
        .iter()
        .filter_map(|&i| phases.get(i).copied())
        .collect();
    if valid.is_empty() {
        return 0.0;
    }
    compute_order_parameter(&valid).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_equal_r_one() {
        let phases = vec![0.5; 16];
        let (r, _) = compute_order_parameter(&phases);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dispersed_r_near_zero() {
        let n = 16;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * TAU / n as f64).collect();
        let (r, _) = compute_order_parameter(&phases);
        assert!(r < 0.15, "dispersed R={r}");
    }

    #[test]
    fn empty_phases() {
        let (r, psi) = compute_order_parameter(&[]);
        assert_eq!(r, 0.0);
        assert_eq!(psi, 0.0);
    }

    #[test]
    fn plv_identical_signals() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        let plv = compute_plv(&a, &a);
        assert!((plv - 1.0).abs() < 1e-9);
    }

    #[test]
    fn plv_orthogonal_signals() {
        let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = a.iter().map(|&x| x + std::f64::consts::FRAC_PI_2).collect();
        let plv = compute_plv(&a, &b);
        // Constant phase difference → PLV = 1
        assert!((plv - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plv_empty() {
        assert_eq!(compute_plv(&[], &[]), 0.0);
    }

    #[test]
    fn layer_coherence_subset() {
        let phases = vec![0.5, 0.5, 3.0, 0.5];
        let r_all = compute_order_parameter(&phases).0;
        let r_sub = compute_layer_coherence(&phases, &[0, 1, 3]);
        assert!(
            r_sub > r_all,
            "subset {r_sub:.4} should be more coherent than all {r_all:.4}"
        );
    }

    #[test]
    fn layer_coherence_empty_indices() {
        assert_eq!(compute_layer_coherence(&[1.0, 2.0], &[]), 0.0);
    }

    #[test]
    fn layer_coherence_out_of_bounds() {
        let r = compute_layer_coherence(&[1.0], &[0, 5, 10]);
        assert!(r > 0.0); // only index 0 is valid → R=1.0
    }

    #[test]
    fn order_param_psi_range() {
        let phases = vec![1.0, 2.0, 3.0];
        let (_, psi) = compute_order_parameter(&phases);
        assert!((0.0..TAU).contains(&psi), "psi={psi} out of [0, 2π)");
    }
}
