// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Order parameters

use spo_types::{SpoError, SpoResult};
use std::f64::consts::TAU;

/// Kuramoto global order parameter: (R, psi_mean).
#[must_use]
pub fn compute_order_parameter(phases: &[f64]) -> (f64, f64) {
    let n = phases.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    for &th in phases {
        let (s, c) = th.sin_cos();
        sum_sin += s;
        sum_cos += c;
    }
    assemble_r_psi(sum_sin, sum_cos, n as f64)
}

/// Compute order parameter from precomputed sin/cos arrays.
#[must_use]
pub fn compute_order_parameter_from_sincos(sin_theta: &[f64], cos_theta: &[f64]) -> (f64, f64) {
    let n = sin_theta.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let sum_sin: f64 = sin_theta.iter().sum();
    let sum_cos: f64 = cos_theta.iter().sum();
    assemble_r_psi(sum_sin, sum_cos, n as f64)
}

#[inline]
fn assemble_r_psi(sum_sin: f64, sum_cos: f64, n: f64) -> (f64, f64) {
    let mean_sin = sum_sin / n;
    let mean_cos = sum_cos / n;
    let r = (mean_sin * mean_sin + mean_cos * mean_cos)
        .sqrt()
        .clamp(0.0, 1.0);
    let psi = mean_sin.atan2(mean_cos).rem_euclid(TAU);
    (r, psi)
}

pub fn compute_plv(phases_a: &[f64], phases_b: &[f64]) -> SpoResult<f64> {
    if phases_a.len() != phases_b.len() {
        return Err(SpoError::InvalidDimension(format!(
            "PLV requires equal-length arrays, got {} vs {}",
            phases_a.len(),
            phases_b.len()
        )));
    }
    let n = phases_a.len();
    if n == 0 {
        return Ok(0.0);
    }
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    for (&a, &b) in phases_a.iter().zip(phases_b) {
        let (s, c) = (a - b).sin_cos();
        sum_sin += s;
        sum_cos += c;
    }
    Ok(assemble_r_psi(sum_sin, sum_cos, n as f64).0)
}

#[must_use]
pub fn compute_layer_coherence(phases: &[f64], indices: &[usize]) -> f64 {
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut count = 0;
    for &i in indices {
        if let Some(&th) = phases.get(i) {
            let (s, c) = th.sin_cos();
            sum_sin += s;
            sum_cos += c;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    assemble_r_psi(sum_sin, sum_cos, count as f64).0
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
        assert!(r < 0.15);
    }
}
