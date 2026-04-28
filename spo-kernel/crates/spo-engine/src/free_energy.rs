// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — SSGF free energy and Langevin dynamics

//! SSGF free energy helpers: Langevin noise, Boltzmann weight, effective temperature.
//!
//! Gardiner 2009, Stochastic Methods, §4.3.

/// Add Langevin stochastic noise: z_new = z + √(2·T·dt) · η, η ~ N(0,1).
///
/// Uses LCG + Box-Muller transform for deterministic Gaussian noise.
#[must_use]
pub fn add_langevin_noise(z: &[f64], temperature: f64, dt: f64, seed: u64) -> Vec<f64> {
    if temperature <= 0.0 || dt <= 0.0 {
        return z.to_vec();
    }
    let sigma = (2.0 * temperature * dt).sqrt();
    let mut result = z.to_vec();
    let mut rng_state = seed;

    // Box-Muller: generate pairs of Gaussian samples from uniform pairs
    let mut i = 0;
    while i < result.len() {
        // Generate two uniform random numbers in (0, 1)
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = if u1 < 1e-30 { 1e-30 } else { u1 };

        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let g1 = r * theta.cos();
        let g2 = r * theta.sin();

        result[i] += sigma * g1;
        if i + 1 < result.len() {
            result[i + 1] += sigma * g2;
        }
        i += 2;
    }
    result
}

/// Boltzmann factor exp(−U/T), clamped to avoid over/underflow.
#[must_use]
pub fn boltzmann_weight(u_total: f64, temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return if u_total <= 0.0 { 1.0 } else { 0.0 };
    }
    let exponent = (-u_total / temperature).clamp(-700.0, 700.0);
    exponent.exp()
}

/// Effective temperature from cost fluctuations: T_eff = Var(U) / (2·|<U>|).
#[must_use]
pub fn effective_temperature(costs: &[f64]) -> f64 {
    if costs.len() < 2 {
        return 0.0;
    }
    let n = costs.len() as f64;
    let mean = costs.iter().sum::<f64>() / n;
    let var = costs.iter().map(|&c| (c - mean) * (c - mean)).sum::<f64>() / (n - 1.0);
    if mean.abs() < 1e-30 {
        return 0.0;
    }
    var / (2.0 * mean.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin_zero_temperature() {
        let z = vec![1.0, 2.0, 3.0];
        let result = add_langevin_noise(&z, 0.0, 0.01, 42);
        assert_eq!(result, z);
    }

    #[test]
    fn test_langevin_changes_values() {
        let z = vec![0.0; 10];
        let result = add_langevin_noise(&z, 1.0, 0.01, 42);
        let changed = result.iter().any(|&v| v.abs() > 1e-10);
        assert!(changed, "Langevin noise should modify values");
    }

    #[test]
    fn test_langevin_deterministic() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r1 = add_langevin_noise(&z, 0.5, 0.01, 123);
        let r2 = add_langevin_noise(&z, 0.5, 0.01, 123);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_boltzmann_zero_energy() {
        let w = boltzmann_weight(0.0, 1.0);
        assert!((w - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_boltzmann_high_energy() {
        let w = boltzmann_weight(100.0, 1.0);
        assert!(w < 1e-40, "high energy should give near-zero weight");
    }

    #[test]
    fn test_boltzmann_zero_temperature() {
        assert_eq!(boltzmann_weight(1.0, 0.0), 0.0);
        assert_eq!(boltzmann_weight(0.0, 0.0), 1.0);
        assert_eq!(boltzmann_weight(-1.0, 0.0), 1.0);
    }

    #[test]
    fn test_boltzmann_negative_energy() {
        let w = boltzmann_weight(-1.0, 1.0);
        assert!(w > 1.0, "negative energy should give weight > 1");
    }

    #[test]
    fn test_effective_temperature_constant() {
        let costs = vec![1.0; 100];
        assert_eq!(effective_temperature(&costs), 0.0);
    }

    #[test]
    fn test_effective_temperature_positive() {
        let costs: Vec<f64> = (0..100).map(|i| 1.0 + 0.1 * (i as f64).sin()).collect();
        let t = effective_temperature(&costs);
        assert!(t > 0.0, "fluctuating costs should give positive T");
    }

    #[test]
    fn test_effective_temperature_short() {
        assert_eq!(effective_temperature(&[1.0]), 0.0);
        assert_eq!(effective_temperature(&[]), 0.0);
    }
}
