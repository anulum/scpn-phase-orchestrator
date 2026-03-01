// SCPN Phase Orchestrator — Coupling Matrix Builder
//!
//! K_ij = base_strength * exp(-decay_alpha * |i - j|), zero diagonal.

use spo_types::{CouplingConfig, SpoError, SpoResult};

#[derive(Debug, Clone)]
pub struct CouplingState {
    pub knm: Vec<f64>,
    pub alpha: Vec<f64>,
    pub n: usize,
}

pub struct CouplingBuilder;

impl CouplingBuilder {
    /// Build exponential-decay coupling matrix (row-major N×N).
    pub fn build(n: usize, config: &CouplingConfig) -> SpoResult<CouplingState> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;

        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist = (i as f64 - j as f64).abs();
                    knm[i * n + j] = config.base_strength * (-config.decay_alpha * dist).exp();
                }
            }
        }

        let alpha = vec![0.0; n * n];
        Ok(CouplingState { knm, alpha, n })
    }
}

/// Project Knm to satisfy: symmetric, non-negative, zero diagonal.
pub fn project_knm(knm: &mut [f64], n: usize) {
    // Symmetry: K = (K + K^T) / 2
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (knm[i * n + j] + knm[j * n + i]);
            knm[i * n + j] = avg;
            knm[j * n + i] = avg;
        }
    }
    // Non-negative
    for v in knm.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    // Zero diagonal
    for i in 0..n {
        knm[i * n + i] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default() {
        let cs = CouplingBuilder::build(4, &CouplingConfig::default()).unwrap();
        assert_eq!(cs.knm.len(), 16);
        assert_eq!(cs.alpha.len(), 16);
        // Zero diagonal
        for i in 0..4 {
            assert_eq!(cs.knm[i * 4 + i], 0.0);
        }
    }

    #[test]
    fn build_symmetric() {
        let cs = CouplingBuilder::build(4, &CouplingConfig::default()).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (cs.knm[i * 4 + j] - cs.knm[j * 4 + i]).abs() < 1e-12,
                    "knm[{i},{j}] != knm[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn build_non_negative() {
        let cs = CouplingBuilder::build(8, &CouplingConfig::default()).unwrap();
        assert!(cs.knm.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn build_exponential_decay() {
        let cfg = CouplingConfig {
            base_strength: 1.0,
            decay_alpha: 1.0,
        };
        let cs = CouplingBuilder::build(4, &cfg).unwrap();
        // K[0,1] = exp(-1) > K[0,2] = exp(-2) > K[0,3] = exp(-3)
        assert!(cs.knm[1] > cs.knm[2]);
        assert!(cs.knm[2] > cs.knm[3]);
    }

    #[test]
    fn build_zero_n_rejected() {
        assert!(CouplingBuilder::build(0, &CouplingConfig::default()).is_err());
    }

    #[test]
    fn project_enforces_constraints() {
        let n = 3;
        let mut knm = vec![
            0.0, -0.5, 0.3, // row 0: negative, asymmetric
            0.1, 0.0, 0.2, // row 1
            0.4, 0.7, 0.0, // row 2
        ];
        project_knm(&mut knm, n);

        // Zero diagonal
        for i in 0..n {
            assert_eq!(knm[i * n + i], 0.0);
        }
        // Non-negative
        assert!(knm.iter().all(|&v| v >= 0.0));
        // Symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (knm[i * n + j] - knm[j * n + i]).abs() < 1e-12,
                    "not symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn build_n_one() {
        let cs = CouplingBuilder::build(1, &CouplingConfig::default()).unwrap();
        assert_eq!(cs.knm, vec![0.0]);
    }

    #[test]
    fn alpha_initially_zero() {
        let cs = CouplingBuilder::build(4, &CouplingConfig::default()).unwrap();
        assert!(cs.alpha.iter().all(|&v| v == 0.0));
    }
}
