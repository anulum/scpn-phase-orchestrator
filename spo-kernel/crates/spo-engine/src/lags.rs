// SCPN Phase Orchestrator — Lag Model
//!
//! Estimates phase lags α_ij from propagation distances and speed.
//! α_ij = 2π * distance_ij / speed (antisymmetric).

use std::f64::consts::TAU;

pub struct LagModel {
    pub alpha: Vec<f64>,
    pub n: usize,
}

impl LagModel {
    /// Build α matrix from pairwise distances (row-major N×N) and propagation speed.
    ///
    /// α_ij = 2π * distances_ij / speed, antisymmetric: α_ji = -α_ij.
    #[must_use]
    pub fn estimate_from_distances(distances: &[f64], n: usize, speed: f64) -> Self {
        let mut alpha = vec![0.0; n * n];
        if speed <= 0.0 || !speed.is_finite() {
            return Self { alpha, n };
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let lag = TAU * distances[i * n + j] / speed;
                alpha[i * n + j] = lag;
                alpha[j * n + i] = -lag;
            }
        }
        Self { alpha, n }
    }

    /// Zero lag model.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            alpha: vec![0.0; n * n],
            n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_model() {
        let lm = LagModel::zeros(4);
        assert!(lm.alpha.iter().all(|&v| v == 0.0));
        assert_eq!(lm.n, 4);
    }

    #[test]
    fn antisymmetric() {
        let n = 3;
        let distances = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0];
        let lm = LagModel::estimate_from_distances(&distances, n, 1.0);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (lm.alpha[i * n + j] + lm.alpha[j * n + i]).abs() < 1e-12,
                    "not antisymmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn zero_speed_gives_zeros() {
        let lm = LagModel::estimate_from_distances(&[0.0, 1.0, 1.0, 0.0], 2, 0.0);
        assert!(lm.alpha.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scaling() {
        let n = 2;
        let distances = vec![0.0, 1.0, 1.0, 0.0];
        let lm1 = LagModel::estimate_from_distances(&distances, n, 1.0);
        let lm2 = LagModel::estimate_from_distances(&distances, n, 2.0);
        // Half speed → double lag
        assert!((lm1.alpha[1] - 2.0 * lm2.alpha[1]).abs() < 1e-12);
    }

    #[test]
    fn nan_distance_no_panic() {
        let n = 2;
        let distances = vec![0.0, f64::NAN, f64::NAN, 0.0];
        let lm = LagModel::estimate_from_distances(&distances, n, 1.0);
        // NaN propagates into alpha — verify no panic and alpha has NaN
        assert!(lm.alpha[1].is_nan());
        assert!(lm.alpha[2].is_nan());
    }
}
