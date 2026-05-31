// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Plasticity model

use spo_types::{SpoError, SpoResult};

/// Three-factor Hebbian plasticity:
/// dK_ij/dt = lr * modulator * cos(theta_j - theta_i) - decay * K_ij
#[derive(Debug, Clone)]
pub struct PlasticityModel {
    pub lr: f64,
    pub decay: f64,
}

impl PlasticityModel {
    /// Create a non-negative Hebbian learning and decay rule.
    ///
    /// # Errors
    /// Returns `InvalidConfig` when either rate is negative.
    pub fn new(lr: f64, decay: f64) -> SpoResult<Self> {
        if !lr.is_finite() || !decay.is_finite() || lr < 0.0 || decay < 0.0 {
            return Err(SpoError::InvalidConfig(
                "lr and decay must be finite and non-negative".into(),
            ));
        }
        Ok(Self { lr, decay })
    }

    /// Update a dense row-major coupling matrix in place.
    ///
    /// The eligibility trace is `cos(theta_j - theta_i)`, implemented from
    /// precomputed sine and cosine buffers. Self-couplings are zeroed to
    /// preserve the Kuramoto no-self-edge coupling invariant.
    pub fn update(
        &self,
        sin_theta: &[f64],
        cos_theta: &[f64],
        knm: &mut [f64],
        modulator: f64,
        dt: f64,
    ) {
        let n = sin_theta.len();
        if knm.len() != n * n
            || cos_theta.len() != n
            || !modulator.is_finite()
            || !dt.is_finite()
            || dt < 0.0
            || sin_theta.iter().any(|v| !v.is_finite())
            || cos_theta.iter().any(|v| !v.is_finite())
            || knm.iter().any(|v| !v.is_finite())
        {
            return;
        }
        let decay_factor = (-self.decay * dt).exp();
        for i in 0..n {
            let ci = cos_theta[i];
            let si = sin_theta[i];
            for j in 0..n {
                if i == j {
                    knm[i * n + j] = 0.0;
                    continue;
                }
                let idx = i * n + j;
                let elig = cos_theta[j] * ci + sin_theta[j] * si;
                let delta = self.lr * modulator * elig * dt;
                knm[idx] = (knm[idx] * decay_factor + delta).max(0.0);
            }
        }
    }

    /// Update sparse CSR coupling values in place.
    ///
    /// The sparse structure is interpreted as `row_ptr` plus `col_indices`;
    /// entries where source and target are the same neuron are zeroed.
    pub fn update_sparse(
        &self,
        sin_theta: &[f64],
        cos_theta: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        modulator: f64,
        dt: f64,
    ) {
        let n = sin_theta.len();
        if cos_theta.len() != n
            || row_ptr.len() != n + 1
            || col_indices.len() != knm_values.len()
            || row_ptr.first().copied().unwrap_or_default() != 0
            || row_ptr.last().copied().unwrap_or_default() != col_indices.len()
            || row_ptr.windows(2).any(|w| w[1] < w[0])
            || col_indices.iter().any(|&idx| idx >= n)
            || !modulator.is_finite()
            || !dt.is_finite()
            || dt < 0.0
            || sin_theta.iter().any(|v| !v.is_finite())
            || cos_theta.iter().any(|v| !v.is_finite())
            || knm_values.iter().any(|v| !v.is_finite())
        {
            return;
        }
        let decay_factor = (-self.decay * dt).exp();
        for i in 0..n {
            let ci = cos_theta[i];
            let si = sin_theta[i];
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for idx in start..end {
                let j = col_indices[idx];
                if i == j {
                    knm_values[idx] = 0.0;
                    continue;
                }
                let elig = cos_theta[j] * ci + sin_theta[j] * si;
                let delta = self.lr * modulator * elig * dt;
                knm_values[idx] = (knm_values[idx] * decay_factor + delta).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_rates_are_rejected() {
        assert!(PlasticityModel::new(-0.1, 0.0).is_err());
        assert!(PlasticityModel::new(0.0, -0.1).is_err());
        assert!(PlasticityModel::new(f64::NAN, 0.0).is_err());
    }

    #[test]
    fn dense_update_zeros_self_edges_and_clamps_non_negative() {
        let model = PlasticityModel::new(1.0, 0.0).expect("model init failed");
        let phases = [0.0_f64, std::f64::consts::PI];
        let sin_theta: Vec<f64> = phases.iter().map(|p| p.sin()).collect();
        let cos_theta: Vec<f64> = phases.iter().map(|p| p.cos()).collect();
        let mut knm = vec![7.0, 0.0, 0.0, 9.0];

        model.update(&sin_theta, &cos_theta, &mut knm, 1.0, 1.0);

        assert_eq!(knm[0], 0.0);
        assert_eq!(knm[3], 0.0);
        assert_eq!(knm[1], 0.0);
        assert_eq!(knm[2], 0.0);
    }

    #[test]
    fn sparse_update_zeros_self_edges() {
        let model = PlasticityModel::new(1.0, 0.0).expect("model init failed");
        let phases = [0.0_f64, 0.0];
        let sin_theta: Vec<f64> = phases.iter().map(|p| p.sin()).collect();
        let cos_theta: Vec<f64> = phases.iter().map(|p| p.cos()).collect();
        let row_ptr = vec![0, 2, 2];
        let col_indices = vec![0, 1];
        let mut values = vec![5.0, 0.25];

        model.update_sparse(
            &sin_theta,
            &cos_theta,
            &row_ptr,
            &col_indices,
            &mut values,
            1.0,
            0.1,
        );

        assert_eq!(values[0], 0.0);
        assert!((values[1] - 0.35).abs() < 1e-12);
    }

    #[test]
    fn sparse_update_matches_positive_eligibility() {
        let model = PlasticityModel::new(0.5, 0.0).expect("model init failed");
        let phases = [0.0_f64, 0.0];
        let sin_theta: Vec<f64> = phases.iter().map(|p| p.sin()).collect();
        let cos_theta: Vec<f64> = phases.iter().map(|p| p.cos()).collect();
        let row_ptr = vec![0, 1, 1];
        let col_indices = vec![1];
        let mut values = vec![0.25];

        model.update_sparse(
            &sin_theta,
            &cos_theta,
            &row_ptr,
            &col_indices,
            &mut values,
            2.0,
            0.1,
        );

        assert!((values[0] - 0.35).abs() < 1e-12);
    }
}
