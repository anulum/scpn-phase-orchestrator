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
    pub fn new(lr: f64, decay: f64) -> SpoResult<Self> {
        if lr < 0.0 || decay < 0.0 {
            return Err(SpoError::InvalidConfig("lr and decay must be non-negative".into()));
        }
        Ok(Self { lr, decay })
    }

    pub fn update(&self, sin_theta: &[f64], cos_theta: &[f64], knm: &mut [f64], modulator: f64, dt: f64) {
        let n = sin_theta.len();
        if knm.len() != n * n { return; }
        let decay_factor = (-self.decay * dt).exp();
        for i in 0..n {
            let ci = cos_theta[i];
            let si = sin_theta[i];
            for j in 0..n {
                if i == j { continue; }
                let idx = i * n + j;
                let elig = cos_theta[j] * ci + sin_theta[j] * si;
                let delta = self.lr * modulator * elig * dt;
                knm[idx] = (knm[idx] * decay_factor + delta).max(0.0);
            }
        }
    }

    pub fn update_sparse(&self, sin_theta: &[f64], cos_theta: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], modulator: f64, dt: f64) {
        let n = sin_theta.len();
        let decay_factor = (-self.decay * dt).exp();
        for i in 0..n {
            let ci = cos_theta[i];
            let si = sin_theta[i];
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for idx in start..end {
                let j = col_indices[idx];
                if i == j { continue; }
                let elig = cos_theta[j] * ci + sin_theta[j] * si;
                let delta = self.lr * modulator * elig * dt;
                knm_values[idx] = (knm_values[idx] * decay_factor + delta).max(0.0);
            }
        }
    }
}
