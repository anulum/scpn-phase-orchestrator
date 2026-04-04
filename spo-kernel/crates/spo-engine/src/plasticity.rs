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
///
/// This model implements a sub-microsecond plasticity update directly
/// in the Rust integration loop. It allows coupling matrices (dense or
/// sparse) to adapt their topology on-the-fly based on phase coherence.
///
/// Factors:
/// 1. Pre/Post-synaptic correlation (Hebbian): cos(theta_j - theta_i)
/// 2. Neuromodulatory reward/error signal: modulator
/// 3. Structural decay: exp(-decay * dt)
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

    /// Update coupling matrix K_nm in-place based on current phases.
    pub fn update(&self, phases: &[f64], knm: &mut [f64], modulator: f64, dt: f64) {
        let n = phases.len();
        if knm.len() != n * n { return; }

        let decay_factor = (-self.decay * dt).exp();

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let idx = i * n + j;
                let elig = (phases[j] - phases[i]).cos();
                let delta = self.lr * modulator * elig * dt;
                knm[idx] = (knm[idx] * decay_factor + delta).max(0.0);
            }
        }
    }

    /// Update sparse coupling matrix values in-place.
    pub fn update_sparse(
        &self,
        phases: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        modulator: f64,
        dt: f64
    ) {
        let n = phases.len();
        let decay_factor = (-self.decay * dt).exp();

        for i in 0..n {
            let start = row_ptr[i];
            let end = row_ptr[i+1];
            for idx in start..end {
                let j = col_indices[idx];
                if i == j { continue; }
                let elig = (phases[j] - phases[i]).cos();
                let delta = self.lr * modulator * elig * dt;
                knm_values[idx] = (knm_values[idx] * decay_factor + delta).max(0.0);
            }
        }
    }
}
