// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Active Inference Control Agent

use spo_engine::order_params::compute_order_parameter;
use spo_types::{SpoError, SpoResult};

/// Active Inference agent for control of phase synchronisation.
pub struct ActiveInferenceAgent {
    pub n_hidden: usize,
    pub target_r: f64,
    pub lr: f64,
    state: Vec<f64>,
    sin_state: Vec<f64>,
    cos_state: Vec<f64>,
}

impl ActiveInferenceAgent {
    /// Create an active inference controller.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_hidden` is zero.
    pub fn new(n_hidden: usize, target_r: f64, lr: f64) -> SpoResult<Self> {
        if n_hidden == 0 {
            return Err(SpoError::InvalidConfig("n_hidden > 0".into()));
        }
        Ok(Self {
            n_hidden,
            target_r,
            lr,
            state: vec![0.0; n_hidden],
            sin_state: vec![0.0; n_hidden],
            cos_state: vec![0.0; n_hidden],
        })
    }

    pub fn control(&mut self, phases: &[f64]) -> (f64, f64) {
        let n = phases.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let nh = self.n_hidden;

        for i in 0..nh {
            let (s, c) = self.state[i].sin_cos();
            self.sin_state[i] = s;
            self.cos_state[i] = c;
        }

        let (r_obs, psi_obs): (f64, f64) = compute_order_parameter(phases);
        let error: f64 = self.target_r - r_obs;

        for i in 0..nh {
            let ci = self.cos_state[i];
            let si = self.sin_state[i];
            let mut internal_coupling = 0.0;
            for j in 0..nh {
                if i == j {
                    continue;
                }
                internal_coupling += self.sin_state[j] * ci - self.cos_state[j] * si;
            }
            let d_state =
                self.lr * (error * (psi_obs - self.state[i]).sin() + internal_coupling / nh as f64);
            self.state[i] = (self.state[i] + d_state).rem_euclid(2.0 * std::f64::consts::PI);
        }

        let zeta = (error * 10.0).clamp(0.0, 5.0);
        let psi = if error > 0.0 {
            (psi_obs + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        } else {
            psi_obs
        };

        (zeta, psi)
    }

    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}
