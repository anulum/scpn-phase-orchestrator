// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Active Inference Control Agent

use spo_types::{SpoError, SpoResult};

/// Active Inference Controller using a State-Space Predictive Model.
///
/// This agent implements Karl Friston's Variational Free Energy Principle 
/// (FEP) to autonomously control oscillator synchronization. It treats 
/// control as the minimization of prediction error (Variational Free Energy) 
/// relative to a target coherence level $R_{target}$.
///
/// The controller represents the next-generation of SPO supervisory logic, 
/// moving away from static heuristic rules to dynamic internal world-models.
///
/// # Internal State
/// The agent maintains a low-dimensional predictive state $x$ that evolves 
/// according to:
/// $\dot{x} = A x + B \epsilon$
/// where $\epsilon = R_{obs} - R_{target}$ is the prediction error.
///
/// This agent implements Karl Friston's Variational Free Energy Principle 
/// (FEP) to autonomously control oscillator synchronization. It treats 
/// control as the minimization of prediction error (Variational Free Energy) 
/// relative to a target coherence level {target}$.
///
/// The controller represents the next-generation of SPO supervisory logic, 
/// moving away from static heuristic rules to dynamic internal world-models.
///
/// # Internal State
/// The agent maintains a low-dimensional predictive state $ that evolves 
/// according to:
/// $\dot{x} = A x + B \epsilon$
/// where $\epsilon = R_{obs} - R_{target}$ is the prediction error.
/// Minimizes variational free energy (prediction error) between
/// current coherence R and target R_target.
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    pub target_r: f64,
    pub lr: f64,
    pub internal_state: Vec<f64>,
    pub weights: Vec<f64>, // SSM/RNN weights
}

impl ActiveInferenceAgent {
    pub fn new(n_hidden: usize, target_r: f64, lr: f64) -> SpoResult<Self> {
        if target_r < 0.0 || target_r > 1.0 {
            return Err(SpoError::InvalidConfig("target_r must be in [0, 1]".into()));
        }
        Ok(Self {
            target_r,
            lr,
            internal_state: vec![0.0; n_hidden],
            weights: vec![0.1; n_hidden * n_hidden + n_hidden * 2], // Simple RNN/SSM weights
        })
    }

    /// Update internal state and compute optimal control knobs (zeta, psi).
    ///
    /// The agent outputs an 'action' (zeta, psi) that acts on the world 
    /// (the oscillator network) to resolve the divergence between predicted 
    /// and observed coherence.
    ///
    /// # Arguments
    /// * `r_obs` - Currently observed Order Parameter R.
    /// * `psi_obs` - Currently observed global phase Psi.
    /// * `dt` - Timestep for internal state-space update.
    ///
    /// The agent outputs an 'action' (zeta, psi) that acts on the world 
    /// (the oscillator network) to resolve the divergence between predicted 
    /// and observed coherence.
    ///
    /// # Arguments
    /// *  - Currently observed Order Parameter R.
    /// *  - Currently observed global phase Psi.
    /// *  - Timestep for internal state-space update.
    ///
    /// # Arguments
    /// * `r_obs` - Currently observed Order Parameter R.
    /// * `psi_obs` - Currently observed global phase Psi.
    /// * `dt` - Timestep.
    pub fn control(&mut self, r_obs: f64, psi_obs: f64, dt: f64) -> (f64, f64) {
        // Friston-style Active Inference: 
        // minimize error e = (r_obs - target_r)
        let error = r_obs - self.target_r;
        
        // Update internal predictive state (simple LNN update)
        // x_dot = A*x + B*error
        let n = self.internal_state.len();
        for i in 0..n {
            let mut x_dot = 0.0;
            for j in 0..n {
                x_dot += self.weights[i * n + j] * self.internal_state[j];
            }
            x_dot += self.weights[n * n + i] * error;
            self.internal_state[i] += x_dot * dt;
        }
        
        // Output control zeta: proportional to error + internal integrated state
        // In Active Inference, zeta is the 'action' that reduces prediction error
        let zeta = (self.lr * error.abs() + self.internal_state.iter().sum::<f64>()).clamp(0.0, 10.0);
        
        // Target phase psi: align with observed global phase to encourage sync, 
        // or anti-align to suppress it.
        let psi = if error > 0.0 {
            // Suppress sync: drive out of phase
            (psi_obs + std::f64::consts::PI) % std::f64::consts::TAU
        } else {
            // Encourage sync: drive in phase
            psi_obs
        };
        
        (zeta, psi)
    }
}
