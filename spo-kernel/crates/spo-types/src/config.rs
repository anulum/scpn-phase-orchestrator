// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Configuration types

use serde::{Deserialize, Serialize};

use crate::error::{SpoError, SpoResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    Euler,
    RK4,
    RK45,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub dt: f64,
    pub method: Method,
    pub n_substeps: u32,
    pub atol: f64,
    pub rtol: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            method: Method::Euler,
            n_substeps: 1,
            atol: 1e-6,
            rtol: 1e-3,
        }
    }
}

impl IntegrationConfig {
    /// # Errors
    /// Returns `InvalidConfig` if dt is non-positive/non-finite, n_substeps is 0,
    /// or atol/rtol are non-positive/non-finite.
    pub fn validate(&self) -> SpoResult<()> {
        if self.dt <= 0.0 || !self.dt.is_finite() {
            return Err(SpoError::InvalidConfig(format!(
                "dt must be positive finite, got {}",
                self.dt
            )));
        }
        if self.n_substeps == 0 {
            return Err(SpoError::InvalidConfig("n_substeps must be >= 1".into()));
        }
        if self.atol <= 0.0 || !self.atol.is_finite() {
            return Err(SpoError::InvalidConfig(format!(
                "atol must be positive finite, got {}",
                self.atol
            )));
        }
        if self.rtol <= 0.0 || !self.rtol.is_finite() {
            return Err(SpoError::InvalidConfig(format!(
                "rtol must be positive finite, got {}",
                self.rtol
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingConfig {
    pub base_strength: f64,
    pub decay_alpha: f64,
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            base_strength: 0.45,
            decay_alpha: 0.3,
        }
    }
}

impl CouplingConfig {
    /// # Errors
    /// Returns `InvalidConfig` if base_strength is non-finite or decay_alpha is negative/non-finite.
    pub fn validate(&self) -> SpoResult<()> {
        if !self.base_strength.is_finite() {
            return Err(SpoError::InvalidConfig(format!(
                "base_strength must be finite, got {}",
                self.base_strength
            )));
        }
        if !self.decay_alpha.is_finite() || self.decay_alpha < 0.0 {
            return Err(SpoError::InvalidConfig(format!(
                "decay_alpha must be non-negative finite, got {}",
                self.decay_alpha
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_integration_validates() {
        IntegrationConfig::default().validate().unwrap();
    }

    #[test]
    fn zero_dt_rejected() {
        let mut c = IntegrationConfig::default();
        c.dt = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn negative_dt_rejected() {
        let mut c = IntegrationConfig::default();
        c.dt = -0.01;
        assert!(c.validate().is_err());
    }

    #[test]
    fn nan_dt_rejected() {
        let mut c = IntegrationConfig::default();
        c.dt = f64::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_substeps_rejected() {
        let mut c = IntegrationConfig::default();
        c.n_substeps = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn default_coupling_validates() {
        CouplingConfig::default().validate().unwrap();
    }

    #[test]
    fn negative_decay_rejected() {
        let mut c = CouplingConfig::default();
        c.decay_alpha = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn nan_base_strength_rejected() {
        let mut c = CouplingConfig::default();
        c.base_strength = f64::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn method_serde_roundtrip() {
        for method in [Method::Euler, Method::RK4, Method::RK45] {
            let json = serde_json::to_string(&method).unwrap();
            let m: Method = serde_json::from_str(&json).unwrap();
            assert_eq!(m, method);
        }
    }

    #[test]
    fn integration_config_serde() {
        let c = IntegrationConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: IntegrationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.dt, c.dt);
        assert_eq!(c2.method, c.method);
    }
}
