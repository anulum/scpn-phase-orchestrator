// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Imprint model

//!
//! m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure_k * dt,
//! clipped to [0, saturation].

use spo_types::{SpoError, SpoResult};

/// Exponential exposure accumulation with decay and saturation clamping.
#[derive(Debug)]
pub struct ImprintModel {
    pub m: Vec<f64>,
    decay_rate: f64,
    saturation: f64,
}

impl ImprintModel {
    /// # Errors
    /// Returns `InvalidConfig` if decay_rate is negative or saturation is non-positive.
    pub fn new(n: usize, decay_rate: f64, saturation: f64) -> SpoResult<Self> {
        if decay_rate < 0.0 {
            return Err(SpoError::InvalidConfig(format!(
                "decay_rate must be non-negative, got {decay_rate}"
            )));
        }
        if saturation <= 0.0 {
            return Err(SpoError::InvalidConfig(format!(
                "saturation must be positive, got {saturation}"
            )));
        }
        Ok(Self {
            m: vec![0.0; n],
            decay_rate,
            saturation,
        })
    }

    /// Update imprint state: exponential decay + exposure accumulation.
    ///
    /// # Arguments
    ///
    /// * `exposure` - Per-oscillator exposure intensities (length N).
    /// * `dt` - Integration timestep in seconds.
    pub fn update(&mut self, exposure: &[f64], dt: f64) {
        let decay = (-self.decay_rate * dt).exp();
        for (m_k, &e_k) in self.m.iter_mut().zip(exposure.iter()) {
            *m_k = (*m_k * decay + e_k * dt).clamp(0.0, self.saturation);
        }
    }

    /// Scale Knm rows by (1 + m_k). `knm` is row-major N×N.
    ///
    /// # Errors
    /// Returns `InvalidDimension` if `knm.len() != n * n`.
    pub fn modulate_coupling(&self, knm: &mut [f64]) -> SpoResult<()> {
        let n = self.m.len();
        if knm.len() != n * n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {}={n}*{n}, got {}",
                n * n,
                knm.len()
            )));
        }
        for i in 0..n {
            let scale = 1.0 + self.m[i];
            for j in 0..n {
                knm[i * n + j] *= scale;
            }
        }
        Ok(())
    }

    /// Shift alpha lags by imprint magnitude per oscillator.
    ///
    /// # Errors
    /// Returns `InvalidDimension` if `alpha.len() != n * n`.
    pub fn modulate_lag(&self, alpha: &mut [f64]) -> SpoResult<()> {
        let n = self.m.len();
        if alpha.len() != n * n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {}={n}*{n}, got {}",
                n * n,
                alpha.len()
            )));
        }
        for i in 0..n {
            for j in 0..n {
                alpha[i * n + j] += self.m[i] - self.m[j];
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.m.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_decay_rejected() {
        assert!(ImprintModel::new(4, -1.0, 1.0).is_err());
    }

    #[test]
    fn zero_saturation_rejected() {
        assert!(ImprintModel::new(4, 0.1, 0.0).is_err());
    }

    #[test]
    fn initial_imprint_zero() {
        let im = ImprintModel::new(4, 0.1, 1.0).unwrap();
        assert!(im.m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn accumulation() {
        let mut im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        im.update(&[1.0, 2.0], 1.0);
        assert!((im.m[0] - 1.0).abs() < 1e-12);
        assert!((im.m[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn saturation_clamp() {
        let mut im = ImprintModel::new(1, 0.0, 0.5).unwrap();
        im.update(&[10.0], 1.0);
        assert_eq!(im.m[0], 0.5);
    }

    #[test]
    fn decay() {
        let mut im = ImprintModel::new(1, 1.0, 10.0).unwrap();
        im.m[0] = 1.0;
        im.update(&[0.0], 1.0);
        // m = 1.0 * exp(-1) ≈ 0.368
        assert!((im.m[0] - (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn modulate_coupling_effect() {
        let mut im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        im.m[0] = 0.5;
        im.m[1] = 0.0;
        let mut knm = vec![0.0, 1.0, 1.0, 0.0];
        im.modulate_coupling(&mut knm).unwrap();
        assert!((knm[1] - 1.5).abs() < 1e-12);
        assert!((knm[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn modulate_coupling_dimension_mismatch() {
        let im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        let mut knm = vec![1.0; 3];
        assert!(im.modulate_coupling(&mut knm).is_err());
    }

    #[test]
    fn reset_clears() {
        let mut im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        im.update(&[1.0, 1.0], 1.0);
        im.reset();
        assert!(im.m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn modulate_lag_shifts_rows() {
        let mut im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        im.m[0] = 0.3;
        im.m[1] = 0.0;
        // alpha[i,j] += m[i] - m[j]
        // [0,0]: 0.0 + (0.3 - 0.3) = 0.0
        // [0,1]: 1.0 + (0.3 - 0.0) = 1.3
        // [1,0]: -1.0 + (0.0 - 0.3) = -1.3
        // [1,1]: 0.0 + (0.0 - 0.0) = 0.0
        let mut alpha = vec![0.0, 1.0, -1.0, 0.0];
        im.modulate_lag(&mut alpha).unwrap();
        assert!((alpha[0] - 0.0).abs() < 1e-12);
        assert!((alpha[1] - 1.3).abs() < 1e-12);
        assert!((alpha[2] - (-1.3)).abs() < 1e-12);
        assert!((alpha[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn modulate_lag_antisymmetric() {
        let mut im = ImprintModel::new(3, 0.0, 10.0).unwrap();
        im.m[0] = 0.5;
        im.m[1] = 0.2;
        im.m[2] = 0.8;
        let n = 3;
        let mut alpha = vec![0.0; n * n];
        im.modulate_lag(&mut alpha).unwrap();
        for i in 0..n {
            for j in 0..n {
                let sum = alpha[i * n + j] + alpha[j * n + i];
                assert!(sum.abs() < 1e-12, "alpha[{i},{j}] + alpha[{j},{i}] = {sum}");
            }
        }
    }

    #[test]
    fn modulate_lag_dimension_mismatch() {
        let im = ImprintModel::new(2, 0.0, 10.0).unwrap();
        let mut alpha = vec![1.0; 5];
        assert!(im.modulate_lag(&mut alpha).is_err());
    }
}
