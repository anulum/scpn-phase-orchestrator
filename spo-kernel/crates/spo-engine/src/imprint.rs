// SCPN Phase Orchestrator — Imprint Model
//!
//! m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure_k * dt,
//! clipped to [0, saturation].

use spo_types::{SpoError, SpoResult};

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
    pub fn update(&mut self, exposure: &[f64], dt: f64) {
        let decay = (-self.decay_rate * dt).exp();
        for (m_k, &e_k) in self.m.iter_mut().zip(exposure.iter()) {
            *m_k = (*m_k * decay + e_k * dt).clamp(0.0, self.saturation);
        }
    }

    /// Scale Knm rows by (1 + m_k). `knm` is row-major N×N.
    pub fn modulate_coupling(&self, knm: &mut [f64]) {
        let n = self.m.len();
        for i in 0..n {
            let scale = 1.0 + self.m[i];
            for j in 0..n {
                knm[i * n + j] *= scale;
            }
        }
    }

    /// Shift alpha lags by imprint magnitude per oscillator.
    pub fn modulate_lag(&self, alpha: &mut [f64]) {
        let n = self.m.len();
        for i in 0..n {
            for j in 0..n {
                alpha[i * n + j] += self.m[i];
            }
        }
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
        im.modulate_coupling(&mut knm);
        // Row 0 scaled by 1.5
        assert!((knm[1] - 1.5).abs() < 1e-12);
        // Row 1 unscaled
        assert!((knm[2] - 1.0).abs() < 1e-12);
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
        // 2×2 alpha, row-major
        let mut alpha = vec![0.0, 1.0, -1.0, 0.0];
        im.modulate_lag(&mut alpha);
        // Row 0 shifted by m[0]=0.3
        assert!((alpha[0] - 0.3).abs() < 1e-12);
        assert!((alpha[1] - 1.3).abs() < 1e-12);
        // Row 1 unshifted (m[1]=0.0)
        assert!((alpha[2] - (-1.0)).abs() < 1e-12);
        assert!((alpha[3] - 0.0).abs() < 1e-12);
    }
}
