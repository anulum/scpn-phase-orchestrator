// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — spatial coupling modulation

use spo_types::{SpoError, SpoResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDecayForm {
    InversePlusOne,
    Exponential,
    PowerLaw,
    InverseDistance,
}

impl SpatialDecayForm {
    pub fn from_code(code: i32) -> SpoResult<Self> {
        match code {
            0 => Ok(Self::InversePlusOne),
            1 => Ok(Self::Exponential),
            2 => Ok(Self::PowerLaw),
            3 => Ok(Self::InverseDistance),
            _ => Err(SpoError::InvalidConfig(format!(
                "unknown spatial decay form code {code}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialCouplingModulator {
    pub k_base: f64,
    pub decay_form: SpatialDecayForm,
    pub decay_exponent: f64,
    pub decay_length_scale: f64,
    pub epsilon: f64,
}

impl SpatialCouplingModulator {
    pub fn new(
        k_base: f64,
        decay_form: SpatialDecayForm,
        decay_exponent: f64,
        decay_length_scale: f64,
        epsilon: f64,
    ) -> SpoResult<Self> {
        if !k_base.is_finite() || k_base < 0.0 {
            return Err(SpoError::InvalidConfig(
                "k_base must be finite and non-negative".into(),
            ));
        }
        if !decay_exponent.is_finite() || decay_exponent <= 0.0 {
            return Err(SpoError::InvalidConfig(
                "decay_exponent must be finite and positive".into(),
            ));
        }
        if !decay_length_scale.is_finite() || decay_length_scale <= 0.0 {
            return Err(SpoError::InvalidConfig(
                "decay_length_scale must be finite and positive".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(SpoError::InvalidConfig(
                "epsilon must be finite and positive".into(),
            ));
        }
        Ok(Self {
            k_base,
            decay_form,
            decay_exponent,
            decay_length_scale,
            epsilon,
        })
    }

    pub fn weight(&self, distance: f64) -> f64 {
        match self.decay_form {
            SpatialDecayForm::InversePlusOne => self.k_base / (1.0 + distance),
            SpatialDecayForm::Exponential => {
                self.k_base * (-distance / self.decay_length_scale).exp()
            }
            SpatialDecayForm::PowerLaw => {
                self.k_base * (1.0 + distance / self.decay_length_scale).powf(-self.decay_exponent)
            }
            SpatialDecayForm::InverseDistance => {
                self.k_base / (distance * distance + self.epsilon).sqrt()
            }
        }
    }

    pub fn modulate(
        &self,
        knm: &[f64],
        positions: &[f64],
        n: usize,
        dim: usize,
    ) -> SpoResult<Vec<f64>> {
        spatial_modulate_flat(
            knm,
            positions,
            n,
            dim,
            self.k_base,
            self.decay_form,
            self.decay_exponent,
            self.decay_length_scale,
            self.epsilon,
        )
    }
}

pub fn spatial_modulate_flat(
    knm: &[f64],
    positions: &[f64],
    n: usize,
    dim: usize,
    k_base: f64,
    decay_form: SpatialDecayForm,
    decay_exponent: f64,
    decay_length_scale: f64,
    epsilon: f64,
) -> SpoResult<Vec<f64>> {
    if n == 0 || dim == 0 {
        return Err(SpoError::InvalidDimension(
            "n and dim must be positive".into(),
        ));
    }
    let expected_k = n
        .checked_mul(n)
        .ok_or_else(|| SpoError::InvalidDimension("n*n overflows usize".into()))?;
    let expected_pos = n
        .checked_mul(dim)
        .ok_or_else(|| SpoError::InvalidDimension("n*dim overflows usize".into()))?;
    if knm.len() != expected_k || positions.len() != expected_pos {
        return Err(SpoError::InvalidDimension(format!(
            "expected knm={expected_k}, positions={expected_pos}; got knm={}, positions={}",
            knm.len(),
            positions.len()
        )));
    }
    if knm.iter().any(|value| !value.is_finite())
        || positions.iter().any(|value| !value.is_finite())
    {
        return Err(SpoError::IntegrationDiverged(
            "spatial modulator inputs contain NaN/Inf".into(),
        ));
    }
    for i in 0..n {
        if knm[i * n + i].abs() > 1e-12 {
            return Err(SpoError::InvalidConfig("knm diagonal must be zero".into()));
        }
    }
    let model = SpatialCouplingModulator::new(
        k_base,
        decay_form,
        decay_exponent,
        decay_length_scale,
        epsilon,
    )?;
    let mut out = vec![0.0; expected_k];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            if i == j {
                out[idx] = 0.0;
                continue;
            }
            let mut d2 = 0.0;
            for d in 0..dim {
                let delta = positions[i * dim + d] - positions[j * dim + d];
                d2 += delta * delta;
            }
            let distance = d2.sqrt();
            let weight = model.weight(distance);
            if !weight.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "spatial modulation weight diverged".into(),
                ));
            }
            out[idx] = knm[idx] * weight;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_plus_one_matches_two_body_reference() {
        let knm = vec![0.0, 2.0, 2.0, 0.0];
        let positions = vec![0.0, 3.0];
        let out = spatial_modulate_flat(
            &knm,
            &positions,
            2,
            1,
            0.5,
            SpatialDecayForm::InversePlusOne,
            1.0,
            1.0,
            1e-12,
        )
        .unwrap();
        assert!((out[1] - 0.25).abs() < 1e-12);
        assert!((out[2] - 0.25).abs() < 1e-12);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn rejects_self_coupling() {
        let err = spatial_modulate_flat(
            &[1.0],
            &[0.0],
            1,
            1,
            1.0,
            SpatialDecayForm::InversePlusOne,
            1.0,
            1.0,
            1e-12,
        )
        .unwrap_err();
        assert!(err.to_string().contains("diagonal"));
    }
}
