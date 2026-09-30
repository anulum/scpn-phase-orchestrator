// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Cellular Sheaf Kuramoto Engine

use crate::dp_tableau as dp;
use rayon::prelude::*;
use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};

/// Sheaf UPDE Stepper for multi-dimensional phase vectors.
///
/// Phase per oscillator is a vector of dimension D.
/// Restriction maps (coupling blocks) B_ij are D x D matrices mapping
/// the phase space of oscillator j into the space of oscillator i.
///
/// d(theta_{i,d})/dt = omega_{i,d}
///                     + sum_j sum_k B_ij^{dk} sin(theta_{j,k} - theta_{i,d})
///                     + zeta * sin(Psi_d - theta_{i,d})
#[derive(Debug, Clone)]
pub struct SheafUPDEStepper {
    n: usize,
    d: usize,
    config: IntegrationConfig,
    tmp_phases: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    k5: Vec<f64>,
    k6: Vec<f64>,
    k7: Vec<f64>,
    err_buf: Vec<f64>,
    last_dt: f64,
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
    sin_psi: Vec<f64>,
    cos_psi: Vec<f64>,
}

impl SheafUPDEStepper {
    /// # Errors
    /// Rejects zero/overflowing geometry, invalid controls or an unrepresentable substep.
    pub fn new(n: usize, d: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 || d == 0 {
            return Err(SpoError::InvalidDimension(
                "n and d must both be > 0".into(),
            ));
        }
        config.validate()?;
        if config.dt / f64::from(config.n_substeps) <= 0.0 {
            return Err(SpoError::InvalidConfig(
                "sheaf substep must be positive".into(),
            ));
        }
        let size = n
            .checked_mul(d)
            .filter(|size| size.checked_mul(*size).is_some())
            .ok_or_else(|| SpoError::InvalidDimension("sheaf geometry overflows usize".into()))?;
        let last_dt = config.dt;
        Ok(Self {
            n,
            d,
            config,
            tmp_phases: vec![0.0; size],
            k1: vec![0.0; size],
            k2: vec![0.0; size],
            k3: vec![0.0; size],
            k4: vec![0.0; size],
            k5: vec![0.0; size],
            k6: vec![0.0; size],
            k7: vec![0.0; size],
            err_buf: vec![0.0; size],
            last_dt,
            sin_theta: vec![0.0; size],
            cos_theta: vec![0.0; size],
            sin_psi: vec![0.0; d],
            cos_psi: vec![0.0; d],
        })
    }

    /// Return the configured oscillator count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the phase-vector dimension per oscillator.
    #[must_use]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Return the next adaptive substep proposal, or configured `dt` otherwise.
    #[must_use]
    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    /// # Errors
    /// Returns `InvalidDimension` on shape mismatch or `IntegrationDiverged` on
    /// non-finite input/output or unsuccessful adaptive integration. Rounded upper
    /// torus endpoints and signed zero canonicalise to positive zero.
    /// Phases and the published timestep remain unchanged on refusal.
    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
    ) -> SpoResult<()> {
        self.validate_inputs(phases, omegas, restriction_maps, zeta, psi)?;
        let previous_dt = self.last_dt;
        let mut candidate = phases.to_vec();
        match self.advance(&mut candidate, omegas, restriction_maps, zeta, psi) {
            Ok(()) => {
                phases.copy_from_slice(&candidate);
                Ok(())
            }
            Err(error) => {
                self.last_dt = previous_dt;
                Err(error)
            }
        }
    }

    /// Validate full sheaf geometry and all finite integration inputs.
    fn validate_inputs(
        &self,
        phases: &[f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
    ) -> SpoResult<()> {
        let size = self.n * self.d;
        if phases.len() != size || omegas.len() != size || psi.len() != self.d {
            return Err(SpoError::InvalidDimension(
                "Phase/omega/psi size mismatch".into(),
            ));
        }
        if restriction_maps.len() != self.n * self.n * self.d * self.d {
            return Err(SpoError::InvalidDimension(
                "Restriction map size mismatch".into(),
            ));
        }
        if phases.iter().any(|v| !v.is_finite())
            || omegas.iter().any(|v| !v.is_finite())
            || restriction_maps.iter().any(|v| !v.is_finite())
            || psi.iter().any(|v| !v.is_finite())
            || !zeta.is_finite()
        {
            return Err(SpoError::IntegrationDiverged(
                "sheaf UPDE inputs contain NaN/Inf".into(),
            ));
        }

        Ok(())
    }

    /// Advance a private candidate state; public calls publish only on success.
    fn advance(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
    ) -> SpoResult<()> {
        let dt = self.config.dt;
        let n_substeps = self.config.n_substeps.max(1);
        let sub_dt = dt / (n_substeps as f64);

        if zeta != 0.0 {
            for i in 0..self.d {
                let (s, c) = psi[i].sin_cos();
                self.sin_psi[i] = s;
                self.cos_psi[i] = c;
            }
        }

        for _ in 0..n_substeps {
            match self.config.method {
                Method::Euler => {
                    self.euler_step(phases, omegas, restriction_maps, zeta, psi, sub_dt);
                }
                Method::RK4 => {
                    self.rk4_step(phases, omegas, restriction_maps, zeta, psi, sub_dt);
                }
                Method::RK45 => {
                    self.rk45_step(phases, omegas, restriction_maps, zeta, psi, sub_dt)?;
                }
            }
        }
        wrap_phases(phases);
        if phases
            .iter()
            .any(|phase| !phase.is_finite() || *phase < 0.0 || *phase >= std::f64::consts::TAU)
        {
            return Err(SpoError::IntegrationDiverged(
                "sheaf output outside finite torus".into(),
            ));
        }
        Ok(())
    }

    /// # Errors
    /// Validates inputs even for zero steps and propagates integration errors.
    /// The entire call preserves phases and its entry timestep on refusal.
    pub fn run(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
        n_steps: u64,
    ) -> SpoResult<()> {
        self.validate_inputs(phases, omegas, restriction_maps, zeta, psi)?;
        let mut candidate = phases.to_vec();
        let previous_dt = self.last_dt;
        for _ in 0..n_steps {
            if let Err(error) = self.advance(&mut candidate, omegas, restriction_maps, zeta, psi) {
                self.last_dt = previous_dt;
                return Err(error);
            }
        }
        phases.copy_from_slice(&candidate);
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn euler_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        #[allow(unused_variables)] psi: &[f64],
        dt: f64,
    ) {
        compute_derivative(
            self.n,
            self.d,
            phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            restriction_maps,
            zeta,
            &self.sin_psi,
            &self.cos_psi,
            &mut self.k1,
        );
        for i in 0..phases.len() {
            phases[i] += dt * self.k1[i];
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn rk4_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        #[allow(unused_variables)] psi: &[f64],
        dt: f64,
    ) {
        let size = phases.len();

        compute_derivative(
            self.n,
            self.d,
            phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            restriction_maps,
            zeta,
            &self.sin_psi,
            &self.cos_psi,
            &mut self.k1,
        );

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(
            self.n,
            self.d,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            restriction_maps,
            zeta,
            &self.sin_psi,
            &self.cos_psi,
            &mut self.k2,
        );

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k2[i];
        }
        compute_derivative(
            self.n,
            self.d,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            restriction_maps,
            zeta,
            &self.sin_psi,
            &self.cos_psi,
            &mut self.k3,
        );

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + dt * self.k3[i];
        }
        compute_derivative(
            self.n,
            self.d,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            restriction_maps,
            zeta,
            &self.sin_psi,
            &self.cos_psi,
            &mut self.k4,
        );

        for i in 0..size {
            phases[i] +=
                (dt / 6.0) * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn rk45_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        #[allow(unused_variables)] psi: &[f64],
        horizon: f64,
    ) -> SpoResult<()> {
        let mut dt = self.last_dt;
        let mut t_remaining = horizon;
        let size = phases.len();
        let mut rejects = 0;

        for _ in 0..100_000 {
            dt = dt.min(t_remaining);

            compute_derivative(
                self.n,
                self.d,
                phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k1,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * dp::A21 * self.k1[i];
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k2,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]);
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k3,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]);
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k4,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A51 * self.k1[i]
                        + dp::A52 * self.k2[i]
                        + dp::A53 * self.k3[i]
                        + dp::A54 * self.k4[i]);
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k5,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A61 * self.k1[i]
                        + dp::A62 * self.k2[i]
                        + dp::A63 * self.k3[i]
                        + dp::A64 * self.k4[i]
                        + dp::A65 * self.k5[i]);
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k6,
            );

            for i in 0..size {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A71 * self.k1[i]
                        + dp::A73 * self.k3[i]
                        + dp::A74 * self.k4[i]
                        + dp::A75 * self.k5[i]
                        + dp::A76 * self.k6[i]);
            }
            compute_derivative(
                self.n,
                self.d,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                restriction_maps,
                zeta,
                &self.sin_psi,
                &self.cos_psi,
                &mut self.k7,
            );

            let mut err_norm: f64 = 0.0;
            for i in 0..size {
                let err = dt
                    * ((dp::B4[0] - dp::B5[0]) * self.k1[i]
                        + (dp::B4[2] - dp::B5[2]) * self.k3[i]
                        + (dp::B4[3] - dp::B5[3]) * self.k4[i]
                        + (dp::B4[4] - dp::B5[4]) * self.k5[i]
                        + (dp::B4[5] - dp::B5[5]) * self.k6[i]
                        + (dp::B4[6] - dp::B5[6]) * self.k7[i]);
                self.err_buf[i] = err;

                let scale = self.config.atol
                    + self.config.rtol * phases[i].abs().max(self.tmp_phases[i].abs());
                let scaled_err = err / scale;
                if !scaled_err.is_finite() || !self.tmp_phases[i].is_finite() || !scale.is_finite()
                {
                    return Err(SpoError::IntegrationDiverged(
                        "sheaf RK45 non-finite arithmetic".into(),
                    ));
                }
                err_norm = err_norm.max(scaled_err.abs());
            }

            let factor = if err_norm > 0.0 {
                (0.9 * err_norm.powf(-0.2)).clamp(0.2, 5.0)
            } else {
                5.0
            };
            let dt_next = (dt * factor).min(self.config.dt);
            if !dt_next.is_finite() || dt_next <= 0.0 {
                return Err(SpoError::IntegrationDiverged(
                    "sheaf RK45 timestep cannot advance".into(),
                ));
            }

            if err_norm <= 1.0 {
                phases[..size].copy_from_slice(&self.tmp_phases[..size]);
                let next_remaining = t_remaining - dt;
                if next_remaining == t_remaining {
                    return Err(SpoError::IntegrationDiverged(
                        "sheaf RK45 timestep cannot advance".into(),
                    ));
                }
                t_remaining = next_remaining;
                self.last_dt = dt_next;
                rejects = 0;
                if t_remaining <= 0.0 {
                    return Ok(());
                }
            } else {
                rejects += 1;
                if rejects >= 64 {
                    return Err(SpoError::IntegrationDiverged(
                        "sheaf RK45 rejection limit exceeded".into(),
                    ));
                }
            }
            dt = dt_next;
        }
        Err(SpoError::IntegrationDiverged(
            "sheaf RK45 substep limit exceeded".into(),
        ))
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn compute_derivative(
    n: usize,
    d: usize,
    theta: &[f64],
    sin_theta: &mut [f64],
    cos_theta: &mut [f64],
    omegas: &[f64],
    restriction_maps: &[f64],
    zeta: f64,
    sin_psi: &[f64],
    cos_psi: &[f64],
    out: &mut [f64],
) {
    let size = n * d;
    for i in 0..size {
        let (s, c) = theta[i].sin_cos();
        sin_theta[i] = s;
        cos_theta[i] = c;
    }

    let st = &*sin_theta;
    let ct = &*cos_theta;

    out.par_chunks_mut(d).enumerate().for_each(|(i, out_i)| {
        for dim in 0..d {
            let i_idx = i * d + dim;
            let ci = ct[i_idx];
            let si = st[i_idx];
            let mut coupling_sum = 0.0;

            let row_offset = i * n * d * d + dim * d;
            for j in 0..n {
                let block_offset = row_offset + j * d * d;
                for k in 0..d {
                    let j_idx = j * d + k;
                    let b_val = restriction_maps[block_offset + k];
                    if b_val != 0.0 {
                        // sin(tj - ti) = sj*ci - cj*si
                        coupling_sum += b_val * (st[j_idx] * ci - ct[j_idx] * si);
                    }
                }
            }

            out_i[dim] = omegas[i_idx] + coupling_sum;
            if zeta != 0.0 {
                out_i[dim] += zeta * (sin_psi[dim] * ci - cos_psi[dim] * si);
            }
        }
    });
}

/// Canonicalise finite torus values, including a rounded upper endpoint.
fn wrap_phases(phases: &mut [f64]) {
    let two_pi = 2.0 * std::f64::consts::PI;
    for p in phases.iter_mut() {
        *p %= two_pi;
        if *p < 0.0 {
            *p += two_pi;
        }
        if *p >= two_pi || *p == 0.0 {
            *p = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_and_substeps_preserve_outer_horizon() {
        for method in [Method::Euler, Method::RK4, Method::RK45] {
            for n_substeps in [1, 3] {
                let mut stepper = SheafUPDEStepper::new(
                    1,
                    2,
                    IntegrationConfig {
                        method,
                        n_substeps,
                        dt: 0.01,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut phases = [0.2, 0.5];
                stepper
                    .run(&mut phases, &[1.0, -0.5], &[0.0; 4], 0.0, &[0.0; 2], 7)
                    .unwrap();
                assert!((phases[0] - 0.27).abs() < 2e-15);
                assert!((phases[1] - 0.465).abs() < 2e-15);
                assert_eq!((stepper.n(), stepper.d()), (1, 2));
                assert!(stepper.last_dt() > 0.0 && stepper.last_dt() <= 0.01);
            }
        }
    }

    #[test]
    fn adaptive_forcing_matches_closed_form_relaxation() {
        let mut stepper = SheafUPDEStepper::new(
            1,
            1,
            IntegrationConfig {
                method: Method::RK45,
                dt: 0.4,
                atol: 1e-12,
                rtol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        let mut phases = [0.1];
        stepper
            .run(&mut phases, &[0.0], &[0.0], 2.0, &[1.2], 3)
            .unwrap();
        let expected = 1.2 - 2.0 * (((1.2_f64 - 0.1) / 2.0).tan() * (-2.0_f64 * 1.2).exp()).atan();
        assert!((phases[0] - expected).abs() < 2e-11);
        assert!(stepper.last_dt() > 0.0 && stepper.last_dt() < 0.4);
    }

    #[test]
    fn adaptive_tiny_interval_still_advances() {
        let mut stepper = SheafUPDEStepper::new(
            1,
            1,
            IntegrationConfig {
                method: Method::RK45,
                dt: 1e-14,
                ..Default::default()
            },
        )
        .unwrap();
        let mut phases = [0.0];
        stepper
            .run(&mut phases, &[1.0], &[0.0], 0.0, &[0.0], 3)
            .unwrap();
        assert!((phases[0] - 3e-14).abs() < 1e-28);
    }

    #[test]
    fn unusable_geometry_and_configurations_refuse() {
        assert!(SheafUPDEStepper::new(0, 1, IntegrationConfig::default()).is_err());
        assert!(SheafUPDEStepper::new(1, 0, IntegrationConfig::default()).is_err());
        assert!(SheafUPDEStepper::new(usize::MAX, 2, IntegrationConfig::default()).is_err());
        assert!(SheafUPDEStepper::new(usize::MAX, 1, IntegrationConfig::default()).is_err());
        assert!(SheafUPDEStepper::new(
            1,
            1,
            IntegrationConfig {
                dt: 0.0,
                ..Default::default()
            }
        )
        .is_err());
        assert!(SheafUPDEStepper::new(
            1,
            1,
            IntegrationConfig {
                dt: f64::from_bits(1),
                n_substeps: 2,
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn zero_batch_validates_shapes_and_every_numeric_input() {
        let mut stepper = SheafUPDEStepper::new(1, 1, IntegrationConfig::default()).unwrap();
        for (omega, maps, zeta, psi) in [
            (vec![], vec![0.0], 0.0, vec![0.0]),
            (vec![0.0], vec![], 0.0, vec![0.0]),
            (vec![0.0], vec![0.0], 0.0, vec![]),
            (vec![f64::NAN], vec![0.0], 0.0, vec![0.0]),
            (vec![0.0], vec![f64::INFINITY], 0.0, vec![0.0]),
            (vec![0.0], vec![0.0], f64::INFINITY, vec![0.0]),
            (vec![0.0], vec![0.0], 0.0, vec![f64::NAN]),
        ] {
            let mut phases = [-0.4];
            assert!(stepper
                .run(&mut phases, &omega, &maps, zeta, &psi, 0)
                .is_err());
            assert_eq!(phases, [-0.4]);
            assert_eq!(stepper.last_dt(), 0.01);
        }
        let mut invalid = [f64::NAN];
        assert!(stepper
            .run(&mut invalid, &[0.0], &[0.0], 0.0, &[0.0], 0)
            .is_err());
        let mut phases = [-0.4];
        stepper
            .run(&mut phases, &[0.0], &[0.0], 0.0, &[0.0], 0)
            .unwrap();
        assert_eq!(phases, [-0.4]);
    }

    #[test]
    fn numerical_refusal_preserves_phase_and_timestep_and_recovers() {
        for method in [Method::Euler, Method::RK4, Method::RK45] {
            let mut stepper = SheafUPDEStepper::new(
                1,
                1,
                IntegrationConfig {
                    method,
                    dt: 10.0,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut phases = [0.0];
            assert!(stepper
                .step(&mut phases, &[1e308], &[0.0], 0.0, &[0.0])
                .is_err());
            assert_eq!(phases, [0.0]);
            assert_eq!(stepper.last_dt(), 10.0);
            assert!(stepper
                .run(&mut phases, &[1e308], &[0.0], 0.0, &[0.0], 2)
                .is_err());
            assert_eq!(phases, [0.0]);
            stepper
                .run(&mut phases, &[0.1], &[0.0], 0.0, &[0.0], 2)
                .unwrap();
            assert!((phases[0] - 2.0).abs() < 1e-14);
        }
    }

    #[test]
    fn later_batch_failure_restores_original_phase() {
        let angle = 1e308_f64.rem_euclid(std::f64::consts::TAU) + std::f64::consts::FRAC_PI_2;
        let mut stepper = SheafUPDEStepper::new(
            1,
            1,
            IntegrationConfig {
                dt: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut first = [angle];
        stepper
            .step(&mut first, &[1e308], &[0.0], 1e308, &[angle])
            .unwrap();
        assert!((first[0] - 1e308_f64.rem_euclid(std::f64::consts::TAU)).abs() < 1e-15);
        let mut phases = [angle];
        assert!(stepper
            .run(&mut phases, &[1e308], &[0.0], 1e308, &[angle], 2)
            .is_err());
        assert_eq!(phases, [angle]);
        assert_eq!(stepper.last_dt(), 1.0);
    }

    #[test]
    fn wrapping_canonicalises_rounding_and_negative_crossings() {
        for method in [Method::Euler, Method::RK4, Method::RK45] {
            let mut stepper = SheafUPDEStepper::new(
                1,
                1,
                IntegrationConfig {
                    method,
                    ..Default::default()
                },
            )
            .unwrap();
            for (phase, omega) in [
                (-1e-300, 0.0),
                (0.0, -1e-15),
                (-std::f64::consts::TAU, 0.0),
                (-2.0 * std::f64::consts::TAU, 0.0),
            ] {
                let mut phases = [phase];
                stepper
                    .step(&mut phases, &[omega], &[0.0], 0.0, &[0.0])
                    .unwrap();
                assert_eq!(phases, [0.0]);
                assert!(!phases[0].is_sign_negative());
            }
        }
        let mut stepper = SheafUPDEStepper::new(1, 1, IntegrationConfig::default()).unwrap();
        let mut valid = [-1.0];
        stepper
            .step(&mut valid, &[0.0], &[0.0], 0.0, &[0.0])
            .unwrap();
        assert!((valid[0] - (std::f64::consts::TAU - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn sheaf_stepper_reports_geometry_and_timestep() {
        let stepper = SheafUPDEStepper::new(
            2,
            3,
            IntegrationConfig {
                dt: 0.05,
                method: Method::Euler,
                ..Default::default()
            },
        )
        .expect("stepper init failed");

        assert_eq!(stepper.n(), 2);
        assert_eq!(stepper.d(), 3);
        assert_eq!(stepper.last_dt(), 0.05);
    }

    #[test]
    fn sheaf_stepper_rejects_mismatched_state_vectors() {
        let mut stepper =
            SheafUPDEStepper::new(2, 2, IntegrationConfig::default()).expect("stepper init failed");
        let mut phases = vec![0.0; 3];
        let omegas = vec![0.0; 4];
        let restriction_maps = vec![0.0; 16];
        let psi = vec![0.0; 2];

        assert!(matches!(
            stepper.step(&mut phases, &omegas, &restriction_maps, 0.0, &psi),
            Err(SpoError::InvalidDimension(_))
        ));
    }
}
