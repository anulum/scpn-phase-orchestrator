// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Strang splitting integrator (Sequential Fixed)

use spo_types::{IntegrationConfig, SpoError, SpoResult};
use std::f64::consts::TAU;

pub struct SplittingStepper {
    n: usize,
    dt: f64,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp_p: Vec<f64>,
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
}

impl std::fmt::Debug for SplittingStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplittingStepper")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl SplittingStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;
        Ok(Self {
            n,
            dt: config.dt,
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            tmp_p: vec![0.0; n],
            sin_theta: vec![0.0; n],
            cos_theta: vec![0.0; n],
        })
    }

    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
    ) -> SpoResult<()> {
        let n = self.n;
        let dt = self.dt;
        let half_dt = 0.5 * dt;
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);

        for i in 0..n {
            phases[i] = (phases[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }
        self.rk4_coupling(phases, knm, alpha, zeta, psi, dt, alpha_zero);
        for i in 0..n {
            phases[i] = (phases[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
        n_steps: usize,
    ) -> SpoResult<()> {
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);
        for _ in 0..n_steps {
            self.step_with_alpha(phases, omegas, knm, alpha, zeta, psi, alpha_zero)?;
        }
        Ok(())
    }

    fn step_with_alpha(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
        alpha_zero: bool,
    ) -> SpoResult<()> {
        let n = self.n;
        let dt = self.dt;
        let half_dt = 0.5 * dt;
        for i in 0..n {
            phases[i] = (phases[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }
        self.rk4_coupling(phases, knm, alpha, zeta, psi, dt, alpha_zero);
        for i in 0..n {
            phases[i] = (phases[i] + half_dt * omegas[i]).rem_euclid(TAU);
        }
        Ok(())
    }

    pub fn order_parameter(&self) -> (f64, f64) {
        crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta)
    }

    fn rk4_coupling(
        &mut self,
        p: &mut [f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
        dt: f64,
        alpha_zero: bool,
    ) {
        let n = self.n;
        compute_coupling_deriv(
            n,
            p,
            knm,
            alpha,
            zeta,
            psi,
            alpha_zero,
            &mut self.sin_theta,
            &mut self.cos_theta,
            &mut self.k1,
        );
        for i in 0..n {
            self.tmp_p[i] = (p[i] + 0.5 * dt * self.k1[i]).rem_euclid(TAU);
        }
        compute_coupling_deriv(
            n,
            &self.tmp_p,
            knm,
            alpha,
            zeta,
            psi,
            alpha_zero,
            &mut self.sin_theta,
            &mut self.cos_theta,
            &mut self.k2,
        );
        for i in 0..n {
            self.tmp_p[i] = (p[i] + 0.5 * dt * self.k2[i]).rem_euclid(TAU);
        }
        compute_coupling_deriv(
            n,
            &self.tmp_p,
            knm,
            alpha,
            zeta,
            psi,
            alpha_zero,
            &mut self.sin_theta,
            &mut self.cos_theta,
            &mut self.k3,
        );
        for i in 0..n {
            self.tmp_p[i] = (p[i] + dt * self.k3[i]).rem_euclid(TAU);
        }
        compute_coupling_deriv(
            n,
            &self.tmp_p,
            knm,
            alpha,
            zeta,
            psi,
            alpha_zero,
            &mut self.sin_theta,
            &mut self.cos_theta,
            &mut self.k4,
        );
        let dt6 = dt / 6.0;
        for i in 0..n {
            p[i] = (p[i] + dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]))
                .rem_euclid(TAU);
        }
    }
}

fn compute_coupling_deriv(
    n: usize,
    theta: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    alpha_zero: bool,
    sin_theta: &mut [f64],
    cos_theta: &mut [f64],
    out: &mut [f64],
) {
    for i in 0..n {
        let (s, c) = theta[i].sin_cos();
        sin_theta[i] = s;
        cos_theta[i] = c;
    }
    let (zs_psi, zc_psi) = if zeta != 0.0 {
        let (s, c) = psi.sin_cos();
        (zeta * s, zeta * c)
    } else {
        (0.0, 0.0)
    };
    for i in 0..n {
        let offset = i * n;
        let ci = cos_theta[i];
        let si = sin_theta[i];
        let mut acc = 0.0;
        if alpha_zero {
            for j in 0..n {
                acc += knm[offset + j] * (sin_theta[j] * ci - cos_theta[j] * si);
            }
        } else {
            for j in 0..n {
                acc += knm[offset + j] * (theta[j] - theta[i] - alpha[offset + j]).sin();
            }
        }
        out[i] = acc;
        if zeta != 0.0 {
            out[i] += zs_psi * ci - zc_psi * si;
        }
    }
}

pub fn splitting_run(
    phases: &[f64],
    omegas: &[f64],
    knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let n = phases.len();
    let mut s = SplittingStepper::new(
        n,
        IntegrationConfig {
            dt,
            ..Default::default()
        },
    )
    .expect("stepper init failed");
    let mut p = phases.to_vec();
    s.run(&mut p, omegas, knm, alpha, zeta, psi, n_steps)
        .expect("stepper init failed");
    p
}
