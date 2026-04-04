// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Cellular Sheaf Kuramoto Engine

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};
use crate::dp_tableau as dp;

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
}

impl SheafUPDEStepper {
    pub fn new(n: usize, d: usize, config: IntegrationConfig) -> SpoResult<Self> {
        let size = n * d;
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
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
    ) -> SpoResult<()> {
        let size = self.n * self.d;
        if phases.len() != size || omegas.len() != size || psi.len() != self.d {
            return Err(SpoError::InvalidDimension("Phase/omega/psi size mismatch".into()));
        }
        if restriction_maps.len() != self.n * self.n * self.d * self.d {
            return Err(SpoError::InvalidDimension("Restriction map size mismatch".into()));
        }

        let dt = self.config.dt;
        let n_substeps = self.config.n_substeps.max(1);
        let sub_dt = dt / (n_substeps as f64);

        for _ in 0..n_substeps {
            match self.config.method {
                Method::Euler => {
                    self.euler_step(phases, omegas, restriction_maps, zeta, psi, sub_dt);
                }
                Method::RK4 => {
                    self.rk4_step(phases, omegas, restriction_maps, zeta, psi, sub_dt);
                }
                Method::RK45 => {
                    self.rk45_step(phases, omegas, restriction_maps, zeta, psi);
                }
            }
        }
        wrap_phases(phases);
        Ok(())
    }

    pub fn run(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
        n_steps: u64,
    ) -> SpoResult<()> {
        for _ in 0..n_steps {
            self.step(phases, omegas, restriction_maps, zeta, psi)?;
        }
        Ok(())
    }

    fn euler_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
        dt: f64,
    ) {
        compute_derivative(self.n, self.d, phases, omegas, restriction_maps, zeta, psi, &mut self.k1);
        for i in 0..phases.len() {
            phases[i] += dt * self.k1[i];
        }
    }

    fn rk4_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
        dt: f64,
    ) {
        let size = phases.len();

        compute_derivative(self.n, self.d, phases, omegas, restriction_maps, zeta, psi, &mut self.k1);

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k2);

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k2[i];
        }
        compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k3);

        for i in 0..size {
            self.tmp_phases[i] = phases[i] + dt * self.k3[i];
        }
        compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k4);

        for i in 0..size {
            phases[i] += (dt / 6.0) * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }

    fn rk45_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        restriction_maps: &[f64],
        zeta: f64,
        psi: &[f64],
    ) {
        let mut dt = self.last_dt;
        let mut t_remaining = self.config.dt;
        let size = phases.len();

        while t_remaining > 1e-12 {
            dt = dt.min(t_remaining);

            compute_derivative(self.n, self.d, phases, omegas, restriction_maps, zeta, psi, &mut self.k1);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * dp::A21 * self.k1[i];
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k2);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]);
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k3);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]);
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k4);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A51 * self.k1[i] + dp::A52 * self.k2[i] + dp::A53 * self.k3[i] + dp::A54 * self.k4[i]);
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k5);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A61 * self.k1[i] + dp::A62 * self.k2[i] + dp::A63 * self.k3[i] + dp::A64 * self.k4[i] + dp::A65 * self.k5[i]);
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k6);

            for i in 0..size {
                self.tmp_phases[i] = phases[i] + dt * (dp::A71 * self.k1[i] + dp::A73 * self.k3[i] + dp::A74 * self.k4[i] + dp::A75 * self.k5[i] + dp::A76 * self.k6[i]);
            }
            compute_derivative(self.n, self.d, &self.tmp_phases, omegas, restriction_maps, zeta, psi, &mut self.k7);

            let mut err_sq = 0.0;
            for i in 0..size {
                let err = dt * (
                    (dp::B4[0] - dp::B5[0]) * self.k1[i] +
                    (dp::B4[2] - dp::B5[2]) * self.k3[i] +
                    (dp::B4[3] - dp::B5[3]) * self.k4[i] +
                    (dp::B4[4] - dp::B5[4]) * self.k5[i] +
                    (dp::B4[5] - dp::B5[5]) * self.k6[i] +
                    (dp::B4[6] - dp::B5[6]) * self.k7[i]
                );
                self.err_buf[i] = err;

                let scale = self.config.atol + self.config.rtol * phases[i].abs().max(self.tmp_phases[i].abs());
                let scaled_err = err / scale;
                err_sq += scaled_err * scaled_err;
            }

            let err_norm = (err_sq / size as f64).sqrt();
            let mut dt_next = dt * 0.9 * err_norm.powf(-0.2);
            dt_next = dt_next.clamp(dt * 0.1, dt * 5.0).clamp(1e-9, self.config.dt);

            if err_norm <= 1.0 {
                for i in 0..size {
                    phases[i] = self.tmp_phases[i];
                }
                t_remaining -= dt;
                self.last_dt = dt_next;
            }
            dt = dt_next;
        }
    }
}

fn compute_derivative(
    n: usize,
    d: usize,
    theta: &[f64],
    omegas: &[f64],
    restriction_maps: &[f64],
    zeta: f64,
    psi: &[f64],
    out: &mut [f64],
) {
    for i in 0..n {
        for dim in 0..d {
            let i_idx = i * d + dim;
            let mut coupling_sum = 0.0;

            for j in 0..n {
                for k in 0..d {
                    let j_idx = j * d + k;
                    // B_{ij}^{dim, k} is located at:
                    // matrix_idx = i * (N * D * D) + j * (D * D) + dim * D + k
                    let b_idx = i * n * d * d + j * d * d + dim * d + k;
                    let b_val = restriction_maps[b_idx];

                    if b_val != 0.0 {
                        coupling_sum += b_val * (theta[j_idx] - theta[i_idx]).sin();
                    }
                }
            }

            out[i_idx] = omegas[i_idx] + coupling_sum;

            if zeta != 0.0 {
                out[i_idx] += zeta * (psi[dim] - theta[i_idx]).sin();
            }
        }
    }
}

fn wrap_phases(phases: &mut [f64]) {
    let two_pi = 2.0 * std::f64::consts::PI;
    for p in phases.iter_mut() {
        *p %= two_pi;
        if *p < 0.0 {
            *p += two_pi;
        }
    }
}
