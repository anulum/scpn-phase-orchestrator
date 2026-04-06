// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// SCPN Phase Orchestrator — Sparse UPDE solver

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};
use crate::dp_tableau as dp;
use crate::plasticity::PlasticityModel;

pub struct SparseUPDEStepper {
    n: usize,
    dt: f64,
    n_substeps: u32,
    method: Method,
    atol: f64,
    rtol: f64,
    last_dt: f64,
    deriv_buf: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    k5: Vec<f64>,
    k6: Vec<f64>,
    k7: Vec<f64>,
    y5: Vec<f64>,
    tmp_phases: Vec<f64>,
    pub sin_theta: Vec<f64>,
    pub cos_theta: Vec<f64>,
    pub plasticity: Option<PlasticityModel>,
    pub modulator: f64,
}

impl std::fmt::Debug for SparseUPDEStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseUPDEStepper").field("n", &self.n).finish_non_exhaustive()
    }
}

impl SparseUPDEStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 { return Err(SpoError::InvalidDimension("n must be > 0".into())); }
        config.validate()?;
        Ok(Self {
            n, dt: config.dt, n_substeps: config.n_substeps, method: config.method,
            atol: config.atol, rtol: config.rtol, last_dt: config.dt,
            deriv_buf: vec![0.0; n], k1: vec![0.0; n], k2: vec![0.0; n], k3: vec![0.0; n],
            k4: vec![0.0; n], k5: vec![0.0; n], k6: vec![0.0; n], k7: vec![0.0; n],
            y5: vec![0.0; n], tmp_phases: vec![0.0; n], sin_theta: vec![0.0; n], cos_theta: vec![0.0; n],
            plasticity: None, modulator: 1.0,
        })
    }

    pub fn step(&mut self, phases: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64]) -> SpoResult<()> {
        let n = self.n;
        let alpha_zero = alpha_values.iter().all(|&a| a == 0.0);
        match self.method {
            Method::RK45 => { self.rk45_step(phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero); }
            _ => {
                let sub_dt = self.dt / f64::from(self.n_substeps);
                for _ in 0..self.n_substeps {
                    match self.method {
                        Method::Euler => { self.euler_step(phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, sub_dt, alpha_zero); }
                        Method::RK4 => { self.rk4_step(phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, sub_dt, alpha_zero); }
                        Method::RK45 => unreachable!(),
                    }
                }
            }
        }
        if let Some(ref plast) = self.plasticity {
            plast.update_sparse(&self.sin_theta, &self.cos_theta, row_ptr, col_indices, knm_values, self.modulator, self.dt);
        }
        wrap_phases(phases);
        Ok(())
    }

    pub fn run(&mut self, phases: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64], n_steps: u64) -> SpoResult<()> {
        for _ in 0..n_steps { self.step(phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values)?; }
        Ok(())
    }

    pub fn n(&self) -> usize { self.n }
    pub fn last_dt(&self) -> f64 { self.last_dt }

    fn euler_step(&mut self, phases: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64], dt: f64, alpha_zero: bool) {
        compute_derivative(self.n, phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.deriv_buf);
        for i in 0..self.n { phases[i] += dt * self.deriv_buf[i]; }
    }

    fn rk4_step(&mut self, phases: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64], dt: f64, alpha_zero: bool) {
        let n = self.n;
        compute_derivative(n, phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k1);
        for i in 0..n { self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i]; }
        compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k2);
        for i in 0..n { self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k2[i]; }
        compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k3);
        for i in 0..n { self.tmp_phases[i] = phases[i] + dt * self.k3[i]; }
        compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k4);
        let dt6 = dt / 6.0;
        for i in 0..n { phases[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]); }
    }

    fn rk45_step(&mut self, phases: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64], alpha_zero: bool) {
        let n = self.n;
        let mut dt = self.last_dt;
        for _ in 0..=3 {
            compute_derivative(n, phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k1);
            for i in 0..n { self.tmp_phases[i] = phases[i] + dt * dp::A21 * self.k1[i]; }
            compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k2);
            for i in 0..n { self.tmp_phases[i] = phases[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]); }
            compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k3);
            for i in 0..n { self.tmp_phases[i] = phases[i] + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]); }
            compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k4);
            for i in 0..n { self.tmp_phases[i] = phases[i] + dt * (dp::A51 * self.k1[i] + dp::A52 * self.k2[i] + dp::A53 * self.k3[i] + dp::A54 * self.k4[i]); }
            compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k5);
            for i in 0..n { self.tmp_phases[i] = phases[i] + dt * (dp::A61 * self.k1[i] + dp::A62 * self.k2[i] + dp::A63 * self.k3[i] + dp::A64 * self.k4[i] + dp::A65 * self.k5[i]); }
            compute_derivative(n, &self.tmp_phases, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k6);
            for i in 0..n { self.y5[i] = phases[i] + dt * (dp::B5[0] * self.k1[i] + dp::B5[2] * self.k3[i] + dp::B5[3] * self.k4[i] + dp::B5[4] * self.k5[i] + dp::B5[5] * self.k6[i]); }
            compute_derivative(n, &self.y5, &mut self.sin_theta, &mut self.cos_theta, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values, alpha_zero, &mut self.k7);
            let mut err_norm: f64 = 0.0;
            for i in 0..n {
                let y4 = phases[i] + dt * (dp::B4[0] * self.k1[i] + dp::B4[2] * self.k3[i] + dp::B4[3] * self.k4[i] + dp::B4[4] * self.k5[i] + dp::B4[5] * self.k6[i] + dp::B4[6] * self.k7[i]);
                let err_i = (self.y5[i] - y4).abs();
                let scale = self.atol + self.rtol * phases[i].abs().max(self.y5[i].abs());
                let ratio = err_i / scale;
                if ratio > err_norm { err_norm = ratio; }
            }
            if err_norm <= 1.0 {
                let factor = if err_norm > 0.0 { (0.9 * err_norm.powf(-0.2)).min(5.0) } else { 5.0 };
                self.last_dt = (dt * factor).min(self.dt * 10.0);
                phases.copy_from_slice(&self.y5[..n]);
                return;
            }
            let factor = (0.9 * err_norm.powf(-0.25)).max(0.2);
            dt *= factor;
        }
        self.last_dt = dt;
        phases.copy_from_slice(&self.y5[..n]);
    }
}

fn compute_derivative(n: usize, theta: &[f64], sin_theta: &mut [f64], cos_theta: &mut [f64], omegas: &[f64], row_ptr: &[usize], col_indices: &[usize], knm_values: &mut [f64], zeta: f64, psi: f64, alpha_values: &[f64], alpha_zero: bool, out: &mut [f64]) {
    for i in 0..n { let (s, c) = theta[i].sin_cos(); sin_theta[i] = s; cos_theta[i] = c; }
    let (s_psi, c_psi) = if zeta != 0.0 { psi.sin_cos() } else { (0.0, 0.0) };
    if alpha_zero {
        for i in 0..n {
            let mut coupling_sum = 0.0;
            let start = row_ptr[i]; let end = row_ptr[i+1];
            let ci = cos_theta[i]; let si = sin_theta[i];
            for idx in start..end {
                let j = col_indices[idx];
                coupling_sum += knm_values[idx] * (sin_theta[j] * ci - cos_theta[j] * si);
            }
            out[i] = omegas[i] + coupling_sum;
            if zeta != 0.0 { out[i] += zeta * (s_psi * ci - c_psi * si); }
        }
    } else {
        for i in 0..n {
            let mut coupling_sum = 0.0;
            let start = row_ptr[i]; let end = row_ptr[i+1];
            let ci = cos_theta[i]; let si = sin_theta[i];
            for idx in start..end {
                let j = col_indices[idx];
                coupling_sum += knm_values[idx] * (theta[j] - theta[i] - alpha_values[idx]).sin();
            }
            out[i] = omegas[i] + coupling_sum;
            if zeta != 0.0 { out[i] += zeta * (s_psi * ci - c_psi * si); }
        }
    }
}

fn wrap_phases(phases: &mut [f64]) {
    for p in phases.iter_mut() { *p = p.rem_euclid(std::f64::consts::TAU); }
}
