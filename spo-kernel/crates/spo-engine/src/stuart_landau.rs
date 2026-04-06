// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Stuart-Landau model

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};
use crate::dp_tableau as dp;
use rayon::prelude::*;

pub struct StuartLandauStepper {
    n: usize, dt: f64, n_substeps: u32, method: Method,
    atol: f64, rtol: f64, last_dt: f64,
    deriv_buf: Vec<f64>, k1: Vec<f64>, k2: Vec<f64>, k3: Vec<f64>,
    k4: Vec<f64>, k5: Vec<f64>, k6: Vec<f64>, k7: Vec<f64>,
    y5: Vec<f64>, tmp_state: Vec<f64>,
    sin_theta: Vec<f64>, cos_theta: Vec<f64>,
}

impl std::fmt::Debug for StuartLandauStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StuartLandauStepper").field("n", &self.n).finish_non_exhaustive()
    }
}

impl StuartLandauStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 { return Err(SpoError::InvalidDimension("n must be > 0".into())); }
        config.validate()?;
        let dim = 2 * n;
        Ok(Self {
            n, dt: config.dt, n_substeps: config.n_substeps, method: config.method,
            atol: config.atol, rtol: config.rtol, last_dt: config.dt,
            deriv_buf: vec![0.0; dim], k1: vec![0.0; dim], k2: vec![0.0; dim], k3: vec![0.0; dim],
            k4: vec![0.0; dim], k5: vec![0.0; dim], k6: vec![0.0; dim], k7: vec![0.0; dim],
            y5: vec![0.0; dim], tmp_state: vec![0.0; dim],
            sin_theta: vec![0.0; n], cos_theta: vec![0.0; n],
        })
    }

    pub fn step(&mut self, state: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], epsilon: f64) -> SpoResult<()> {
        #[allow(unused_variables)] let n = self.n;
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);
        match self.method {
            Method::RK45 => { self.rk45_step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, alpha_zero); }
            _ => {
                let sub_dt = self.dt / f64::from(self.n_substeps);
                for _ in 0..self.n_substeps {
                    match self.method {
                        Method::Euler => self.euler_step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, sub_dt, alpha_zero),
                        Method::RK4 => self.rk4_step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, sub_dt, alpha_zero),
                        Method::RK45 => unreachable!(),
                    }
                }
            }
        }
        post_step(self.n, state);
        Ok(())
    }

    pub fn run(&mut self, state: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], epsilon: f64, n_steps: u64) -> SpoResult<()> {
        for _ in 0..n_steps { self.step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon)?; }
        Ok(())
    }

    pub fn n(&self) -> usize { self.n }
    pub fn last_dt(&self) -> f64 { self.last_dt }
    pub fn order_parameter(&self) -> (f64, f64) { crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta) }

    fn euler_step(&mut self, state: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], epsilon: f64, dt: f64, alpha_zero: bool) {
        compute_derivative(self.n, state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.deriv_buf);
        let dim = 2 * self.n;
        for i in 0..dim { state[i] += dt * self.deriv_buf[i]; }
    }

    fn rk4_step(&mut self, state: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], epsilon: f64, dt: f64, alpha_zero: bool) {
        let dim = 2 * self.n;
        compute_derivative(self.n, state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k1);
        for i in 0..dim { self.tmp_state[i] = state[i] + 0.5 * dt * self.k1[i]; }
        compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k2);
        for i in 0..dim { self.tmp_state[i] = state[i] + 0.5 * dt * self.k2[i]; }
        compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k3);
        for i in 0..dim { self.tmp_state[i] = state[i] + dt * self.k3[i]; }
        compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k4);
        let dt6 = dt / 6.0;
        for i in 0..dim { state[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]); }
    }

    fn rk45_step(&mut self, state: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], epsilon: f64, alpha_zero: bool) {
        let dim = 2 * self.n;
        let mut dt = self.last_dt;
        for _ in 0..=3 {
            compute_derivative(self.n, state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k1);
            for i in 0..dim { self.tmp_state[i] = state[i] + dt * dp::A21 * self.k1[i]; }
            compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k2);
            for i in 0..dim { self.tmp_state[i] = state[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]); }
            compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k3);
            for i in 0..dim { self.tmp_state[i] = state[i] + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]); }
            compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k4);
            for i in 0..dim { self.tmp_state[i] = state[i] + dt * (dp::A51 * self.k1[i] + dp::A52 * self.k2[i] + dp::A53 * self.k3[i] + dp::A54 * self.k4[i]); }
            compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k5);
            for i in 0..dim { self.tmp_state[i] = state[i] + dt * (dp::A61 * self.k1[i] + dp::A62 * self.k2[i] + dp::A63 * self.k3[i] + dp::A64 * self.k4[i] + dp::A65 * self.k5[i]); }
            compute_derivative(self.n, &self.tmp_state, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k6);
            for i in 0..dim { self.y5[i] = state[i] + dt * (dp::B5[0] * self.k1[i] + dp::B5[2] * self.k3[i] + dp::B5[3] * self.k4[i] + dp::B5[4] * self.k5[i] + dp::B5[5] * self.k6[i]); }
            compute_derivative(self.n, &self.y5, &mut self.sin_theta, &mut self.cos_theta, omegas, mu, knm, knm_r, zeta, psi, alpha, alpha_zero, epsilon, &mut self.k7);
            let mut err_norm: f64 = 0.0;
            for i in 0..dim {
                let y4 = state[i] + dt * (dp::B4[0] * self.k1[i] + dp::B4[2] * self.k3[i] + dp::B4[3] * self.k4[i] + dp::B4[4] * self.k5[i] + dp::B4[5] * self.k6[i] + dp::B4[6] * self.k7[i]);
                let err_i = (self.y5[i] - y4).abs();
                let scale = self.atol + self.rtol * state[i].abs().max(self.y5[i].abs());
                let ratio = err_i / scale;
                if ratio > err_norm { err_norm = ratio; }
            }
            if err_norm <= 1.0 {
                let factor = if err_norm > 0.0 { (0.9 * err_norm.powf(-0.2)).min(5.0) } else { 5.0 };
                self.last_dt = (dt * factor).min(self.dt * 10.0);
                state.copy_from_slice(&self.y5[..dim]);
                return;
            }
            let factor = (0.9 * err_norm.powf(-0.25)).max(0.2);
            dt *= factor;
        }
        self.last_dt = dt;
        state.copy_from_slice(&self.y5[..dim]);
    }
}

fn compute_derivative(n: usize, state: &[f64], sin_theta: &mut [f64], cos_theta: &mut [f64], omegas: &[f64], mu: &[f64], knm: &[f64], knm_r: &[f64], zeta: f64, psi: f64, alpha: &[f64], alpha_zero: bool, epsilon: f64, out: &mut [f64]) {
    let theta = &state[..n];
    let r = &state[n..];
    for i in 0..n { let (s, c) = theta[i].sin_cos(); sin_theta[i] = s; cos_theta[i] = c; }
    let (zs_psi, zc_psi) = if zeta != 0.0 { let (s, c) = psi.sin_cos(); (zeta * s, zeta * c) } else { (0.0, 0.0) };
    let st = &*sin_theta;
    let ct = &*cos_theta;
    let (phase_out, amp_out) = out.split_at_mut(n);

    if n >= 256 {
        phase_out.par_iter_mut().zip(amp_out.par_iter_mut()).enumerate().for_each(|(i, (p_val, a_val))| {
            let offset = i * n;
            let k_row = &knm[offset..offset + n];
            let kr_row = &knm_r[offset..offset + n];
            let mut p_fs = 0.0; let mut a_fs = 0.0;
            if alpha_zero {
                let mut k_iter = k_row.chunks_exact(8);
                let mut kr_iter = kr_row.chunks_exact(8);
                let mut s_iter = st.chunks_exact(8);
                let mut c_iter = ct.chunks_exact(8);
                let mut r_iter = r.chunks_exact(8);
                let mut p_acc0 = 0.0; let mut p_acc1 = 0.0; let mut p_acc2 = 0.0; let mut p_acc3 = 0.0;
                let mut p_acc4 = 0.0; let mut p_acc5 = 0.0; let mut p_acc6 = 0.0; let mut p_acc7 = 0.0;
                let mut a_acc0 = 0.0; let mut a_acc1 = 0.0; let mut a_acc2 = 0.0; let mut a_acc3 = 0.0;
                let mut a_acc4 = 0.0; let mut a_acc5 = 0.0; let mut a_acc6 = 0.0; let mut a_acc7 = 0.0;
                for (((kc, krc), sc), cc) in k_iter.by_ref().zip(kr_iter.by_ref()).zip(s_iter.by_ref()).zip(c_iter.by_ref()) {
                    let rc = r_iter.next().unwrap();
                    p_acc0 += kc[0] * (sc[0] * ct[i] - cc[0] * st[i]);
                    p_acc1 += kc[1] * (sc[1] * ct[i] - cc[1] * st[i]);
                    p_acc2 += kc[2] * (sc[2] * ct[i] - cc[2] * st[i]);
                    p_acc3 += kc[3] * (sc[3] * ct[i] - cc[3] * st[i]);
                    p_acc4 += kc[4] * (sc[4] * ct[i] - cc[4] * st[i]);
                    p_acc5 += kc[5] * (sc[5] * ct[i] - cc[5] * st[i]);
                    p_acc6 += kc[6] * (sc[6] * ct[i] - cc[6] * st[i]);
                    p_acc7 += kc[7] * (sc[7] * ct[i] - cc[7] * st[i]);
                    a_acc0 += krc[0] * rc[0].max(0.0) * (cc[0] * ct[i] + sc[0] * st[i]);
                    a_acc1 += krc[1] * rc[1].max(0.0) * (cc[1] * ct[i] + sc[1] * st[i]);
                    a_acc2 += krc[2] * rc[2].max(0.0) * (cc[2] * ct[i] + sc[2] * st[i]);
                    a_acc3 += krc[3] * rc[3].max(0.0) * (cc[3] * ct[i] + sc[3] * st[i]);
                    a_acc4 += krc[4] * rc[4].max(0.0) * (cc[4] * ct[i] + sc[4] * st[i]);
                    a_acc5 += krc[5] * rc[5].max(0.0) * (cc[5] * ct[i] + sc[5] * st[i]);
                    a_acc6 += krc[6] * rc[6].max(0.0) * (cc[6] * ct[i] + sc[6] * st[i]);
                    a_acc7 += krc[7] * rc[7].max(0.0) * (cc[7] * ct[i] + sc[7] * st[i]);
                }
                p_fs = p_acc0 + p_acc1 + p_acc2 + p_acc3 + p_acc4 + p_acc5 + p_acc6 + p_acc7;
                a_fs = a_acc0 + a_acc1 + a_acc2 + a_acc3 + a_acc4 + a_acc5 + a_acc6 + a_acc7;
                for ((((&kj, &krj), &sj), &cj), &rj) in k_iter.remainder().iter().zip(kr_iter.remainder()).zip(s_iter.remainder()).zip(c_iter.remainder()).zip(r_iter.remainder()) {
                    p_fs += kj * (sj * ct[i] - cj * st[i]);
                    a_fs += krj * rj.max(0.0) * (cj * ct[i] + sj * st[i]);
                }
            } else {
                for j in 0..n {
                    let diff = theta[j] - theta[i];
                    p_fs += knm[offset + j] * (diff - alpha[offset + j]).sin();
                    a_fs += knm_r[offset + j] * r[j].max(0.0) * (diff - alpha[offset + j]).cos();
                }
            }
            *p_val = omegas[i] + p_fs;
            if zeta != 0.0 { *p_val += zs_psi * ct[i] - zc_psi * st[i]; }
            let ri = r[i];
            *a_val = (mu[i] - ri * ri) * ri + epsilon * a_fs;
        });
    } else {
        for i in 0..n {
            let mut p_fs = 0.0; let mut a_fs = 0.0;
            let offset = i * n;
            if alpha_zero {
                for j in 0..n {
                    p_fs += knm[offset + j] * (sin_theta[j] * cos_theta[i] - cos_theta[j] * sin_theta[i]);
                    a_fs += knm_r[offset + j] * r[j].max(0.0) * (cos_theta[j] * cos_theta[i] + sin_theta[j] * sin_theta[i]);
                }
            } else {
                for j in 0..n {
                    let diff = theta[j] - theta[i];
                    p_fs += knm[offset + j] * (diff - alpha[offset + j]).sin();
                    a_fs += knm_r[offset + j] * r[j].max(0.0) * (diff - alpha[offset + j]).cos();
                }
            }
            phase_out[i] = omegas[i] + p_fs;
            if zeta != 0.0 { phase_out[i] += zs_psi * cos_theta[i] - zc_psi * sin_theta[i]; }
            let ri = r[i];
            amp_out[i] = (mu[i] - ri * ri) * ri + epsilon * a_fs;
        }
    }
}

fn post_step(n: usize, state: &mut [f64]) {
    for p in state[..n].iter_mut() { *p = p.rem_euclid(std::f64::consts::TAU); }
    for a in state[n..].iter_mut() { if *a < 0.0 { *a = 0.0; } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::IntegrationConfig;
    fn make_stepper(n: usize) -> StuartLandauStepper { StuartLandauStepper::new(n, IntegrationConfig::default()).expect("valid config") }
    #[test]
    fn single_euler_step() {
        let n = 4; let mut s = make_stepper(n); let mut state = vec![0.0; 2 * n];
        for i in 0..n { state[n + i] = 1.0; }
        let omegas = vec![1.0; n]; let mu = vec![1.0; n];
        let knm = vec![0.0; n * n]; let knm_r = vec![0.0; n * n]; let alpha = vec![0.0; n * n];
        s.step(&mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0).expect("step");
        for i in 0..n { assert!((state[i] - 0.01).abs() < 1e-12); assert!((state[n + i] - 1.0).abs() < 1e-12); }
    }
}
