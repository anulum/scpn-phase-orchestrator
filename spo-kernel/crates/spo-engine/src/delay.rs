// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

use std::f64::consts::TAU;
use rayon::prelude::*;
use spo_types::{IntegrationConfig, SpoResult, SpoError};

pub struct DelayedStepper {
    n: usize, dt: f64, delay_steps: usize, head: usize,
    history_sincos: Vec<f64>, sin_theta: Vec<f64>, cos_theta: Vec<f64>, deriv_buf: Vec<f64>,
}

impl std::fmt::Debug for DelayedStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelayedStepper").field("n", &self.n).field("delay_steps", &self.delay_steps).finish_non_exhaustive()
    }
}

impl DelayedStepper {
    pub fn new(n: usize, delay_steps: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 { return Err(SpoError::InvalidDimension("n must be > 0".into())); }
        config.validate()?;
        let max_buf = delay_steps + 1;
        Ok(Self {
            n, dt: config.dt, delay_steps, head: 0,
            history_sincos: vec![0.0; max_buf * 2 * n], sin_theta: vec![0.0; n], cos_theta: vec![0.0; n], deriv_buf: vec![0.0; n],
        })
    }

    pub fn step(&mut self, phases: &mut [f64], omegas: &[f64], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64, step_idx: usize) -> SpoResult<()> {
        let n = self.n; let max_buf = self.delay_steps + 1;
        for i in 0..n {
            let (s, c) = phases[i].sin_cos();
            self.sin_theta[i] = s; self.cos_theta[i] = c;
            self.history_sincos[self.head * 2 * n + 2 * i] = s;
            self.history_sincos[self.head * 2 * n + 2 * i + 1] = c;
        }
        let delayed_idx = if self.delay_steps > 0 && step_idx >= self.delay_steps { (self.head + max_buf - self.delay_steps) % max_buf } else { self.head };
        let d_sc = &self.history_sincos[delayed_idx * 2 * n .. (delayed_idx + 1) * 2 * n];
        let (zs_psi, zc_psi) = if zeta != 0.0 { let (s, c) = psi.sin_cos(); (zeta * s, zeta * c) } else { (0.0, 0.0) };
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);
        let st = &*self.sin_theta; let ct = &*self.cos_theta;

        if n >= 256 {
            self.deriv_buf.par_iter_mut().enumerate().for_each(|(i, val)| {
                let offset = i * n; let k_row = &knm[offset..offset + n];
                let ci = ct[i]; let si = st[i];
                let mut coupling = 0.0;
                if alpha_zero {
                    let mut k_iter = k_row.chunks_exact(8);
                    let mut d_sc_iter = d_sc.chunks_exact(16);
                    let mut acc = 0.0;
                    for (kc, dsc8) in k_iter.by_ref().zip(d_sc_iter.by_ref()) {
                        acc += kc[0]*(dsc8[0]*ci - dsc8[1]*si) + kc[1]*(dsc8[2]*ci - dsc8[3]*si) + kc[2]*(dsc8[4]*ci - dsc8[5]*si) + kc[3]*(dsc8[6]*ci - dsc8[7]*si) +
                               kc[4]*(dsc8[8]*ci - dsc8[9]*si) + kc[5]*(dsc8[10]*ci - dsc8[11]*si) + kc[6]*(dsc8[12]*ci - dsc8[13]*si) + kc[7]*(dsc8[14]*ci - dsc8[15]*si);
                    }
                    coupling = acc;
                    for (&kj, dsc2) in k_iter.remainder().iter().zip(d_sc_iter.remainder().chunks_exact(2)) { coupling += kj * (dsc2[0]*ci - dsc2[1]*si); }
                } else {
                    for j in 0..n { let tj_d = d_sc[2*j].atan2(d_sc[2*j+1]); coupling += k_row[j] * (tj_d - phases[i] - alpha[offset+j]).sin(); }
                }
                *val = omegas[i] + coupling + (if zeta != 0.0 { zs_psi * ci - zc_psi * si } else { 0.0 });
            });
        } else {
            for i in 0..n {
                let offset = i * n; let k_row = &knm[offset..offset + n];
                let ci = ct[i]; let si = st[i];
                let mut coupling = 0.0;
                if alpha_zero { for j in 0..n { coupling += k_row[j] * (d_sc[2*j]*ci - d_sc[2*j+1]*si); } }
                else { for j in 0..n { let tj_d = d_sc[2*j].atan2(d_sc[2*j+1]); coupling += k_row[j] * (tj_d - phases[i] - alpha[offset+j]).sin(); } }
                self.deriv_buf[i] = omegas[i] + coupling + (if zeta != 0.0 { zs_psi * ci - zc_psi * si } else { 0.0 });
            }
        }
        for i in 0..n { phases[i] = (phases[i] + self.dt * self.deriv_buf[i]).rem_euclid(TAU); }
        self.head = (self.head + 1) % max_buf;
        Ok(())
    }

    pub fn run(&mut self, phases: &mut [f64], omegas: &[f64], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64, n_steps: usize) -> SpoResult<()> {
        for step_idx in 0..n_steps { self.step(phases, omegas, knm, alpha, zeta, psi, step_idx)?; }
        Ok(())
    }

    pub fn order_parameter(&self) -> (f64, f64) {
        crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta)
    }
}

pub fn delayed_kuramoto_run(phases_init: &[f64], omegas: &[f64], knm_flat: &[f64], alpha_flat: &[f64], n: usize, zeta: f64, psi: f64, dt: f64, delay_steps: usize, n_steps: usize) -> Vec<f64> {
    let mut s = DelayedStepper::new(n, delay_steps, IntegrationConfig { dt, ..Default::default() }).unwrap();
    let mut p = phases_init.to_vec();
    s.run(&mut p, omegas, knm_flat, alpha_flat, zeta, psi, n_steps).unwrap();
    p
}
