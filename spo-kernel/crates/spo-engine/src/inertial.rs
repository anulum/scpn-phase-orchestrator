// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// SCPN Phase Orchestrator — Inertial coupling

use std::f64::consts::TAU;
use spo_types::{IntegrationConfig, SpoError, SpoResult};
use rayon::prelude::*;

pub struct InertialStepper {
    n: usize, dt: f64,
    k1t: Vec<f64>, k1o: Vec<f64>,
    k2t: Vec<f64>, k2o: Vec<f64>,
    k3t: Vec<f64>, k3o: Vec<f64>,
    k4t: Vec<f64>, k4o: Vec<f64>,
    tmp_th: Vec<f64>, tmp_od: Vec<f64>,
    sin_theta: Vec<f64>, cos_theta: Vec<f64>,
}

impl std::fmt::Debug for InertialStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InertialStepper").field("n", &self.n).finish_non_exhaustive()
    }
}

impl InertialStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 { return Err(SpoError::InvalidDimension("n must be > 0".into())); }
        config.validate()?;
        let dim = n;
        Ok(Self {
            n, dt: config.dt,
            k1t: vec![0.0; dim], k1o: vec![0.0; dim],
            k2t: vec![0.0; dim], k2o: vec![0.0; dim],
            k3t: vec![0.0; dim], k3o: vec![0.0; dim],
            k4t: vec![0.0; dim], k4o: vec![0.0; dim],
            tmp_th: vec![0.0; dim], tmp_od: vec![0.0; dim],
            sin_theta: vec![0.0; dim], cos_theta: vec![0.0; dim],
        })
    }

    pub fn step(&mut self, theta: &mut [f64], omega_dot: &mut [f64], power: &[f64], knm: &[f64], inertia: &[f64], damping: &[f64]) -> SpoResult<()> {
        let n = self.n;
        let dt = self.dt;

        compute_derivative(n, theta, omega_dot, power, knm, inertia, damping, &mut self.sin_theta, &mut self.cos_theta, &mut self.k1t, &mut self.k1o);

        for i in 0..n {
            self.tmp_th[i] = theta[i] + 0.5 * dt * self.k1t[i];
            self.tmp_od[i] = omega_dot[i] + 0.5 * dt * self.k1o[i];
        }
        compute_derivative(n, &self.tmp_th, &self.tmp_od, power, knm, inertia, damping, &mut self.sin_theta, &mut self.cos_theta, &mut self.k2t, &mut self.k2o);

        for i in 0..n {
            self.tmp_th[i] = theta[i] + 0.5 * dt * self.k2t[i];
            self.tmp_od[i] = omega_dot[i] + 0.5 * dt * self.k2o[i];
        }
        compute_derivative(n, &self.tmp_th, &self.tmp_od, power, knm, inertia, damping, &mut self.sin_theta, &mut self.cos_theta, &mut self.k3t, &mut self.k3o);

        for i in 0..n {
            self.tmp_th[i] = theta[i] + dt * self.k3t[i];
            self.tmp_od[i] = omega_dot[i] + dt * self.k3o[i];
        }
        compute_derivative(n, &self.tmp_th, &self.tmp_od, power, knm, inertia, damping, &mut self.sin_theta, &mut self.cos_theta, &mut self.k4t, &mut self.k4o);

        let dt6 = dt / 6.0;
        for i in 0..n {
            let raw = theta[i] + dt6 * (self.k1t[i] + 2.0 * self.k2t[i] + 2.0 * self.k3t[i] + self.k4t[i]);
            theta[i] = raw.rem_euclid(TAU);
            omega_dot[i] += dt6 * (self.k1o[i] + 2.0 * self.k2o[i] + 2.0 * self.k3o[i] + self.k4o[i]);
        }
        Ok(())
    }
}

fn compute_derivative(n: usize, theta: &[f64], omega_dot: &[f64], power: &[f64], knm: &[f64], inertia: &[f64], damping: &[f64], sin_theta: &mut [f64], cos_theta: &mut [f64], out_t: &mut [f64], out_o: &mut [f64]) {
    for i in 0..n {
        let (s, c) = theta[i].sin_cos();
        sin_theta[i] = s;
        cos_theta[i] = c;
    }
    let st = &*sin_theta;
    let ct = &*cos_theta;

    out_t.copy_from_slice(omega_dot);

    out_o.par_iter_mut().enumerate().for_each(|(i, val)| {
        let offset = i * n;
        let k_row = &knm[offset..offset + n];
        let ci = ct[i]; let si = st[i];
        let mut acc = 0.0;
        let mut k_iter = k_row.chunks_exact(8);
        let mut s_iter = st.chunks_exact(8);
        let mut c_iter = ct.chunks_exact(8);
        for ((kc, sc), cc) in k_iter.by_ref().zip(s_iter.by_ref()).zip(c_iter.by_ref()) {
            acc += kc[0]*(sc[0]*ci - cc[0]*si) + kc[1]*(sc[1]*ci - cc[1]*si) + kc[2]*(sc[2]*ci - cc[2]*si) + kc[3]*(sc[3]*ci - cc[3]*si) +
                   kc[4]*(sc[4]*ci - cc[4]*si) + kc[5]*(sc[5]*ci - cc[5]*si) + kc[6]*(sc[6]*ci - cc[6]*si) + kc[7]*(sc[7]*ci - cc[7]*si);
        }
        let mut coupling = acc;
        for ((&kj, &sj), &cj) in k_iter.remainder().iter().zip(s_iter.remainder()).zip(c_iter.remainder()) {
            coupling += kj * (sj * ci - cj * si);
        }
        *val = (power[i] + coupling - damping[i] * omega_dot[i]) / inertia[i];
    });
}

pub fn inertial_step(theta: &[f64], omega_dot: &[f64], power: &[f64], knm: &[f64], inertia: &[f64], damping: &[f64], n: usize, dt: f64) -> (Vec<f64>, Vec<f64>) {
    let mut s = InertialStepper::new(n, IntegrationConfig { dt, ..Default::default() }).unwrap();
    let mut th = theta.to_vec();
    let mut od = omega_dot.to_vec();
    s.step(&mut th, &mut od, power, knm, inertia, damping).unwrap();
    (th, od)
}

pub fn inertial_run(theta_init: &[f64], omega_init: &[f64], power: &[f64], knm: &[f64], inertia: &[f64], damping: &[f64], n: usize, dt: f64, n_steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = InertialStepper::new(n, IntegrationConfig { dt, ..Default::default() }).unwrap();
    let mut th = theta_init.to_vec();
    let mut od = omega_init.to_vec();
    let mut th_traj = Vec::with_capacity(n_steps * n);
    let mut od_traj = Vec::with_capacity(n_steps * n);
    for _ in 0..n_steps {
        s.step(&mut th, &mut od, power, knm, inertia, damping).unwrap();
        th_traj.extend_from_slice(&th);
        od_traj.extend_from_slice(&od);
    }
    (th, od, th_traj, od_traj)
}
