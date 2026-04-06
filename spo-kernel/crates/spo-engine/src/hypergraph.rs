// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// SCPN Phase Orchestrator — Hypergraph coupling

use std::f64::consts::TAU;
use spo_types::{IntegrationConfig, SpoError, SpoResult};
use rayon::prelude::*;

pub struct Hyperedge {
    pub nodes: Vec<usize>,
    pub strength: f64,
}

pub struct HypergraphStepper {
    n: usize, dt: f64,
    deriv_buf: Vec<f64>, sin_theta: Vec<f64>, cos_theta: Vec<f64>,
}

impl std::fmt::Debug for HypergraphStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HypergraphStepper").field("n", &self.n).finish_non_exhaustive()
    }
}

impl HypergraphStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 { return Err(SpoError::InvalidDimension("n must be > 0".into())); }
        config.validate()?;
        Ok(Self {
            n, dt: config.dt,
            deriv_buf: vec![0.0; n], sin_theta: vec![0.0; n], cos_theta: vec![0.0; n],
        })
    }

    pub fn step(&mut self, phases: &mut [f64], omegas: &[f64], edges: &[Hyperedge], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64) -> SpoResult<()> {
        self.compute_derivative(phases, omegas, edges, knm, alpha, zeta, psi);
        for i in 0..self.n {
            phases[i] = (phases[i] + self.dt * self.deriv_buf[i]).rem_euclid(TAU);
        }
        Ok(())
    }

    pub fn run(&mut self, phases: &mut [f64], omegas: &[f64], edges: &[Hyperedge], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64, n_steps: usize) -> SpoResult<()> {
        for _ in 0..n_steps {
            self.compute_derivative(phases, omegas, edges, knm, alpha, zeta, psi);
            for i in 0..self.n {
                phases[i] = (phases[i] + self.dt * self.deriv_buf[i]).rem_euclid(TAU);
            }
        }
        Ok(())
    }

    pub fn order_parameter(&self) -> (f64, f64) {
        crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta)
    }

    fn compute_derivative(&mut self, theta: &[f64], omegas: &[f64], edges: &[Hyperedge], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64) {
        let n = self.n;
        for i in 0..n {
            let (s, c) = theta[i].sin_cos();
            self.sin_theta[i] = s;
            self.cos_theta[i] = c;
        }
        let (zs_psi, zc_psi) = if zeta != 0.0 {
            let (s, c) = psi.sin_cos();
            (zeta * s, zeta * c)
        } else {
            (0.0, 0.0)
        };
        let st = &*self.sin_theta;
        let ct = &*self.cos_theta;
        let has_pairwise = knm.len() == n * n;
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);

        self.deriv_buf.par_iter_mut().enumerate().for_each(|(i, val)| {
            let mut pw = 0.0;
            if has_pairwise {
                let offset = i * n;
                let k_row = &knm[offset..offset + n];
                let ci = ct[i]; let si = st[i];
                if alpha_zero {
                    let mut k_iter = k_row.chunks_exact(8);
                    let mut s_iter = st.chunks_exact(8);
                    let mut c_iter = ct.chunks_exact(8);
                    let mut acc = 0.0;
                    for ((kc, sc), cc) in k_iter.by_ref().zip(s_iter.by_ref()).zip(c_iter.by_ref()) {
                        acc += kc[0]*(sc[0]*ci - cc[0]*si) + kc[1]*(sc[1]*ci - cc[1]*si) + kc[2]*(sc[2]*ci - cc[2]*si) + kc[3]*(sc[3]*ci - cc[3]*si) +
                               kc[4]*(sc[4]*ci - cc[4]*si) + kc[5]*(sc[5]*ci - cc[5]*si) + kc[6]*(sc[6]*ci - cc[6]*si) + kc[7]*(sc[7]*ci - cc[7]*si);
                    }
                    pw = acc;
                    for ((&kj, &sj), &cj) in k_iter.remainder().iter().zip(s_iter.remainder()).zip(c_iter.remainder()) {
                        pw += kj * (sj * ci - cj * si);
                    }
                } else {
                    for j in 0..n { pw += k_row[j] * (theta[j] - theta[i] - alpha[offset + j]).sin(); }
                }
            }
            *val = omegas[i] + pw;
            if zeta != 0.0 { *val += zs_psi * ct[i] - zc_psi * st[i]; }
        });

        // Hyperedges are processed sequentially but updated onto the parallel deriv_buf
        for edge in edges {
            let k = edge.nodes.len();
            let phase_sum: f64 = edge.nodes.iter().map(|&idx| theta[idx]).sum();
            for &m in &edge.nodes {
                self.deriv_buf[m] += edge.strength * (phase_sum - (k as f64) * theta[m]).sin();
            }
        }
    }
}

pub fn hypergraph_run(phases: &[f64], omegas: &[f64], n: usize, edges: &[Hyperedge], knm: &[f64], alpha: &[f64], zeta: f64, psi: f64, dt: f64, n_steps: usize) -> Vec<f64> {
    let mut s = HypergraphStepper::new(n, IntegrationConfig { dt, ..Default::default() }).unwrap();
    let mut p = phases.to_vec();
    s.run(&mut p, omegas, edges, knm, alpha, zeta, psi, n_steps).unwrap();
    p
}

pub fn order_parameter(phases: &[f64]) -> f64 {
    crate::order_params::compute_order_parameter(phases).0
}
