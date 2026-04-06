// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — UPDE solver

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};

use crate::dp_tableau as dp;
use crate::plasticity::PlasticityModel;
use rayon::prelude::*;

/// Kuramoto UPDE integrator with pre-allocated scratch arrays.
pub struct UPDEStepper {
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
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
    pub plasticity: Option<PlasticityModel>,
    pub modulator: f64,
}

impl std::fmt::Debug for UPDEStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UPDEStepper")
            .field("n", &self.n)
            .field("dt", &self.dt)
            .field("n_substeps", &self.n_substeps)
            .field("method", &self.method)
            .field("last_dt", &self.last_dt)
            .finish_non_exhaustive()
    }
}

impl UPDEStepper {
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;
        Ok(Self {
            n,
            dt: config.dt,
            n_substeps: config.n_substeps,
            method: config.method,
            atol: config.atol,
            rtol: config.rtol,
            last_dt: config.dt,
            deriv_buf: vec![0.0; n],
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            k5: vec![0.0; n],
            k6: vec![0.0; n],
            k7: vec![0.0; n],
            y5: vec![0.0; n],
            tmp_phases: vec![0.0; n],
            sin_theta: vec![0.0; n],
            cos_theta: vec![0.0; n],
            plasticity: None,
            modulator: 1.0,
        })
    }

    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
    ) -> SpoResult<()> {
        #[allow(unused_variables)]
        let n = self.n;
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);

        match self.method {
            Method::RK45 => {
                self.rk45_step(phases, omegas, knm, zeta, psi, alpha, alpha_zero);
            }
            _ => {
                let sub_dt = self.dt / f64::from(self.n_substeps);
                for _ in 0..self.n_substeps {
                    match self.method {
                        Method::Euler => {
                            self.euler_step(
                                phases, omegas, knm, zeta, psi, alpha, sub_dt, alpha_zero,
                            );
                        }
                        Method::RK4 => {
                            self.rk4_step(
                                phases, omegas, knm, zeta, psi, alpha, sub_dt, alpha_zero,
                            );
                        }
                        Method::RK45 => unreachable!(),
                    }
                }
            }
        }

        if let Some(ref plast) = self.plasticity {
            plast.update(
                &self.sin_theta,
                &self.cos_theta,
                knm,
                self.modulator,
                self.dt,
            );
        }
        wrap_phases(phases);
        Ok(())
    }

    pub fn run(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        n_steps: u64,
    ) -> SpoResult<()> {
        for _ in 0..n_steps {
            self.step(phases, omegas, knm, zeta, psi, alpha)?;
        }
        Ok(())
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    pub fn order_parameter(&self) -> (f64, f64) {
        crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta)
    }

    fn euler_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        dt: f64,
        alpha_zero: bool,
    ) {
        compute_derivative(
            self.n,
            phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            alpha_zero,
            &mut self.deriv_buf,
        );
        for i in 0..self.n {
            phases[i] += dt * self.deriv_buf[i];
        }
    }

    fn rk4_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        dt: f64,
        alpha_zero: bool,
    ) {
        #[allow(unused_variables)]
        let n = self.n;
        compute_derivative(
            n,
            phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            alpha_zero,
            &mut self.k1,
        );
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            alpha_zero,
            &mut self.k2,
        );
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k2[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            alpha_zero,
            &mut self.k3,
        );
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + dt * self.k3[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            &mut self.sin_theta,
            &mut self.cos_theta,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            alpha_zero,
            &mut self.k4,
        );
        let dt6 = dt / 6.0;
        for i in 0..n {
            phases[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }

    fn rk45_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        alpha_zero: bool,
    ) {
        #[allow(unused_variables)]
        let n = self.n;
        let mut dt = self.last_dt;
        for _ in 0..=3 {
            compute_derivative(
                n,
                phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k1,
            );
            for i in 0..n {
                self.tmp_phases[i] = phases[i] + dt * dp::A21 * self.k1[i];
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k2,
            );
            for i in 0..n {
                self.tmp_phases[i] = phases[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k3,
            );
            for i in 0..n {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k4,
            );
            for i in 0..n {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A51 * self.k1[i]
                        + dp::A52 * self.k2[i]
                        + dp::A53 * self.k3[i]
                        + dp::A54 * self.k4[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k5,
            );
            for i in 0..n {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A61 * self.k1[i]
                        + dp::A62 * self.k2[i]
                        + dp::A63 * self.k3[i]
                        + dp::A64 * self.k4[i]
                        + dp::A65 * self.k5[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k6,
            );
            for i in 0..n {
                self.y5[i] = phases[i]
                    + dt * (dp::B5[0] * self.k1[i]
                        + dp::B5[2] * self.k3[i]
                        + dp::B5[3] * self.k4[i]
                        + dp::B5[4] * self.k5[i]
                        + dp::B5[5] * self.k6[i]);
            }
            compute_derivative(
                n,
                &self.y5,
                &mut self.sin_theta,
                &mut self.cos_theta,
                omegas,
                knm,
                zeta,
                psi,
                alpha,
                alpha_zero,
                &mut self.k7,
            );
            let mut err_norm: f64 = 0.0;
            for i in 0..n {
                let y4 = phases[i]
                    + dt * (dp::B4[0] * self.k1[i]
                        + dp::B4[2] * self.k3[i]
                        + dp::B4[3] * self.k4[i]
                        + dp::B4[4] * self.k5[i]
                        + dp::B4[5] * self.k6[i]
                        + dp::B4[6] * self.k7[i]);
                let err_i = (self.y5[i] - y4).abs();
                let scale = self.atol + self.rtol * phases[i].abs().max(self.y5[i].abs());
                let ratio = err_i / scale;
                if ratio > err_norm {
                    err_norm = ratio;
                }
            }
            if err_norm <= 1.0 {
                let factor = if err_norm > 0.0 {
                    (0.9 * err_norm.powf(-0.2)).min(5.0)
                } else {
                    5.0
                };
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

fn compute_derivative(
    n: usize,
    theta: &[f64],
    sin_theta: &mut [f64],
    cos_theta: &mut [f64],
    omegas: &[f64],
    knm: &mut [f64],
    zeta: f64,
    psi: f64,
    alpha: &[f64],
    alpha_zero: bool,
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
    let st = &*sin_theta;
    let ct = &*cos_theta;

    if n >= 256 {
        let block_size = 64;
        out.par_chunks_mut(block_size)
            .enumerate()
            .for_each(|(block_idx, chunk)| {
                let start_i = block_idx * block_size;
                if alpha_zero {
                    for (local_i, val) in chunk.iter_mut().enumerate() {
                        let i = start_i + local_i;
                        let offset = i * n;
                        let k_row = &knm[offset..offset + n];
                        let mut s_acc0 = 0.0;
                        let mut s_acc1 = 0.0;
                        let mut s_acc2 = 0.0;
                        let mut s_acc3 = 0.0;
                        let mut s_acc4 = 0.0;
                        let mut s_acc5 = 0.0;
                        let mut s_acc6 = 0.0;
                        let mut s_acc7 = 0.0;
                        let mut c_acc0 = 0.0;
                        let mut c_acc1 = 0.0;
                        let mut c_acc2 = 0.0;
                        let mut c_acc3 = 0.0;
                        let mut c_acc4 = 0.0;
                        let mut c_acc5 = 0.0;
                        let mut c_acc6 = 0.0;
                        let mut c_acc7 = 0.0;
                        let mut k_iter = k_row.chunks_exact(8);
                        let mut s_iter = st.chunks_exact(8);
                        let mut c_iter = ct.chunks_exact(8);
                        for ((kc, sc), cc) in
                            k_iter.by_ref().zip(s_iter.by_ref()).zip(c_iter.by_ref())
                        {
                            s_acc0 += kc[0] * sc[0];
                            s_acc1 += kc[1] * sc[1];
                            s_acc2 += kc[2] * sc[2];
                            s_acc3 += kc[3] * sc[3];
                            s_acc4 += kc[4] * sc[4];
                            s_acc5 += kc[5] * sc[5];
                            s_acc6 += kc[6] * sc[6];
                            s_acc7 += kc[7] * sc[7];
                            c_acc0 += kc[0] * cc[0];
                            c_acc1 += kc[1] * cc[1];
                            c_acc2 += kc[2] * cc[2];
                            c_acc3 += kc[3] * cc[3];
                            c_acc4 += kc[4] * cc[4];
                            c_acc5 += kc[5] * cc[5];
                            c_acc6 += kc[6] * cc[6];
                            c_acc7 += kc[7] * cc[7];
                        }
                        let mut fs =
                            s_acc0 + s_acc1 + s_acc2 + s_acc3 + s_acc4 + s_acc5 + s_acc6 + s_acc7;
                        let mut fc =
                            c_acc0 + c_acc1 + c_acc2 + c_acc3 + c_acc4 + c_acc5 + c_acc6 + c_acc7;
                        for ((&kj, &sj), &cj) in k_iter
                            .remainder()
                            .iter()
                            .zip(s_iter.remainder())
                            .zip(c_iter.remainder())
                        {
                            fs += kj * sj;
                            fc += kj * cj;
                        }
                        let coupling_sum = fs * ct[i] - fc * st[i];
                        *val = omegas[i] + coupling_sum;
                        if zeta != 0.0 {
                            *val += zs_psi * ct[i] - zc_psi * st[i];
                        }
                    }
                } else {
                    for (local_i, val) in chunk.iter_mut().enumerate() {
                        let i = start_i + local_i;
                        let mut coupling_sum = 0.0;
                        let offset = i * n;
                        for j in 0..n {
                            coupling_sum +=
                                knm[offset + j] * (theta[j] - theta[i] - alpha[offset + j]).sin();
                        }
                        *val = omegas[i] + coupling_sum;
                        if zeta != 0.0 {
                            *val += zs_psi * ct[i] - zc_psi * st[i];
                        }
                    }
                }
            });
    } else {
        if alpha_zero {
            for i in 0..n {
                let mut fs = 0.0;
                let mut fc = 0.0;
                let offset = i * n;
                for j in 0..n {
                    fs += knm[offset + j] * st[j];
                    fc += knm[offset + j] * ct[j];
                }
                let coupling_sum = fs * ct[i] - fc * st[i];
                out[i] = omegas[i] + coupling_sum;
                if zeta != 0.0 {
                    out[i] += zs_psi * ct[i] - zc_psi * st[i];
                }
            }
        } else {
            for i in 0..n {
                let mut coupling_sum = 0.0;
                let offset = i * n;
                for j in 0..n {
                    coupling_sum +=
                        knm[offset + j] * (theta[j] - theta[i] - alpha[offset + j]).sin();
                }
                out[i] = omegas[i] + coupling_sum;
                if zeta != 0.0 {
                    out[i] += zs_psi * ct[i] - zc_psi * st[i];
                }
            }
        }
    }
}

fn wrap_phases(phases: &mut [f64]) {
    for p in phases.iter_mut() {
        *p = p.rem_euclid(std::f64::consts::TAU);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::IntegrationConfig;
    fn make_stepper(n: usize) -> UPDEStepper {
        UPDEStepper::new(n, IntegrationConfig::default()).unwrap()
    }
    fn zero_alpha(n: usize) -> Vec<f64> {
        vec![0.0; n * n]
    }
    #[test]
    fn zero_n_rejected() {
        assert!(UPDEStepper::new(0, IntegrationConfig::default()).is_err());
    }
    #[test]
    fn single_euler_step() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut phases = vec![0.0; n];
        let omegas = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        s.step(&mut phases, &omegas, &mut knm, 0.0, 0.0, &alpha)
            .unwrap();
        for &p in &phases {
            assert!((p - 0.01).abs() < 1e-12);
        }
    }
    #[test]
    fn synchronisation_tendency() {
        let n = 8;
        let mut s = make_stepper(n);
        let mut phases: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * i as f64).collect();
        let omegas = vec![1.0; n];
        let mut knm = vec![5.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = zero_alpha(n);
        let r_before = crate::order_params::compute_order_parameter(&phases).0;
        s.run(&mut phases, &omegas, &mut knm, 0.0, 0.0, &alpha, 1000)
            .unwrap();
        let r_after = crate::order_params::compute_order_parameter(&phases).0;
        assert!(r_after > r_before);
    }
}
