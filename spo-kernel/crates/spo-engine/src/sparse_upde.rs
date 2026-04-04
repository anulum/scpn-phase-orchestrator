// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — UPDE solver

//!
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
//!
//! Supports Euler and RK4 with pre-allocated scratch arrays for
//! zero-alloc hot-path execution at dynamic N (4-256 oscillators).

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};

use crate::dp_tableau as dp;
use crate::plasticity::PlasticityModel;

/// Kuramoto UPDE integrator with pre-allocated scratch arrays.
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
    pub plasticity: Option<PlasticityModel>,
    pub modulator: f64,
}

impl std::fmt::Debug for SparseUPDEStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseUPDEStepper")
            .field("n", &self.n)
            .field("dt", &self.dt)
            .field("n_substeps", &self.n_substeps)
            .field("method", &self.method)
            .field("last_dt", &self.last_dt)
            .finish_non_exhaustive()
    }
}

impl SparseUPDEStepper {
    /// # Errors
    /// Returns `InvalidDimension` if n is 0, or propagates config validation errors.
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
            plasticity: None,
            modulator: 1.0,
        })
    }

    /// Advance phases in-place by one timestep.
    ///
    /// `knm` is row-major N×N, `alpha` is row-major N×N phase lags.
    ///
    /// # Errors
    /// Returns `InvalidDimension` on length mismatch or `IntegrationDiverged` on NaN/Inf input.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha_values: &[f64],
    ) -> SpoResult<()> {
        let n = self.n;
        if phases.len() != n || omegas.len() != n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {n}, got phases={} omegas={}",
                phases.len(),
                omegas.len()
            )));
        }
        if row_ptr.len() != n + 1 {
            return Err(SpoError::InvalidDimension(format!(
                "expected row_ptr len {}, got {}",
                n + 1,
                row_ptr.len()
            )));
        }
        if knm_values.len() != col_indices.len() || alpha_values.len() != col_indices.len() {
            return Err(SpoError::InvalidDimension(
                "sparse arrays length mismatch".to_string(),
            ));
        }
        for &th in phases.iter() {
            if !th.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "input phases contain NaN/Inf".into(),
                ));
            }
        }
        for &w in omegas {
            if !w.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "omegas contain NaN/Inf".into(),
                ));
            }
        }
        for k in knm_values.iter() {
            if !k.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "knm_values contains NaN/Inf".into(),
                ));
            }
        }
        for &a in alpha_values {
            if !a.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "alpha_values contains NaN/Inf".into(),
                ));
            }
        }
        if !zeta.is_finite() || !psi.is_finite() {
            return Err(SpoError::IntegrationDiverged(
                "zeta/psi contain NaN/Inf".into(),
            ));
        }

        match self.method {
            Method::RK45 => {
                self.rk45_step(
                    phases,
                    omegas,
                    row_ptr,
                    col_indices,
                    knm_values,
                    zeta,
                    psi,
                    alpha_values,
                );
            }
            _ => {
                let sub_dt = self.dt / f64::from(self.n_substeps);
                for _ in 0..self.n_substeps {
                    match self.method {
                        Method::Euler => {
                            self.euler_step(
                                phases,
                                omegas,
                                row_ptr,
                                col_indices,
                                knm_values,
                                zeta,
                                psi,
                                alpha_values,
                                sub_dt,
                            );
                        }
                        Method::RK4 => {
                            self.rk4_step(
                                phases,
                                omegas,
                                row_ptr,
                                col_indices,
                                knm_values,
                                zeta,
                                psi,
                                alpha_values,
                                sub_dt,
                            );
                        }
                        Method::RK45 => unreachable!(),
                    }
                }
            }
        }

        if let Some(ref plast) = self.plasticity {
            plast.update_sparse(
                phases,
                row_ptr,
                col_indices,
                knm_values,
                self.modulator,
                self.dt,
            );
        }
        wrap_phases(phases);
        Ok(())
    }

    /// Run multiple steps, returning the final phases.
    ///
    /// # Errors
    /// Propagates errors from `step()`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha_values: &[f64],
        n_steps: u64,
    ) -> SpoResult<()> {
        for _ in 0..n_steps {
            self.step(
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Actual dt used on the last accepted step (relevant for RK45).
    #[must_use]
    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    fn euler_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha_values: &[f64],
        dt: f64,
    ) {
        compute_derivative(
            self.n,
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            zeta,
            psi,
            alpha_values,
            &mut self.deriv_buf,
        );
        for i in 0..self.n {
            phases[i] += dt * self.deriv_buf[i];
        }
    }

    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    fn rk4_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha_values: &[f64],
        dt: f64,
    ) {
        let n = self.n;

        // k1
        compute_derivative(
            n,
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            zeta,
            psi,
            alpha_values,
            &mut self.k1,
        );

        // k2: phases + 0.5*dt*k1
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            zeta,
            psi,
            alpha_values,
            &mut self.k2,
        );

        // k3: phases + 0.5*dt*k2
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k2[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            zeta,
            psi,
            alpha_values,
            &mut self.k3,
        );

        // k4: phases + dt*k3
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + dt * self.k3[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            zeta,
            psi,
            alpha_values,
            &mut self.k4,
        );

        // phases += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        let dt6 = dt / 6.0;
        for i in 0..n {
            phases[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }

    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    fn rk45_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        row_ptr: &[usize],
        col_indices: &[usize],
        knm_values: &mut [f64],
        zeta: f64,
        psi: f64,
        alpha_values: &[f64],
    ) {
        let n = self.n;
        let max_reject = 3u32;
        let mut dt = self.last_dt;

        for _ in 0..=max_reject {
            // k1
            compute_derivative(
                n,
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k1,
            );

            // k2: phases + dt * a21 * k1
            for i in 0..n {
                self.tmp_phases[i] = phases[i] + dt * dp::A21 * self.k1[i];
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k2,
            );

            // k3: phases + dt * (a31*k1 + a32*k2)
            for i in 0..n {
                self.tmp_phases[i] = phases[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k3,
            );

            // k4: phases + dt * (a41*k1 + a42*k2 + a43*k3)
            for i in 0..n {
                self.tmp_phases[i] = phases[i]
                    + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]);
            }
            compute_derivative(
                n,
                &self.tmp_phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k4,
            );

            // k5: phases + dt * (a51*k1 + a52*k2 + a53*k3 + a54*k4)
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
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k5,
            );

            // k6: phases + dt * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
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
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k6,
            );

            // 5th-order solution (B5[6] = 0, so k7 does not contribute to y5)
            for i in 0..n {
                self.y5[i] = phases[i]
                    + dt * (dp::B5[0] * self.k1[i]
                        + dp::B5[2] * self.k3[i]
                        + dp::B5[3] * self.k4[i]
                        + dp::B5[4] * self.k5[i]
                        + dp::B5[5] * self.k6[i]);
            }

            // k7: evaluate derivative at y5 (FSAL property)
            compute_derivative(
                n,
                &self.y5,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                &mut self.k7,
            );

            // Error estimate using 4th-order weights (B4[6] = 1/40)
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
                // Accept step
                let factor = if err_norm > 0.0 {
                    (0.9 * err_norm.powf(-0.2)).min(5.0)
                } else {
                    5.0
                };
                self.last_dt = (dt * factor).min(self.dt * 10.0);
                phases.copy_from_slice(&self.y5[..n]);
                return;
            }

            // Reject — shrink dt and retry
            let factor = (0.9 * err_norm.powf(-0.25)).max(0.2);
            dt *= factor;
        }

        // Exhausted retries, accept current result
        self.last_dt = dt;
        phases.copy_from_slice(&self.y5[..n]);
    }
}

/// dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn compute_derivative(
    n: usize,
    theta: &[f64],
    omegas: &[f64],
    row_ptr: &[usize],
    col_indices: &[usize],
    knm_values: &mut [f64],
    zeta: f64,
    psi: f64,
    alpha_values: &[f64],
    out: &mut [f64],
) {
    for i in 0..n {
        let mut coupling_sum = 0.0;
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for idx in start..end {
            let j = col_indices[idx];
            coupling_sum += knm_values[idx] * (theta[j] - theta[i] - alpha_values[idx]).sin();
        }
        out[i] = omegas[i] + coupling_sum;
        if zeta != 0.0 {
            out[i] += zeta * (psi - theta[i]).sin();
        }
    }
}

/// Wrap phases to [0, 2π).
fn wrap_phases(phases: &mut [f64]) {
    for p in phases.iter_mut() {
        *p = p.rem_euclid(std::f64::consts::TAU);
    }
}
