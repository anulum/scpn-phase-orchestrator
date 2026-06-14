// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Swarmalator dynamics

use rayon::prelude::*;
use spo_types::{IntegrationConfig, SpoError, SpoResult};
use std::f64::consts::TAU;

/// Spatial phase oscillator stepper for swarmalator dynamics.
pub struct SwarmalatorStepper {
    n: usize,
    dim: usize,
    dt: f64,
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
    dx_buf: Vec<f64>,
    dtheta_buf: Vec<f64>,
}

impl std::fmt::Debug for SwarmalatorStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwarmalatorStepper")
            .field("n", &self.n)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl SwarmalatorStepper {
    /// Create a swarmalator stepper for `n` agents in `dim` spatial dimensions.
    ///
    /// # Errors
    /// Returns `InvalidDimension` when `n` is zero and propagates integration
    /// configuration validation failures.
    pub fn new(n: usize, dim: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;
        Ok(Self {
            n,
            dim,
            dt: config.dt,
            sin_theta: vec![0.0; n],
            cos_theta: vec![0.0; n],
            dx_buf: vec![0.0; n * dim],
            dtheta_buf: vec![0.0; n],
        })
    }

    /// Advance one coupled position/phase swarmalator timestep in place.
    ///
    /// # Errors
    /// Currently returns `Ok(())` after construction-time validation; the result
    /// type is retained for API parity with other Rust steppers.
    pub fn step(
        &mut self,
        pos: &mut [f64],
        phases: &mut [f64],
        omegas: &[f64],
        a: f64,
        b: f64,
        j: f64,
        k: f64,
    ) -> SpoResult<()> {
        let n = self.n;
        let dim = self.dim;
        let dt = self.dt;
        if pos.len() != n * dim || phases.len() != n || omegas.len() != n {
            return Err(SpoError::InvalidDimension(format!(
                "expected pos={}, phases={}, omegas={}; got pos={}, phases={}, omegas={}",
                n * dim,
                n,
                n,
                pos.len(),
                phases.len(),
                omegas.len()
            )));
        }
        if pos.iter().any(|v| !v.is_finite())
            || phases.iter().any(|v| !v.is_finite())
            || omegas.iter().any(|v| !v.is_finite())
            || !a.is_finite()
            || !b.is_finite()
            || !j.is_finite()
            || !k.is_finite()
        {
            return Err(SpoError::IntegrationDiverged(
                "swarmalator inputs contain NaN/Inf".into(),
            ));
        }
        let inv_n = 1.0 / n as f64;
        let eps = 1e-6;

        for i in 0..n {
            let (s, c) = phases[i].sin_cos();
            self.sin_theta[i] = s;
            self.cos_theta[i] = c;
        }

        let st = &*self.sin_theta;
        let ct = &*self.cos_theta;
        let p_slice = &*pos;

        let dx_chunks = self.dx_buf.par_chunks_mut(dim);
        let dt_iter = self.dtheta_buf.par_iter_mut();

        dx_chunks
            .zip(dt_iter)
            .enumerate()
            .for_each(|(i, (dx_i, dt_i))| {
                let mut local_dt = omegas[i];
                let ci = ct[i];
                let si = st[i];
                dx_i.fill(0.0);

                for jj in 0..n {
                    if i == jj {
                        continue;
                    }
                    let mut d2 = 0.0;
                    for d in 0..dim {
                        let delta = p_slice[jj * dim + d] - p_slice[i * dim + d];
                        d2 += delta * delta;
                    }
                    let dist = (d2 + eps).sqrt();
                    let idist = 1.0 / dist;
                    let cd = ct[jj] * ci + st[jj] * si;
                    let sd = st[jj] * ci - ct[jj] * si;
                    let attr = (a + j * cd) * idist;
                    // OHS canonical inverse-distance repulsion: b / |dx|^2.
                    let rep = b / (d2 + eps);
                    for d in 0..dim {
                        let delta = p_slice[jj * dim + d] - p_slice[i * dim + d];
                        dx_i[d] += delta * attr - delta * rep;
                    }
                    local_dt += k * inv_n * sd * idist;
                }
                *dt_i = local_dt;
            });

        for i in 0..n {
            for d in 0..dim {
                pos[i * dim + d] += dt * self.dx_buf[i * dim + d] * inv_n;
            }
            phases[i] = (phases[i] + dt * self.dtheta_buf[i]).rem_euclid(TAU);
        }
        Ok(())
    }

    /// Run swarmalator dynamics and return flattened position/phase traces.
    pub fn run(
        &mut self,
        pos: &mut [f64],
        phases: &mut [f64],
        omegas: &[f64],
        a: f64,
        b: f64,
        j: f64,
        k: f64,
        n_steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let dim = self.dim;
        let mut pt = Vec::with_capacity(n_steps * n * dim);
        let mut pht = Vec::with_capacity(n_steps * n);
        for _ in 0..n_steps {
            self.step(pos, phases, omegas, a, b, j, k)
                .expect("stepper init failed");
            pt.extend_from_slice(pos);
            pht.extend_from_slice(phases);
        }
        (pt, pht)
    }

    /// Return the current Kuramoto order parameter `(R, psi)` from cached phases.
    pub fn order_parameter(&self) -> (f64, f64) {
        crate::order_params::compute_order_parameter_from_sincos(&self.sin_theta, &self.cos_theta)
    }
}

#[allow(clippy::too_many_arguments)]
/// Run swarmalator dynamics from initial positions and phases.
pub fn swarmalator_run(
    pos_init: &[f64],
    phases_init: &[f64],
    omegas: &[f64],
    n: usize,
    dim: usize,
    dt: f64,
    a: f64,
    b: f64,
    j: f64,
    k: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = SwarmalatorStepper::new(
        n,
        dim,
        IntegrationConfig {
            dt,
            ..Default::default()
        },
    )
    .expect("stepper init failed");
    let mut p = pos_init.to_vec();
    let mut ph = phases_init.to_vec();
    let (pt, pht) = s.run(&mut p, &mut ph, omegas, a, b, j, k, n_steps);
    (p, ph, pt, pht)
}

#[cfg(test)]
mod swarmalator_run_tests {
    use super::*;

    #[test]
    fn swarmalator_run_zero_steps_returns_initial_state_and_empty_traces() {
        let pos = vec![0.0, 1.0];
        let phases = vec![0.2, 0.4];
        let omegas = vec![0.0, 0.0];

        let (new_pos, new_phases, pos_trace, phase_trace) =
            swarmalator_run(&pos, &phases, &omegas, 2, 1, 0.1, 1.0, 1.0, 1.0, 1.0, 0);

        assert_eq!(new_pos, pos);
        assert_eq!(new_phases, phases);
        assert!(pos_trace.is_empty());
        assert!(phase_trace.is_empty());
    }

    #[test]
    fn swarmalator_run_advances_uncoupled_phase_frequency() {
        let pos = vec![0.0, 1.0];
        let phases = vec![0.2, std::f64::consts::TAU - 0.05];
        let omegas = vec![1.0, 1.0];

        let (_new_pos, new_phases, pos_trace, phase_trace) =
            swarmalator_run(&pos, &phases, &omegas, 2, 1, 0.1, 0.0, 0.0, 0.0, 0.0, 1);

        assert!((new_phases[0] - 0.3).abs() < 1e-12);
        assert!((new_phases[1] - 0.05).abs() < 1e-12);
        assert_eq!(pos_trace.len(), 2);
        assert_eq!(phase_trace.len(), 2);
    }
}
