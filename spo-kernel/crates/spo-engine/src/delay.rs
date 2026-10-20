// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

use rayon::prelude::*;
use spo_types::{IntegrationConfig, SpoError, SpoResult};
use std::f64::consts::TAU;

/// Time-delayed Kuramoto stepper with ring-buffered sine/cosine history.
pub struct DelayedStepper {
    n: usize,
    dt: f64,
    delay_steps: usize,
    head: usize,
    steps_taken: usize,
    cached_order: (f64, f64),
    history_sincos: Vec<f64>,
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
    deriv_buf: Vec<f64>,
}

impl std::fmt::Debug for DelayedStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelayedStepper")
            .field("n", &self.n)
            .field("delay_steps", &self.delay_steps)
            .finish_non_exhaustive()
    }
}

impl DelayedStepper {
    /// Create a delayed Kuramoto stepper for `n` oscillators.
    ///
    /// # Errors
    /// Returns `InvalidDimension` when `n` is zero and propagates integration
    /// configuration validation failures.
    pub fn new(n: usize, delay_steps: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;
        let max_buf = delay_steps
            .checked_add(1)
            .ok_or_else(|| SpoError::InvalidDimension("delay history count overflow".into()))?;
        n.checked_mul(n)
            .ok_or_else(|| SpoError::InvalidDimension("delay matrix size overflow".into()))?;
        let history_len = max_buf
            .checked_mul(n)
            .and_then(|size| size.checked_mul(2))
            .ok_or_else(|| SpoError::InvalidDimension("delay history size overflow".into()))?;
        Ok(Self {
            n,
            dt: config.dt,
            delay_steps,
            head: 0,
            steps_taken: 0,
            cached_order: (0.0, 0.0),
            history_sincos: zero_buffer(history_len)?,
            sin_theta: zero_buffer(n)?,
            cos_theta: zero_buffer(n)?,
            deriv_buf: zero_buffer(n)?,
        })
    }

    /// Advance one delayed-coupling timestep in place.
    ///
    /// Accepted calls determine whether history is available. The legacy
    /// `step_idx` argument is retained for call compatibility and is ignored.
    ///
    /// # Errors
    /// Refuses invalid dimensions, nonfinite inputs and nonfinite computed phases.
    /// Refusal preserves caller phases, accepted history position and cached order.
    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
        _step_idx: usize,
    ) -> SpoResult<()> {
        let n = self.n;
        self.validate_inputs(phases, omegas, knm, alpha, zeta, psi)?;
        let max_buf = self.delay_steps + 1;
        for i in 0..n {
            let (s, c) = phases[i].sin_cos();
            self.sin_theta[i] = s;
            self.cos_theta[i] = c;
            self.history_sincos[self.head * 2 * n + 2 * i] = s;
            self.history_sincos[self.head * 2 * n + 2 * i + 1] = c;
        }
        let delayed_idx = if self.delay_steps > 0 && self.steps_taken >= self.delay_steps {
            (self.head + max_buf - self.delay_steps) % max_buf
        } else {
            self.head
        };
        let d_sc = &self.history_sincos[delayed_idx * 2 * n..(delayed_idx + 1) * 2 * n];
        let (zs_psi, zc_psi) = if zeta != 0.0 {
            let (s, c) = psi.sin_cos();
            (zeta * s, zeta * c)
        } else {
            (0.0, 0.0)
        };
        let alpha_zero = alpha.iter().all(|&a| a == 0.0);
        let st = &*self.sin_theta;
        let ct = &*self.cos_theta;

        if n >= 256 {
            self.deriv_buf
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, val)| {
                    let offset = i * n;
                    let k_row = &knm[offset..offset + n];
                    let ci = ct[i];
                    let si = st[i];
                    let mut coupling = 0.0;
                    if alpha_zero {
                        let (k_chunks, k_tail) = k_row.as_chunks::<8>();
                        let (d_sc_chunks, d_sc_tail) = d_sc.as_chunks::<16>();
                        let mut acc = 0.0;
                        for (kc, dsc8) in k_chunks.iter().zip(d_sc_chunks.iter()) {
                            acc += kc[0] * (dsc8[0] * ci - dsc8[1] * si)
                                + kc[1] * (dsc8[2] * ci - dsc8[3] * si)
                                + kc[2] * (dsc8[4] * ci - dsc8[5] * si)
                                + kc[3] * (dsc8[6] * ci - dsc8[7] * si)
                                + kc[4] * (dsc8[8] * ci - dsc8[9] * si)
                                + kc[5] * (dsc8[10] * ci - dsc8[11] * si)
                                + kc[6] * (dsc8[12] * ci - dsc8[13] * si)
                                + kc[7] * (dsc8[14] * ci - dsc8[15] * si);
                        }
                        coupling = acc;
                        for (&kj, dsc2) in k_tail.iter().zip(d_sc_tail.as_chunks::<2>().0) {
                            coupling += kj * (dsc2[0] * ci - dsc2[1] * si);
                        }
                    } else {
                        for j in 0..n {
                            let tj_d = d_sc[2 * j].atan2(d_sc[2 * j + 1]);
                            coupling += k_row[j] * (tj_d - phases[i] - alpha[offset + j]).sin();
                        }
                    }
                    *val = omegas[i]
                        + coupling
                        + (if zeta != 0.0 {
                            zs_psi * ci - zc_psi * si
                        } else {
                            0.0
                        });
                });
        } else {
            for i in 0..n {
                let offset = i * n;
                let k_row = &knm[offset..offset + n];
                let ci = ct[i];
                let si = st[i];
                let mut coupling = 0.0;
                if alpha_zero {
                    for j in 0..n {
                        coupling += k_row[j] * (d_sc[2 * j] * ci - d_sc[2 * j + 1] * si);
                    }
                } else {
                    for j in 0..n {
                        let tj_d = d_sc[2 * j].atan2(d_sc[2 * j + 1]);
                        coupling += k_row[j] * (tj_d - phases[i] - alpha[offset + j]).sin();
                    }
                }
                self.deriv_buf[i] = omegas[i]
                    + coupling
                    + (if zeta != 0.0 {
                        zs_psi * ci - zc_psi * si
                    } else {
                        0.0
                    });
            }
        }
        for (candidate, phase) in self.deriv_buf.iter_mut().zip(phases.iter()) {
            let raw = phase + self.dt * *candidate;
            if !raw.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "delay step produced nonfinite phases".into(),
                ));
            }
            let wrapped = raw.rem_euclid(TAU);
            *candidate = if wrapped >= TAU || wrapped == 0.0 {
                0.0
            } else {
                wrapped
            };
        }
        phases.copy_from_slice(&self.deriv_buf);
        self.cached_order = crate::order_params::compute_order_parameter(phases);
        self.head = (self.head + 1) % max_buf;
        self.steps_taken = self.steps_taken.saturating_add(1);
        Ok(())
    }

    /// Validate complete finite state and controls before touching retained history.
    fn validate_inputs(
        &self,
        phases: &[f64],
        omegas: &[f64],
        knm: &[f64],
        alpha: &[f64],
        zeta: f64,
        psi: f64,
    ) -> SpoResult<()> {
        let n = self.n;
        if phases.len() != n || omegas.len() != n {
            return Err(SpoError::InvalidDimension(format!(
                "expected phases and omegas length {n}, got {}/{}",
                phases.len(),
                omegas.len()
            )));
        }
        if knm.len() != n * n || alpha.len() != n * n {
            return Err(SpoError::InvalidDimension(format!(
                "expected knm/alpha length {} got {}/{}",
                n * n,
                knm.len(),
                alpha.len()
            )));
        }
        if phases.iter().any(|v| !v.is_finite())
            || omegas.iter().any(|v| !v.is_finite())
            || knm.iter().any(|v| !v.is_finite())
            || alpha.iter().any(|v| !v.is_finite())
            || !zeta.is_finite()
            || !psi.is_finite()
        {
            return Err(SpoError::IntegrationDiverged(
                "delay step inputs contain NaN/Inf".into(),
            ));
        }
        Ok(())
    }

    /// Advance the delayed Kuramoto system for `n_steps` in-place timesteps.
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::step`].
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
        self.validate_inputs(phases, omegas, knm, alpha, zeta, psi)?;
        for step_idx in 0..n_steps {
            self.step(phases, omegas, knm, alpha, zeta, psi, step_idx)?;
        }
        Ok(())
    }

    /// Return the current Kuramoto order parameter `(R, psi)` from cached phases.
    pub fn order_parameter(&self) -> (f64, f64) {
        self.cached_order
    }
}

/// Run delayed Kuramoto dynamics from an initial phase snapshot.
///
/// # Errors
/// Refuses invalid dimensions/configuration, unavailable history capacity and
/// nonfinite inputs or computed phases. Zero steps return a validated input copy.
pub fn delayed_kuramoto_run(
    phases_init: &[f64],
    omegas: &[f64],
    knm_flat: &[f64],
    alpha_flat: &[f64],
    n: usize,
    zeta: f64,
    psi: f64,
    dt: f64,
    delay_steps: usize,
    n_steps: usize,
) -> SpoResult<Vec<f64>> {
    let mut s = DelayedStepper::new(
        n,
        delay_steps.min(n_steps),
        IntegrationConfig {
            dt,
            ..Default::default()
        },
    )?;
    let mut p = phases_init.to_vec();
    s.run(&mut p, omegas, knm_flat, alpha_flat, zeta, psi, n_steps)?;
    Ok(p)
}

/// Allocate initialised numerical storage while reporting unavailable capacity.
fn zero_buffer(length: usize) -> SpoResult<Vec<f64>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(length).map_err(|error| {
        SpoError::InvalidDimension(format!("delay history capacity unavailable: {error}"))
    })?;
    buffer.resize(length, 0.0);
    Ok(buffer)
}

#[cfg(test)]
mod delayed_kuramoto_run_tests {
    use super::*;

    #[test]
    fn delayed_kuramoto_run_zero_steps_returns_initial_phases() {
        let phases = vec![0.2, 1.1];
        let omegas = vec![0.0, 0.0];
        let knm = vec![0.0; 4];
        let alpha = vec![0.0; 4];

        let result = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, 2, 0.0, 0.0, 0.01, 1, 0)
            .expect("valid zero-step inputs");

        assert_eq!(result, phases);
    }

    #[test]
    fn delayed_kuramoto_run_advances_uncoupled_natural_frequency() {
        let phases = vec![0.2, TAU - 0.05];
        let omegas = vec![1.0, 1.0];
        let knm = vec![0.0; 4];
        let alpha = vec![0.0; 4];

        let result = delayed_kuramoto_run(&phases, &omegas, &knm, &alpha, 2, 0.0, 0.0, 0.1, 1, 1)
            .expect("valid uncoupled rotation");

        assert!((result[0] - 0.3).abs() < 1e-12);
        assert!((result[1] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn rounded_endpoints_and_signed_zero_are_canonical() {
        let interior = f64::from_bits(TAU.to_bits() - 1);
        let phases = [-f64::from_bits(1), -0.0, interior];
        let result = delayed_kuramoto_run(
            &phases, &[0.0; 3], &[0.0; 9], &[0.0; 9], 3, 0.0, 0.0, 0.01, 1, 1,
        )
        .expect("finite endpoint inputs");
        assert_eq!(result, [0.0, 0.0, interior]);
        assert!(result[0].is_sign_positive());
        assert!(result[1].is_sign_positive());
    }

    #[test]
    fn refusal_preserves_accepted_history_and_cached_order() {
        let config = IntegrationConfig {
            dt: 0.01,
            ..Default::default()
        };
        let mut actual = DelayedStepper::new(2, 2, config.clone()).expect("valid config");
        let mut reference = DelayedStepper::new(2, 2, config).expect("valid config");
        let mut phases = [0.4, 1.1];
        let mut expected = phases;
        let omega = [1.0, 1.0];
        let coupling = [0.0, 0.5, 0.2, 0.0];
        let alpha = [0.01; 4];
        actual
            .step(&mut phases, &omega, &coupling, &alpha, 0.2, 0.3, 999)
            .expect("first actual step");
        reference
            .step(&mut expected, &omega, &coupling, &alpha, 0.2, 0.3, 0)
            .expect("first reference step");
        let order = actual.order_parameter();
        assert_eq!(order, crate::order_params::compute_order_parameter(&phases));
        let mut refused = [1.1, 0.4];
        assert!(actual
            .step(
                &mut refused,
                &[f64::MAX; 2],
                &coupling,
                &alpha,
                f64::MAX,
                std::f64::consts::FRAC_PI_2,
                0,
            )
            .is_err());
        assert_eq!(refused, [1.1, 0.4]);
        assert_eq!(actual.order_parameter(), order);
        for _ in 0..5 {
            actual
                .step(&mut phases, &omega, &coupling, &alpha, 0.2, 0.3, 0)
                .expect("recovered actual step");
            reference
                .step(&mut expected, &omega, &coupling, &alpha, 0.2, 0.3, 999)
                .expect("reference continuation");
            assert_eq!(phases, expected);
        }
    }

    #[test]
    fn repeated_stateful_batches_keep_the_same_history() {
        let config = IntegrationConfig {
            dt: 0.05,
            ..Default::default()
        };
        let mut actual = DelayedStepper::new(2, 2, config.clone()).expect("valid config");
        let mut reference = DelayedStepper::new(2, 2, config).expect("valid config");
        let mut phases = [0.4, 1.1];
        let mut expected = phases;
        let omega = [1.0, -0.7];
        let coupling = [0.0, 0.5, -0.2, 0.0];
        let alpha = [0.01; 4];
        for _ in 0..2 {
            actual
                .run(&mut phases, &omega, &coupling, &alpha, 0.2, 0.3, 3)
                .expect("stateful three-step batch");
        }
        reference
            .run(&mut expected, &omega, &coupling, &alpha, 0.2, 0.3, 6)
            .expect("continuous six-step reference");
        assert_eq!(phases, expected);
    }

    #[test]
    fn direct_batches_refuse_invalid_inputs_without_panicking() {
        assert!(delayed_kuramoto_run(&[], &[], &[], &[], 0, 0.0, 0.0, 0.01, 1, 0,).is_err());
        assert!(delayed_kuramoto_run(
            &[0.2],
            &[0.0; 2],
            &[0.0; 4],
            &[0.0; 4],
            2,
            0.0,
            0.0,
            0.01,
            1,
            0,
        )
        .is_err());
        assert!(
            delayed_kuramoto_run(&[f64::NAN], &[0.0], &[0.0], &[0.0], 1, 0.0, 0.0, 0.01, 1, 0,)
                .is_err()
        );
        for value in [0.0, -0.1, f64::INFINITY, f64::NAN] {
            assert!(DelayedStepper::new(
                1,
                1,
                IntegrationConfig {
                    dt: value,
                    ..Default::default()
                },
            )
            .is_err());
        }
        assert!(DelayedStepper::new(usize::MAX, 1, IntegrationConfig::default()).is_err());
        assert!(DelayedStepper::new(1, usize::MAX, IntegrationConfig::default()).is_err());
        assert!(
            DelayedStepper::new(1, isize::MAX as usize / 2, IntegrationConfig::default(),).is_err()
        );
        let initial = [-1.0, TAU + 1.0];
        let copy = delayed_kuramoto_run(
            &initial,
            &[0.0; 2],
            &[0.0; 4],
            &[0.0; 4],
            2,
            0.0,
            0.0,
            0.01,
            usize::MAX,
            0,
        )
        .expect("zero steps need no history allocation");
        assert_eq!(copy, initial);
        assert_ne!(copy.as_ptr(), initial.as_ptr());
    }

    #[test]
    fn scalar_and_parallel_paths_match_independent_full_history() {
        for n in [2, 256, 257] {
            for lag in [0.0, 0.17] {
                let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
                let omega: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.7).collect();
                let coupling: Vec<f64> = (0..n * n)
                    .map(|index| ((index * 3 % 7) as f64 - 3.0) * 0.04)
                    .collect();
                let alpha = vec![lag; n * n];
                let mut history = Vec::new();
                let mut expected = phases.clone();
                for step in 0..6 {
                    history.push(expected.clone());
                    let delayed = if step >= 2 {
                        &history[step - 2]
                    } else {
                        &expected
                    };
                    expected = (0..n)
                        .map(|i| {
                            let force: f64 = (0..n)
                                .map(|j| {
                                    coupling[i * n + j] * (delayed[j] - expected[i] - lag).sin()
                                })
                                .sum();
                            (expected[i]
                                + 0.01 * (omega[i] + force + 0.3 * (0.7 - expected[i]).sin()))
                            .rem_euclid(TAU)
                        })
                        .collect();
                }
                let actual = delayed_kuramoto_run(
                    &phases, &omega, &coupling, &alpha, n, 0.3, 0.7, 0.01, 2, 6,
                )
                .expect("valid scalar/parallel problem");
                assert!(actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, e)| (a - e).abs() < 1e-12));
            }
        }
    }
}
