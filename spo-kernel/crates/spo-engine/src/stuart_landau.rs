// SCPN Phase Orchestrator — Stuart-Landau Phase-Amplitude Integrator
//!
//! Phase: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
//! Amplitude: dr_i/dt = (μ_i - r_i²)·r_i + ε Σ_j K^r_ij · r_j · cos(θ_j - θ_i)
//!
//! Acebrón et al. 2005, Rev. Mod. Phys. 77(1).
//! State vector: [θ_0..θ_{N-1}, r_0..r_{N-1}] (length 2N).

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};

/// Dormand-Prince RK45 Butcher tableau (shared constants with upde.rs).
mod dp {
    pub(super) const A21: f64 = 1.0 / 5.0;
    pub(super) const A31: f64 = 3.0 / 40.0;
    pub(super) const A32: f64 = 9.0 / 40.0;
    pub(super) const A41: f64 = 44.0 / 45.0;
    pub(super) const A42: f64 = -56.0 / 15.0;
    pub(super) const A43: f64 = 32.0 / 9.0;
    pub(super) const A51: f64 = 19372.0 / 6561.0;
    pub(super) const A52: f64 = -25360.0 / 2187.0;
    pub(super) const A53: f64 = 64448.0 / 6561.0;
    pub(super) const A54: f64 = -212.0 / 729.0;
    pub(super) const A61: f64 = 9017.0 / 3168.0;
    pub(super) const A62: f64 = -355.0 / 33.0;
    pub(super) const A63: f64 = 46732.0 / 5247.0;
    pub(super) const A64: f64 = 49.0 / 176.0;
    pub(super) const A65: f64 = -5103.0 / 18656.0;

    pub(super) const B5: [f64; 6] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ];

    pub(super) const B4: [f64; 6] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
    ];
}

/// Stuart-Landau phase-amplitude integrator with pre-allocated scratch arrays.
pub struct StuartLandauStepper {
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
    y5: Vec<f64>,
    tmp_state: Vec<f64>,
}

impl std::fmt::Debug for StuartLandauStepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StuartLandauStepper")
            .field("n", &self.n)
            .field("dt", &self.dt)
            .field("n_substeps", &self.n_substeps)
            .field("method", &self.method)
            .field("last_dt", &self.last_dt)
            .finish_non_exhaustive()
    }
}

impl StuartLandauStepper {
    /// # Errors
    /// Returns `InvalidDimension` if n is 0, or propagates config validation errors.
    pub fn new(n: usize, config: IntegrationConfig) -> SpoResult<Self> {
        if n == 0 {
            return Err(SpoError::InvalidDimension("n must be > 0".into()));
        }
        config.validate()?;
        let dim = 2 * n;
        Ok(Self {
            n,
            dt: config.dt,
            n_substeps: config.n_substeps,
            method: config.method,
            atol: config.atol,
            rtol: config.rtol,
            last_dt: config.dt,
            deriv_buf: vec![0.0; dim],
            k1: vec![0.0; dim],
            k2: vec![0.0; dim],
            k3: vec![0.0; dim],
            k4: vec![0.0; dim],
            k5: vec![0.0; dim],
            k6: vec![0.0; dim],
            y5: vec![0.0; dim],
            tmp_state: vec![0.0; dim],
        })
    }

    /// Advance state `[θ; r]` in-place by one timestep.
    ///
    /// # Arguments
    /// - `state`: `[θ_0..θ_{N-1}, r_0..r_{N-1}]` (length 2N)
    /// - `omegas`: natural frequencies (N)
    /// - `mu`: bifurcation parameters (N)
    /// - `knm`: phase coupling matrix, row-major N×N
    /// - `knm_r`: amplitude coupling matrix, row-major N×N
    /// - `zeta`: external drive strength
    /// - `psi`: external drive phase
    /// - `alpha`: phase lag matrix, row-major N×N
    /// - `epsilon`: amplitude coupling scale
    ///
    /// # Errors
    /// Returns `InvalidDimension` on length mismatch or `IntegrationDiverged` on NaN/Inf.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        state: &mut [f64],
        omegas: &[f64],
        mu: &[f64],
        knm: &[f64],
        knm_r: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        epsilon: f64,
    ) -> SpoResult<()> {
        let n = self.n;
        let dim = 2 * n;
        if state.len() != dim {
            return Err(SpoError::InvalidDimension(format!(
                "state length {}, expected {dim}",
                state.len()
            )));
        }
        if omegas.len() != n || mu.len() != n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {n}, got omegas={} mu={}",
                omegas.len(),
                mu.len()
            )));
        }
        let nn = n * n;
        if knm.len() != nn || knm_r.len() != nn || alpha.len() != nn {
            return Err(SpoError::InvalidDimension(format!(
                "expected {nn}={n}*{n}, got knm={} knm_r={} alpha={}",
                knm.len(),
                knm_r.len(),
                alpha.len()
            )));
        }
        validate_finite_slice(state, "state")?;
        validate_finite_slice(omegas, "omegas")?;
        validate_finite_slice(mu, "mu")?;
        validate_finite_slice(knm, "knm")?;
        if !zeta.is_finite() || !psi.is_finite() || !epsilon.is_finite() {
            return Err(SpoError::IntegrationDiverged(
                "zeta/psi/epsilon contain NaN/Inf".into(),
            ));
        }

        match self.method {
            Method::RK45 => {
                self.rk45_step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon);
            }
            _ => {
                let sub_dt = self.dt / f64::from(self.n_substeps);
                for _ in 0..self.n_substeps {
                    match self.method {
                        Method::Euler => self.euler_step(
                            state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, sub_dt,
                        ),
                        Method::RK4 => self.rk4_step(
                            state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, sub_dt,
                        ),
                        Method::RK45 => unreachable!(),
                    }
                }
            }
        }

        post_step(self.n, state);
        Ok(())
    }

    /// Run multiple steps in-place.
    ///
    /// # Errors
    /// Propagates errors from `step()`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        state: &mut [f64],
        omegas: &[f64],
        mu: &[f64],
        knm: &[f64],
        knm_r: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        epsilon: f64,
        n_steps: u64,
    ) -> SpoResult<()> {
        for _ in 0..n_steps {
            self.step(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    #[must_use]
    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn euler_step(
        &mut self,
        state: &mut [f64],
        omegas: &[f64],
        mu: &[f64],
        knm: &[f64],
        knm_r: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        epsilon: f64,
        dt: f64,
    ) {
        compute_derivative(
            self.n,
            state,
            omegas,
            mu,
            knm,
            knm_r,
            zeta,
            psi,
            alpha,
            epsilon,
            &mut self.deriv_buf,
        );
        let dim = 2 * self.n;
        for i in 0..dim {
            state[i] += dt * self.deriv_buf[i];
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn rk4_step(
        &mut self,
        state: &mut [f64],
        omegas: &[f64],
        mu: &[f64],
        knm: &[f64],
        knm_r: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        epsilon: f64,
        dt: f64,
    ) {
        let dim = 2 * self.n;

        compute_derivative(
            self.n,
            state,
            omegas,
            mu,
            knm,
            knm_r,
            zeta,
            psi,
            alpha,
            epsilon,
            &mut self.k1,
        );

        for i in 0..dim {
            self.tmp_state[i] = state[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(
            self.n,
            &self.tmp_state,
            omegas,
            mu,
            knm,
            knm_r,
            zeta,
            psi,
            alpha,
            epsilon,
            &mut self.k2,
        );

        for i in 0..dim {
            self.tmp_state[i] = state[i] + 0.5 * dt * self.k2[i];
        }
        compute_derivative(
            self.n,
            &self.tmp_state,
            omegas,
            mu,
            knm,
            knm_r,
            zeta,
            psi,
            alpha,
            epsilon,
            &mut self.k3,
        );

        for i in 0..dim {
            self.tmp_state[i] = state[i] + dt * self.k3[i];
        }
        compute_derivative(
            self.n,
            &self.tmp_state,
            omegas,
            mu,
            knm,
            knm_r,
            zeta,
            psi,
            alpha,
            epsilon,
            &mut self.k4,
        );

        let dt6 = dt / 6.0;
        for i in 0..dim {
            state[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn rk45_step(
        &mut self,
        state: &mut [f64],
        omegas: &[f64],
        mu: &[f64],
        knm: &[f64],
        knm_r: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
        epsilon: f64,
    ) {
        let dim = 2 * self.n;
        let max_reject = 3u32;
        let mut dt = self.last_dt;

        for _ in 0..=max_reject {
            compute_derivative(
                self.n,
                state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k1,
            );

            for i in 0..dim {
                self.tmp_state[i] = state[i] + dt * dp::A21 * self.k1[i];
            }
            compute_derivative(
                self.n,
                &self.tmp_state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k2,
            );

            for i in 0..dim {
                self.tmp_state[i] = state[i] + dt * (dp::A31 * self.k1[i] + dp::A32 * self.k2[i]);
            }
            compute_derivative(
                self.n,
                &self.tmp_state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k3,
            );

            for i in 0..dim {
                self.tmp_state[i] = state[i]
                    + dt * (dp::A41 * self.k1[i] + dp::A42 * self.k2[i] + dp::A43 * self.k3[i]);
            }
            compute_derivative(
                self.n,
                &self.tmp_state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k4,
            );

            for i in 0..dim {
                self.tmp_state[i] = state[i]
                    + dt * (dp::A51 * self.k1[i]
                        + dp::A52 * self.k2[i]
                        + dp::A53 * self.k3[i]
                        + dp::A54 * self.k4[i]);
            }
            compute_derivative(
                self.n,
                &self.tmp_state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k5,
            );

            for i in 0..dim {
                self.tmp_state[i] = state[i]
                    + dt * (dp::A61 * self.k1[i]
                        + dp::A62 * self.k2[i]
                        + dp::A63 * self.k3[i]
                        + dp::A64 * self.k4[i]
                        + dp::A65 * self.k5[i]);
            }
            compute_derivative(
                self.n,
                &self.tmp_state,
                omegas,
                mu,
                knm,
                knm_r,
                zeta,
                psi,
                alpha,
                epsilon,
                &mut self.k6,
            );

            let mut err_norm: f64 = 0.0;
            for i in 0..dim {
                let ks = [
                    self.k1[i], self.k2[i], self.k3[i], self.k4[i], self.k5[i], self.k6[i],
                ];
                self.y5[i] = state[i]
                    + dt * (dp::B5[0] * ks[0]
                        + dp::B5[2] * ks[2]
                        + dp::B5[3] * ks[3]
                        + dp::B5[4] * ks[4]
                        + dp::B5[5] * ks[5]);
                let y4 = state[i]
                    + dt * (dp::B4[0] * ks[0]
                        + dp::B4[2] * ks[2]
                        + dp::B4[3] * ks[3]
                        + dp::B4[4] * ks[4]
                        + dp::B4[5] * ks[5]);
                let err_i = (self.y5[i] - y4).abs();
                let scale = self.atol + self.rtol * state[i].abs().max(self.y5[i].abs());
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
                state.copy_from_slice(&self.y5[..dim]);
                return;
            }

            let factor = (0.9 * err_norm.powf(-0.25)).max(0.2);
            dt *= factor;
        }

        self.last_dt = dt;
        state.copy_from_slice(&self.y5[..2 * self.n]);
    }
}

/// Stuart-Landau derivative: phase + amplitude ODEs.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn compute_derivative(
    n: usize,
    state: &[f64],
    omegas: &[f64],
    mu: &[f64],
    knm: &[f64],
    knm_r: &[f64],
    zeta: f64,
    psi: f64,
    alpha: &[f64],
    epsilon: f64,
    out: &mut [f64],
) {
    let theta = &state[..n];
    let r = &state[n..];

    for i in 0..n {
        let mut phase_coupling = 0.0;
        let mut amp_coupling = 0.0;
        for j in 0..n {
            let diff = theta[j] - theta[i];
            phase_coupling += knm[i * n + j] * (diff - alpha[i * n + j]).sin();
            // Clamp r_j >= 0: intermediate RK stages can go negative,
            // flipping the coupling sign (Python parity fix).
            amp_coupling += knm_r[i * n + j] * r[j].max(0.0) * diff.cos();
        }
        out[i] = omegas[i] + phase_coupling;
        if zeta != 0.0 {
            out[i] += zeta * (psi - theta[i]).sin();
        }
        let ri = r[i];
        out[n + i] = (mu[i] - ri * ri) * ri + epsilon * amp_coupling;
    }
}

/// Wrap phases to [0, 2π), clamp amplitudes to ≥ 0.
fn post_step(n: usize, state: &mut [f64]) {
    for p in state[..n].iter_mut() {
        *p = p.rem_euclid(std::f64::consts::TAU);
    }
    for a in state[n..].iter_mut() {
        if *a < 0.0 {
            *a = 0.0;
        }
    }
}

fn validate_finite_slice(s: &[f64], name: &str) -> SpoResult<()> {
    for &v in s {
        if !v.is_finite() {
            return Err(SpoError::IntegrationDiverged(format!(
                "{name} contains NaN/Inf"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::IntegrationConfig;

    fn make_stepper(n: usize) -> StuartLandauStepper {
        StuartLandauStepper::new(n, IntegrationConfig::default()).expect("valid config")
    }

    fn zero_mat(n: usize) -> Vec<f64> {
        vec![0.0; n * n]
    }

    #[test]
    fn zero_n_rejected() {
        assert!(StuartLandauStepper::new(0, IntegrationConfig::default()).is_err());
    }

    #[test]
    fn single_euler_step() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut state = vec![0.0; 2 * n];
        for i in 0..n {
            state[n + i] = 1.0; // r_i = 1
        }
        let omegas = vec![1.0; n];
        let mu = vec![1.0; n];
        let knm = zero_mat(n);
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);
        s.step(
            &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
        )
        .expect("step succeeds");
        // dθ = ω*dt = 0.01; dr = (1-1)*1 = 0
        for i in 0..n {
            assert!((state[i] - 0.01).abs() < 1e-12);
            assert!((state[n + i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn phases_bounded_after_step() {
        let n = 8;
        let mut s = make_stepper(n);
        let mut state = vec![0.0; 2 * n];
        for i in 0..n {
            state[i] = i as f64 * std::f64::consts::TAU / n as f64;
            state[n + i] = 1.0;
        }
        let omegas = vec![2.0; n];
        let mu = vec![1.0; n];
        let knm = vec![0.1; n * n];
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);
        for _ in 0..500 {
            s.step(
                &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
            )
            .expect("step succeeds");
        }
        for i in 0..n {
            assert!(
                (0.0..std::f64::consts::TAU).contains(&state[i]),
                "phase {} = {} out of [0, 2π)",
                i,
                state[i]
            );
        }
    }

    #[test]
    fn amplitude_non_negative() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut state = vec![0.0; 2 * n];
        for i in 0..n {
            state[n + i] = 0.01; // small r
        }
        let omegas = vec![1.0; n];
        let mu = vec![-2.0; n]; // subcritical → amplitude decays
        let knm = zero_mat(n);
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);
        for _ in 0..200 {
            s.step(
                &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
            )
            .expect("step succeeds");
        }
        for i in 0..n {
            assert!(state[n + i] >= 0.0, "r[{i}] = {} < 0", state[n + i]);
        }
    }

    #[test]
    fn limit_cycle_convergence() {
        // μ > 0 → r → √μ
        let n = 1;
        let mu_val = 4.0;
        let config = IntegrationConfig {
            dt: 0.01,
            method: Method::RK4,
            n_substeps: 1,
            ..IntegrationConfig::default()
        };
        let mut s = StuartLandauStepper::new(n, config).expect("valid");
        let mut state = vec![0.0, 0.5]; // θ=0, r=0.5
        let omegas = [1.0];
        let mu = [mu_val];
        let knm = [0.0];
        let knm_r = [0.0];
        let alpha = [0.0];
        for _ in 0..5000 {
            s.step(
                &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
            )
            .expect("step");
        }
        let expected_r = mu_val.sqrt();
        assert!(
            (state[1] - expected_r).abs() < 0.01,
            "r={} expected ≈{expected_r}",
            state[1]
        );
    }

    #[test]
    fn subcritical_decay() {
        // μ < 0 → r → 0
        let n = 1;
        let config = IntegrationConfig {
            dt: 0.01,
            method: Method::RK4,
            n_substeps: 1,
            ..IntegrationConfig::default()
        };
        let mut s = StuartLandauStepper::new(n, config).expect("valid");
        let mut state = vec![0.0, 1.0];
        let omegas = [1.0];
        let mu = [-1.0];
        let knm = [0.0];
        let knm_r = [0.0];
        let alpha = [0.0];
        for _ in 0..2000 {
            s.step(
                &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
            )
            .expect("step");
        }
        assert!(state[1] < 0.05, "r={} should decay toward 0", state[1]);
    }

    #[test]
    fn zero_epsilon_reduces_to_kuramoto() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut state = vec![0.0; 2 * n];
        for i in 0..n {
            state[i] = 0.1 * i as f64;
            state[n + i] = 1.0;
        }
        let omegas = vec![1.0; n];
        let mu = vec![1.0; n]; // at limit cycle
        let knm = vec![0.1; n * n];
        let knm_r = vec![99.0; n * n]; // huge knm_r, but epsilon=0 → ignored
        let alpha = zero_mat(n);

        s.step(
            &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
        )
        .expect("step");
        // Amplitudes unchanged from limit cycle (within rounding)
        for i in 0..n {
            assert!(
                (state[n + i] - 1.0).abs() < 0.01,
                "r[{i}]={} deviated despite epsilon=0",
                state[n + i]
            );
        }
    }

    #[test]
    fn dimension_mismatch() {
        let mut s = make_stepper(4);
        let mut state = vec![0.0; 6]; // wrong: should be 8
        let omegas = vec![1.0; 4];
        let mu = vec![1.0; 4];
        let knm = vec![0.0; 16];
        let knm_r = vec![0.0; 16];
        let alpha = vec![0.0; 16];
        assert!(s
            .step(&mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 1.0)
            .is_err());
    }

    #[test]
    fn nan_input_rejected() {
        let mut s = make_stepper(2);
        let mut state = vec![f64::NAN, 0.0, 1.0, 1.0];
        let omegas = vec![1.0; 2];
        let mu = vec![1.0; 2];
        let knm = vec![0.0; 4];
        let knm_r = vec![0.0; 4];
        let alpha = vec![0.0; 4];
        assert!(s
            .step(&mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 1.0)
            .is_err());
    }

    #[test]
    fn rk4_vs_euler_agreement() {
        let n = 4;
        let euler_cfg = IntegrationConfig {
            dt: 0.01,
            method: Method::Euler,
            n_substeps: 10,
            ..IntegrationConfig::default()
        };
        let rk4_cfg = IntegrationConfig {
            dt: 0.01,
            method: Method::RK4,
            n_substeps: 1,
            ..IntegrationConfig::default()
        };
        let mut se = StuartLandauStepper::new(n, euler_cfg).expect("valid");
        let mut sr = StuartLandauStepper::new(n, rk4_cfg).expect("valid");

        let omegas = vec![1.0; n];
        let mu = vec![1.0; n];
        let knm = zero_mat(n);
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);

        let init: Vec<f64> = (0..2 * n)
            .map(|i| if i < n { 0.1 * i as f64 } else { 1.0 })
            .collect();
        let mut state_e = init.clone();
        let mut state_r = init;

        // One macro-step for both
        se.step(
            &mut state_e,
            &omegas,
            &mu,
            &knm,
            &knm_r,
            0.0,
            0.0,
            &alpha,
            0.0,
        )
        .expect("euler");
        sr.step(
            &mut state_r,
            &omegas,
            &mu,
            &knm,
            &knm_r,
            0.0,
            0.0,
            &alpha,
            0.0,
        )
        .expect("rk4");

        for i in 0..2 * n {
            assert!(
                (state_e[i] - state_r[i]).abs() < 1e-4,
                "idx {i}: euler={} rk4={}",
                state_e[i],
                state_r[i]
            );
        }
    }

    #[test]
    fn rk45_adaptive() {
        let n = 4;
        let config = IntegrationConfig {
            dt: 0.01,
            method: Method::RK45,
            n_substeps: 1,
            atol: 1e-8,
            rtol: 1e-5,
        };
        let mut s = StuartLandauStepper::new(n, config).expect("valid");
        let mut state: Vec<f64> = (0..2 * n)
            .map(|i| if i < n { 0.1 * i as f64 } else { 1.0 })
            .collect();
        let omegas = vec![1.0; n];
        let mu = vec![1.0; n];
        let knm = zero_mat(n);
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);

        s.step(
            &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0,
        )
        .expect("rk45 step");
        // Verify phases bounded and amplitudes non-negative
        for i in 0..n {
            assert!((0.0..std::f64::consts::TAU).contains(&state[i]));
            assert!(state[n + i] >= 0.0);
        }
    }

    #[test]
    fn synchronisation_tendency() {
        let n = 8;
        let config = IntegrationConfig {
            dt: 0.01,
            method: Method::RK4,
            n_substeps: 1,
            ..IntegrationConfig::default()
        };
        let mut s = StuartLandauStepper::new(n, config).expect("valid");
        let mut state: Vec<f64> = (0..2 * n)
            .map(|i| if i < n { 0.1 + 0.02 * i as f64 } else { 1.0 })
            .collect();
        let omegas = vec![1.0; n];
        let mu = vec![1.0; n];
        let mut knm = vec![5.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let knm_r = zero_mat(n);
        let alpha = zero_mat(n);

        let r_before = order_param_r(&state[..n]);
        s.run(
            &mut state, &omegas, &mu, &knm, &knm_r, 0.0, 0.0, &alpha, 0.0, 1000,
        )
        .expect("run");
        let r_after = order_param_r(&state[..n]);

        assert!(
            r_after > r_before,
            "R should increase: {r_before:.4} → {r_after:.4}"
        );
    }

    fn order_param_r(phases: &[f64]) -> f64 {
        crate::order_params::compute_order_parameter(phases).0
    }
}
