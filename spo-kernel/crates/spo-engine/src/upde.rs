// SCPN Phase Orchestrator — UPDE Kuramoto Integrator
//!
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
//!
//! Supports Euler and RK4 with pre-allocated scratch arrays for
//! zero-alloc hot-path execution at dynamic N (4-256 oscillators).

use spo_types::{IntegrationConfig, Method, SpoError, SpoResult};

pub struct UPDEStepper {
    n: usize,
    dt: f64,
    method: Method,
    // Pre-allocated scratch buffers
    deriv_buf: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp_phases: Vec<f64>,
}

impl UPDEStepper {
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
            method: config.method,
            deriv_buf: vec![0.0; n],
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            tmp_phases: vec![0.0; n],
        })
    }

    /// Advance phases in-place by one timestep.
    ///
    /// `knm` is row-major N×N, `alpha` is row-major N×N phase lags.
    ///
    /// # Errors
    /// Returns `InvalidDimension` on length mismatch or `IntegrationDiverged` on NaN/Inf input.
    pub fn step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
    ) -> SpoResult<()> {
        let n = self.n;
        if phases.len() != n || omegas.len() != n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {n}, got phases={} omegas={}",
                phases.len(),
                omegas.len()
            )));
        }
        if knm.len() != n * n || alpha.len() != n * n {
            return Err(SpoError::InvalidDimension(format!(
                "expected {}={n}*{n}, got knm={} alpha={}",
                n * n,
                knm.len(),
                alpha.len()
            )));
        }
        for &th in phases.iter() {
            if !th.is_finite() {
                return Err(SpoError::IntegrationDiverged(
                    "input phases contain NaN/Inf".into(),
                ));
            }
        }
        if !zeta.is_finite() || !psi.is_finite() {
            return Err(SpoError::IntegrationDiverged(
                "zeta/psi contain NaN/Inf".into(),
            ));
        }

        match self.method {
            Method::Euler => self.euler_step(phases, omegas, knm, zeta, psi, alpha),
            Method::RK4 => self.rk4_step(phases, omegas, knm, zeta, psi, alpha),
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
        knm: &[f64],
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

    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    #[allow(clippy::needless_range_loop)]
    fn euler_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
    ) {
        compute_derivative(
            self.n,
            phases,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
            &mut self.deriv_buf,
        );
        let dt = self.dt;
        for i in 0..self.n {
            phases[i] += dt * self.deriv_buf[i];
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn rk4_step(
        &mut self,
        phases: &mut [f64],
        omegas: &[f64],
        knm: &[f64],
        zeta: f64,
        psi: f64,
        alpha: &[f64],
    ) {
        let n = self.n;
        let dt = self.dt;

        // k1
        compute_derivative(n, phases, omegas, knm, zeta, psi, alpha, &mut self.k1);

        // k2: phases + 0.5*dt*k1
        for i in 0..n {
            self.tmp_phases[i] = phases[i] + 0.5 * dt * self.k1[i];
        }
        compute_derivative(
            n,
            &self.tmp_phases,
            omegas,
            knm,
            zeta,
            psi,
            alpha,
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
            knm,
            zeta,
            psi,
            alpha,
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
            knm,
            zeta,
            psi,
            alpha,
            &mut self.k4,
        );

        // phases += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        let dt6 = dt / 6.0;
        for i in 0..n {
            phases[i] += dt6 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }
    }
}

/// dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn compute_derivative(
    n: usize,
    theta: &[f64],
    omegas: &[f64],
    knm: &[f64],
    zeta: f64,
    psi: f64,
    alpha: &[f64],
    out: &mut [f64],
) {
    for i in 0..n {
        let mut coupling_sum = 0.0;
        for j in 0..n {
            coupling_sum += knm[i * n + j] * (theta[j] - theta[i] - alpha[i * n + j]).sin();
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
        let knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .unwrap();
        // dθ = ω*dt = 1.0 * 0.01
        for &p in &phases {
            assert!((p - 0.01).abs() < 1e-12);
        }
    }

    #[test]
    fn phases_bounded_after_step() {
        let n = 8;
        let mut s = make_stepper(n);
        let mut phases: Vec<f64> = (0..n)
            .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
            .collect();
        let omegas = vec![2.0; n];
        let knm = vec![0.1; n * n];
        let alpha = zero_alpha(n);
        for _ in 0..500 {
            s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
                .unwrap();
        }
        for &p in &phases {
            assert!(
                (0.0..std::f64::consts::TAU).contains(&p),
                "phase {p} out of range"
            );
        }
    }

    #[test]
    fn nan_input_rejected() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut phases = vec![f64::NAN; n];
        let omegas = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        assert!(s
            .step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut s = make_stepper(4);
        let mut phases = vec![0.0; 3];
        let omegas = vec![1.0; 4];
        let knm = vec![0.0; 16];
        let alpha = vec![0.0; 16];
        assert!(s
            .step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .is_err());
    }

    #[test]
    fn rk4_single_step() {
        let n = 4;
        let config = IntegrationConfig {
            dt: 0.01,
            method: Method::RK4,
            n_substeps: 1,
        };
        let mut s = UPDEStepper::new(n, config).unwrap();
        let mut phases = vec![0.0; n];
        let omegas = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        s.step(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha)
            .unwrap();
        // Without coupling, RK4 on dθ=ω gives θ=ω*dt exactly
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
        // Strong uniform coupling
        let mut knm = vec![5.0; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        let alpha = zero_alpha(n);

        let r_before = crate::order_params::compute_order_parameter(&phases).0;
        s.run(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha, 1000)
            .unwrap();
        let r_after = crate::order_params::compute_order_parameter(&phases).0;

        assert!(
            r_after > r_before,
            "R should increase: {r_before:.4} → {r_after:.4}"
        );
    }

    #[test]
    fn external_drive() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut phases = vec![0.0; n];
        let omegas = vec![0.0; n];
        let knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        // zeta = 1, psi = pi/2 → dθ = sin(pi/2 - 0) = 1.0
        s.step(
            &mut phases,
            &omegas,
            &knm,
            1.0,
            std::f64::consts::FRAC_PI_2,
            &alpha,
        )
        .unwrap();
        for &p in &phases {
            assert!((p - 0.01).abs() < 1e-6, "expected ~0.01, got {p}");
        }
    }

    #[test]
    fn run_zero_steps_noop() {
        let n = 4;
        let mut s = make_stepper(n);
        let mut phases = vec![0.5; n];
        let original = phases.clone();
        let omegas = vec![1.0; n];
        let knm = vec![0.0; n * n];
        let alpha = zero_alpha(n);
        s.run(&mut phases, &omegas, &knm, 0.0, 0.0, &alpha, 0)
            .unwrap();
        assert_eq!(phases, original);
    }

    #[test]
    fn nan_zeta_rejected() {
        let mut s = make_stepper(4);
        let mut phases = vec![0.0; 4];
        let omegas = vec![1.0; 4];
        let knm = vec![0.0; 16];
        let alpha = zero_alpha(4);
        assert!(s
            .step(&mut phases, &omegas, &knm, f64::NAN, 0.0, &alpha)
            .is_err());
    }

    #[test]
    fn inf_psi_rejected() {
        let mut s = make_stepper(4);
        let mut phases = vec![0.0; 4];
        let omegas = vec![1.0; 4];
        let knm = vec![0.0; 16];
        let alpha = zero_alpha(4);
        assert!(s
            .step(&mut phases, &omegas, &knm, 0.0, f64::INFINITY, &alpha)
            .is_err());
    }

    #[test]
    fn phase_lag_alpha_effect() {
        let n = 2;
        let mut s = make_stepper(n);
        let knm = vec![0.0, 1.0, 1.0, 0.0];

        // Without lag
        let mut p1 = vec![0.0, 1.0];
        let omegas = vec![0.0; n];
        let alpha_zero = vec![0.0; 4];
        s.step(&mut p1, &omegas, &knm, 0.0, 0.0, &alpha_zero)
            .unwrap();

        // With lag alpha[0,1] = 0.5
        let mut p2 = vec![0.0, 1.0];
        let alpha_lag = vec![0.0, 0.5, -0.5, 0.0];
        let mut s2 = make_stepper(n);
        s2.step(&mut p2, &omegas, &knm, 0.0, 0.0, &alpha_lag)
            .unwrap();

        // Phase updates should differ due to lag
        assert!((p1[0] - p2[0]).abs() > 1e-6);
    }
}
