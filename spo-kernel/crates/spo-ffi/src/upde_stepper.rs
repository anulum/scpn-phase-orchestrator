// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Dense stepper NumPy buffer boundary

//! Dense stateful integration with snapshot inputs and writable plasticity.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::{plasticity::PlasticityModel, upde::UPDEStepper};
use spo_types::{IntegrationConfig, Method};

use crate::spo_err;

/// Copy a contiguous readonly input and release its NumPy borrow guard.
fn snapshot_input(input: PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    input
        .as_slice()
        .map(<[f64]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} not contiguous")))
}

/// Dense NumPy integration boundary retaining native solver state.
#[pyclass(name = "PyUPDEStepper")]
pub(crate) struct PyUPDEStepper {
    inner: UPDEStepper,
}

#[pymethods]
impl PyUPDEStepper {
    /// Create a Rust-backed UPDE stepper.
    ///
    /// Parameters
    /// ----------
    /// n
    ///     Number of oscillators in every phase, frequency, and coupling
    ///     input passed to `step` or `run`.
    /// dt
    ///     Base integration timestep in seconds.
    /// method
    ///     Integration method: `"euler"`, `"rk4"`, or `"rk45"`.
    /// n_substeps
    ///     Number of substeps retained in the integration configuration.
    /// atol
    ///     Absolute tolerance used by adaptive RK45.
    /// rtol
    ///     Relative tolerance used by adaptive RK45.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `method` is unknown or the integration configuration is
    ///     rejected by the Rust kernel.
    #[new]
    #[pyo3(signature = (n, dt = 0.01, method = "euler", n_substeps = 1, atol = 1e-6, rtol = 1e-3))]
    fn new(
        n: usize,
        dt: f64,
        method: &str,
        n_substeps: u32,
        atol: f64,
        rtol: f64,
    ) -> PyResult<Self> {
        let integration_method = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            "rk45" => Method::RK45,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt,
            method: integration_method,
            n_substeps,
            atol,
            rtol,
        };
        let inner = UPDEStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Advance once using entry-time snapshots of readonly inputs.
    ///
    /// Coupling remains writable for plasticity, including when input views
    /// overlap it. Borrow, writeability and contiguity faults raise ValueError.
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut phase_values = snapshot_input(phases, "phases")?;
        let frequencies = snapshot_input(omegas, "omegas")?;
        let lag_values = snapshot_input(alpha, "alpha")?;
        let mut coupling_guard = knm
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let coupling_values = coupling_guard
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .step(
                &mut phase_values,
                &frequencies,
                coupling_values,
                zeta,
                psi,
                &lag_values,
            )
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, phase_values))
    }

    /// Advance repeatedly with fixed entry-time phases, frequencies and lag.
    ///
    /// Plasticity updates the caller's coupling after each step. All readonly
    /// snapshots remain fixed throughout the run, even for overlapping views.
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut phase_values = snapshot_input(phases, "phases")?;
        let frequencies = snapshot_input(omegas, "omegas")?;
        let lag_values = snapshot_input(alpha, "alpha")?;
        let mut coupling_guard = knm
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let coupling_values = coupling_guard
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .run(
                &mut phase_values,
                &frequencies,
                coupling_values,
                zeta,
                psi,
                &lag_values,
                n_steps,
            )
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, phase_values))
    }

    /// Advance with an entry-time frequency schedule and lag snapshot.
    ///
    /// Coupling remains writable and plasticity updates persist in the input.
    #[allow(clippy::too_many_arguments)]
    fn run_omega_schedule<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omega_schedule: PyReadonlyArray1<'py, f64>,
        knm: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut phase_values = snapshot_input(phases, "phases")?;
        let schedule = snapshot_input(omega_schedule, "omega_schedule")?;
        let lag_values = snapshot_input(alpha, "alpha")?;
        let mut coupling_guard = knm
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let coupling_values = coupling_guard
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .run_omega_schedule(
                &mut phase_values,
                &schedule,
                coupling_values,
                zeta,
                psi,
                &lag_values,
                n_steps,
            )
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, phase_values))
    }

    /// Advance with entry-time frequency, velocity and lag snapshots.
    ///
    /// Each Doppler correction uses the current writable coupling, including
    /// changes from plasticity in earlier steps.
    #[allow(clippy::too_many_arguments)]
    fn run_doppler_schedule<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omega_schedule: PyReadonlyArray1<'py, f64>,
        knm: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
        velocity_schedule: PyReadonlyArray1<'py, f64>,
        doppler_strength: f64,
        doppler_epsilon: f64,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut phase_values = snapshot_input(phases, "phases")?;
        let schedule = snapshot_input(omega_schedule, "omega_schedule")?;
        let velocities = snapshot_input(velocity_schedule, "velocity_schedule")?;
        let lag_values = snapshot_input(alpha, "alpha")?;
        let mut coupling_guard = knm
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let coupling_values = coupling_guard
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .run_doppler_schedule(
                &mut phase_values,
                &schedule,
                coupling_values,
                zeta,
                psi,
                &lag_values,
                &velocities,
                doppler_strength,
                doppler_epsilon,
                n_steps,
            )
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, phase_values))
    }

    /// Run n_steps with omega, velocity, and axial position schedules.
    #[allow(clippy::too_many_arguments)]
    fn run_moving_frame_schedule<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        positions: PyReadonlyArray1<'py, f64>,
        omega_schedule: PyReadonlyArray1<'py, f64>,
        knm: PyReadonlyArray1<'py, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
        velocity_schedule: PyReadonlyArray1<'py, f64>,
        spatial_k_base: f64,
        spatial_decay_form: u8,
        spatial_decay_exponent: f64,
        spatial_decay_length_scale: f64,
        spatial_epsilon: f64,
        doppler_strength: f64,
        doppler_epsilon: f64,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut phase_values = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let mut position_values = positions
            .to_vec()
            .map_err(|_| PyValueError::new_err("positions not contiguous"))?;
        let schedule = omega_schedule
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let coupling_values = knm
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let lag_values = alpha
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let velocities = velocity_schedule
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .run_moving_frame_schedule(
                &mut phase_values,
                &mut position_values,
                schedule,
                coupling_values,
                zeta,
                psi,
                lag_values,
                velocities,
                spatial_k_base,
                spatial_decay_form,
                spatial_decay_exponent,
                spatial_decay_length_scale,
                spatial_epsilon,
                doppler_strength,
                doppler_epsilon,
                n_steps,
            )
            .map_err(spo_err)?;
        phase_values.extend_from_slice(&position_values);
        Ok(PyArray1::from_vec(py, phase_values))
    }

    /// Enable the native Hebbian rule, writing updates into caller coupling.
    #[pyo3(signature = (lr, decay = 0.0, modulator = 1.0))]
    fn set_plasticity(&mut self, lr: f64, decay: f64, modulator: f64) -> PyResult<()> {
        self.inner.plasticity = Some(PlasticityModel::new(lr, decay).map_err(spo_err)?);
        self.inner.modulator = modulator;
        Ok(())
    }

    /// Stop updating caller coupling without changing phase integration.
    fn disable_plasticity(&mut self) {
        self.inner.plasticity = None;
    }

    /// Return the configured oscillator count.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Return the last adaptive timestep retained by the solver.
    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
    }

    /// Return coherence and mean phase from the last integrated state.
    fn order_parameter(&self) -> (f64, f64) {
        self.inner.order_parameter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_readonly_borrow_refuses_without_panicking() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "spo_kernel")?;
            module.add_class::<PyUPDEStepper>()?;
            let stepper = module.getattr("PyUPDEStepper")?.call1((2,))?;
            let phases = PyArray1::from_vec(py, vec![0.1, 0.7]);
            let frequencies = PyArray1::from_vec(py, vec![0.8, 1.2]);
            let coupling = PyArray1::from_vec(py, vec![0.0, 0.3, 0.4, 0.0]);
            let lag = PyArray1::from_vec(py, vec![0.0; 4]);
            let external_guard = coupling
                .try_readonly()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let before: (f64, f64) = stepper.call_method0("order_parameter")?.extract()?;
            let error = stepper
                .call_method1("step", (&phases, &frequencies, &coupling, 0.0, 0.0, &lag))
                .expect_err("external immutable coupling borrow must refuse writes");
            assert!(error.is_instance_of::<PyValueError>(py));
            let after: (f64, f64) = stepper.call_method0("order_parameter")?.extract()?;
            assert_eq!(after, before);
            drop(external_guard);
            let result = stepper
                .call_method1("step", (&phases, &frequencies, &coupling, 0.0, 0.0, &lag))?
                .cast_into::<PyArray1<f64>>()?;
            assert_eq!(result.len()?, 2);
            Ok(())
        })
    }
}
