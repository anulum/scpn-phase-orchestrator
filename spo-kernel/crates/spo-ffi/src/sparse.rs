// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Sparse stepper NumPy buffer boundary

//! Sparse integration with snapshot inputs and writable plasticity.

use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::{plasticity::PlasticityModel, sparse_upde::SparseUPDEStepper};
use spo_types::{IntegrationConfig, Method};

use crate::spo_err;

/// Copy a contiguous readonly input and release its NumPy borrow guard.
fn snapshot_input<T: Element + Copy>(
    input: PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<Vec<T>> {
    input
        .as_slice()
        .map(<[T]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} not contiguous")))
}

/// Stateful sparse solver with entry-time snapshots and writable CSR coupling.
#[pyclass(name = "PySparseUPDEStepper")]
pub(crate) struct PySparseUPDEStepper {
    inner: SparseUPDEStepper,
}

#[pymethods]
impl PySparseUPDEStepper {
    /// Create a sparse solver with the configured step, method and tolerances.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Positive oscillator count.
    /// dt : float
    ///     Positive integration timestep in seconds.
    /// method : str
    ///     Integration method: euler, rk4 or rk45.
    /// n_substeps : int
    ///     Positive subdivision count for fixed-step methods.
    /// atol : float
    ///     Positive absolute tolerance for adaptive RK45.
    /// rtol : float
    ///     Positive relative tolerance for adaptive RK45.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A method or integration parameter is invalid.
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
        let m = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            "rk45" => Method::RK45,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt,
            method: m,
            n_substeps,
            atol,
            rtol,
        };
        let inner = SparseUPDEStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Enable in-place coupling plasticity after each native step.
    #[pyo3(signature = (lr, decay = 0.0, modulator = 1.0))]
    fn set_plasticity(&mut self, lr: f64, decay: f64, modulator: f64) -> PyResult<()> {
        self.inner.plasticity = Some(PlasticityModel::new(lr, decay).map_err(spo_err)?);
        self.inner.modulator = modulator;
        Ok(())
    }

    /// Keep coupling values fixed across subsequent steps.
    fn disable_plasticity(&mut self) {
        self.inner.plasticity = None;
    }

    /// Advance once using snapshots of every readonly input before borrowing coupling.
    ///
    /// Readonly coupling or conflicting external borrows raise ValueError.
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        row_ptr: PyReadonlyArray1<'py, usize>,
        col_indices: PyReadonlyArray1<'py, usize>,
        knm_values: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha_values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = snapshot_input(phases, "phases")?;
        let p_w = snapshot_input(omegas, "omegas")?;
        let rp = snapshot_input(row_ptr, "row_ptr")?;
        let ci = snapshot_input(col_indices, "col_indices")?;
        let av = snapshot_input(alpha_values, "alpha_values")?;
        let mut kv_bound = knm_values
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let kv = kv_bound
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        self.inner
            .step(&mut p_out, &p_w, &rp, &ci, kv, zeta, psi, &av)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    /// Advance repeatedly with fixed entry-time readonly snapshots.
    ///
    /// Coupling remains writable for plasticity throughout the run.
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        row_ptr: PyReadonlyArray1<'py, usize>,
        col_indices: PyReadonlyArray1<'py, usize>,
        knm_values: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha_values: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = snapshot_input(phases, "phases")?;
        let p_w = snapshot_input(omegas, "omegas")?;
        let rp = snapshot_input(row_ptr, "row_ptr")?;
        let ci = snapshot_input(col_indices, "col_indices")?;
        let av = snapshot_input(alpha_values, "alpha_values")?;
        let mut kv_bound = knm_values
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let kv = kv_bound
            .as_slice_mut()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        self.inner
            .run(&mut p_out, &p_w, &rp, &ci, kv, zeta, psi, &av, n_steps)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    /// Return the configured oscillator count.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Return the RK45 next-step proposal or configured fixed timestep.
    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
    }

    /// Return the order parameter from the solver cached phase trigonometry.
    fn order_parameter(&self) -> (f64, f64) {
        self.inner.order_parameter()
    }
}
