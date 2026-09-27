// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Splitting boundary

//! Stateful and trajectory Python calls for Strang splitting of phase dynamics.

use crate::call_arguments::CallArguments;
use crate::spo_err;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::splitting;
use spo_types::IntegrationConfig;

// ─── PySplittingStepper ──────────────────────────────────────────────────

/// Stateful integrator preserving the model's configured timestep and Python state contract.
#[pyclass(name = "PySplittingStepper")]
pub(crate) struct PySplittingStepper {
    inner: splitting::SplittingStepper,
}

#[pymethods]
impl PySplittingStepper {
    #[new]
    #[pyo3(signature = (n, dt = 0.01))]
    fn new(n: usize, dt: f64) -> PyResult<Self> {
        let config = IntegrationConfig {
            dt,
            ..Default::default()
        };
        let inner = splitting::SplittingStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    fn step<'py>(
        &mut self,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: PyReadonlyArray1<'py, f64>,
        alpha: PyReadonlyArray1<'py, f64>,
        zeta: f64,
        psi: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let py = phases.py();
        let mut p = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = knm
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .step(&mut p, o, k, a, zeta, psi)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    /// Evolve validated splitting state from the original positional or keyword parameters.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, phases, omegas, knm, alpha, zeta, psi, n_steps)")]
    fn run<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let call = CallArguments::bind(
            args,
            kwargs,
            &["phases", "omegas", "knm", "alpha", "zeta", "psi", "n_steps"],
            "run",
        )?;
        let py = args.py();
        let phases: PyReadonlyArray1<'py, f64> = call.extract(0)?;
        let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
        let knm: PyReadonlyArray1<'py, f64> = call.extract(2)?;
        let alpha: PyReadonlyArray1<'py, f64> = call.extract(3)?;
        let zeta: f64 = call.extract(4)?;
        let psi: f64 = call.extract(5)?;
        let n_steps: usize = call.extract(6)?;
        let mut p = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = knm
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .run(&mut p, o, k, a, zeta, psi, n_steps)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    fn order_parameter(&self) -> (f64, f64) {
        self.inner.order_parameter()
    }
}

/// Stateful and trajectory Python calls for Strang splitting of phase dynamics.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(phases, omegas, knm, alpha, _n, zeta, psi, dt, n_steps)")]
pub(crate) fn splitting_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "phases", "omegas", "knm", "alpha", "_n", "zeta", "psi", "dt", "n_steps",
        ],
        "splitting_run_rust",
    )?;
    let py = args.py();
    let phases: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let knm: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let alpha: PyReadonlyArray1<'py, f64> = call.extract(3)?;
    let _n: usize = call.extract(4)?;
    let zeta: f64 = call.extract(5)?;
    let psi: f64 = call.extract(6)?;
    let dt: f64 = call.extract(7)?;
    let n_steps: usize = call.extract(8)?;
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = splitting::splitting_run(p, o, k, a, zeta, psi, dt, n_steps);
    Ok(PyArray1::from_vec(py, result).into())
}
