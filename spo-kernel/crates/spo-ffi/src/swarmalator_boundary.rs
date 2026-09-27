// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Swarmalator boundary

//! Joint position and phase evolution through the Python swarmalator contract.

use crate::call_arguments::CallArguments;
use crate::return_types::{ArrayPair, ArrayQuartet};
use crate::spo_err;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::swarmalator;
use spo_types::IntegrationConfig;

// ─── PySwarmalatorStepper ────────────────────────────────────────────────

/// Stateful integrator preserving the model's configured timestep and Python state contract.
#[pyclass(name = "PySwarmalatorStepper")]
pub(crate) struct PySwarmalatorStepper {
    inner: swarmalator::SwarmalatorStepper,
}

#[pymethods]
impl PySwarmalatorStepper {
    #[new]
    #[pyo3(signature = (n, dim, dt = 0.01))]
    fn new(n: usize, dim: usize, dt: f64) -> PyResult<Self> {
        let config = IntegrationConfig {
            dt,
            ..Default::default()
        };
        let inner = swarmalator::SwarmalatorStepper::new(n, dim, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Evolve validated swarmalator state from the original positional or keyword parameters.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, pos, phases, omegas, a, b, j, k)")]
    fn step<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<ArrayPair<'py>> {
        let call = CallArguments::bind(
            args,
            kwargs,
            &["pos", "phases", "omegas", "a", "b", "j", "k"],
            "step",
        )?;
        let py = args.py();
        let pos: PyReadonlyArray1<'py, f64> = call.extract(0)?;
        let phases: PyReadonlyArray1<'py, f64> = call.extract(1)?;
        let omegas: PyReadonlyArray1<'py, f64> = call.extract(2)?;
        let a: f64 = call.extract(3)?;
        let b: f64 = call.extract(4)?;
        let j: f64 = call.extract(5)?;
        let k: f64 = call.extract(6)?;
        let mut p_pos = pos
            .to_vec()
            .map_err(|_| PyValueError::new_err("pos not contiguous"))?;
        let mut p_phases = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .step(&mut p_pos, &mut p_phases, o, a, b, j, k)
            .map_err(spo_err)?;

        Ok((
            PyArray1::from_vec(py, p_pos),
            PyArray1::from_vec(py, p_phases),
        ))
    }

    fn order_parameter(&self) -> (f64, f64) {
        self.inner.order_parameter()
    }
}

/// Joint position and phase evolution through the Python swarmalator contract.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(pos_init, phases_init, omegas, n, dim, dt, a, b, j, k, n_steps)")]
pub(crate) fn swarmalator_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<ArrayQuartet<'py>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "pos_init",
            "phases_init",
            "omegas",
            "n",
            "dim",
            "dt",
            "a",
            "b",
            "j",
            "k",
            "n_steps",
        ],
        "swarmalator_run_rust",
    )?;
    let py = args.py();
    let pos_init: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let phases_init: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let n: usize = call.extract(3)?;
    let dim: usize = call.extract(4)?;
    let dt: f64 = call.extract(5)?;
    let a: f64 = call.extract(6)?;
    let b: f64 = call.extract(7)?;
    let j: f64 = call.extract(8)?;
    let k: f64 = call.extract(9)?;
    let n_steps: usize = call.extract(10)?;
    let p = pos_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ph = phases_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (fp, fph, pt, pht) =
        swarmalator::swarmalator_run(p, ph, o, n, dim, dt, a, b, j, k, n_steps);
    Ok((
        PyArray1::from_vec(py, fp),
        PyArray1::from_vec(py, fph),
        PyArray1::from_vec(py, pt),
        PyArray1::from_vec(py, pht),
    ))
}
