// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Torus boundary

//! Flat phase trajectories for the torus Kuramoto Python interface.

use crate::call_arguments::CallArguments;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::geometric;

/// Flat phase trajectories for the torus Kuramoto Python interface.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(phases, omegas, knm, alpha, _n, zeta, psi, dt, n_steps)")]
pub(crate) fn torus_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "phases", "omegas", "knm", "alpha", "_n", "zeta", "psi", "dt", "n_steps",
        ],
        "torus_run_rust",
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
    let result = geometric::torus_run(p, o, k, a, zeta, psi, dt, n_steps);
    Ok(PyArray1::from_vec(py, result).into())
}
