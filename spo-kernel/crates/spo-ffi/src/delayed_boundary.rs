// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Delayed boundary

//! Delayed Kuramoto trajectory calls with explicit delay and integration counts.

use crate::call_arguments::CallArguments;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::delay;

/// Delayed Kuramoto trajectory calls with explicit delay and integration counts.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(phases_init, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps)")]
pub(crate) fn delayed_kuramoto_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "phases_init",
            "omegas",
            "knm_flat",
            "alpha_flat",
            "n",
            "zeta",
            "psi",
            "dt",
            "delay_steps",
            "n_steps",
        ],
        "delayed_kuramoto_run_rust",
    )?;
    let py = args.py();
    let phases_init: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let alpha_flat: PyReadonlyArray1<'py, f64> = call.extract(3)?;
    let n: usize = call.extract(4)?;
    let zeta: f64 = call.extract(5)?;
    let psi: f64 = call.extract(6)?;
    let dt: f64 = call.extract(7)?;
    let delay_steps: usize = call.extract(8)?;
    let n_steps: usize = call.extract(9)?;
    let p = phases_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = delay::delayed_kuramoto_run(p, o, k, a, n, zeta, psi, dt, delay_steps, n_steps);
    Ok(PyArray1::from_vec(py, result))
}
