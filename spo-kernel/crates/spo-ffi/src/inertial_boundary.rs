// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Inertial boundary

//! Swing-equation state and trajectory calls with legacy Python argument names.

use crate::call_arguments::CallArguments;
use crate::return_types::{ArrayPair, ArrayQuartet};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::inertial;

/// Swing-equation state and trajectory calls with legacy Python argument names.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(theta, omega_dot, power, knm_flat, inertia, damping, n, dt)")]
pub(crate) fn inertial_step_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<ArrayPair<'py>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "theta",
            "omega_dot",
            "power",
            "knm_flat",
            "inertia",
            "damping",
            "n",
            "dt",
        ],
        "inertial_step_rust",
    )?;
    let py = args.py();
    let theta: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omega_dot: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let power: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(3)?;
    let inertia: PyReadonlyArray1<'py, f64> = call.extract(4)?;
    let damping: PyReadonlyArray1<'py, f64> = call.extract(5)?;
    let n: usize = call.extract(6)?;
    let dt: f64 = call.extract(7)?;
    let th = theta
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let od = omega_dot
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pw = power
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let km = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let in_ = inertia
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dm = damping
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (new_th, new_od) = inertial::inertial_step(th, od, pw, km, in_, dm, n, dt);
    Ok((
        PyArray1::from_vec(py, new_th),
        PyArray1::from_vec(py, new_od),
    ))
}

/// Swing-equation state and trajectory calls with legacy Python argument names.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(theta, omega_dot, power, knm_flat, inertia, damping, n, dt, n_steps)")]
pub(crate) fn inertial_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<ArrayQuartet<'py>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "theta",
            "omega_dot",
            "power",
            "knm_flat",
            "inertia",
            "damping",
            "n",
            "dt",
            "n_steps",
        ],
        "inertial_run_rust",
    )?;
    let py = args.py();
    let theta: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omega_dot: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let power: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(3)?;
    let inertia: PyReadonlyArray1<'py, f64> = call.extract(4)?;
    let damping: PyReadonlyArray1<'py, f64> = call.extract(5)?;
    let n: usize = call.extract(6)?;
    let dt: f64 = call.extract(7)?;
    let n_steps: usize = call.extract(8)?;
    let th = theta
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let od = omega_dot
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pw = power
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let km = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let in_ = inertia
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dm = damping
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (f_th, f_od, t_th, t_od) = inertial::inertial_run(th, od, pw, km, in_, dm, n, dt, n_steps);
    Ok((
        PyArray1::from_vec(py, f_th),
        PyArray1::from_vec(py, f_od),
        PyArray1::from_vec(py, t_th),
        PyArray1::from_vec(py, t_od),
    ))
}
