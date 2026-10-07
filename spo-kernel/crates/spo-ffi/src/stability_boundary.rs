// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Stability boundary

//! Seeded basin sampling and synchronization-transition review measurements.

use crate::call_arguments::CallArguments;
use crate::return_types::ArraysWithScalar;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};
use spo_engine::{basin_stability, bifurcation};

/// Refuse Python and NumPy boolean aliases before scalar extraction.
fn reject_boolean_scalars(call: &CallArguments<'_>, indices: &[usize]) -> PyResult<()> {
    for &index in indices {
        let value: Bound<'_, PyAny> = call.extract(index)?;
        let type_name = value.get_type().name()?;
        let type_name = type_name.to_str()?;
        if value.is_instance_of::<PyBool>() || type_name == "bool" || type_name == "bool_" {
            return Err(PyTypeError::new_err("numerical scalar must not be boolean"));
        }
    }
    Ok(())
}

/// Seeded basin sampling and synchronization-transition review measurements.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(omegas, knm_flat, alpha_flat, n, dt, n_transient, n_measure, n_samples, r_threshold, seed)")]
pub(crate) fn basin_stability_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, usize)> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "omegas",
            "knm_flat",
            "alpha_flat",
            "n",
            "dt",
            "n_transient",
            "n_measure",
            "n_samples",
            "r_threshold",
            "seed",
        ],
        "basin_stability_rust",
    )?;
    reject_boolean_scalars(&call, &[3, 4, 5, 6, 7, 8, 9])?;
    let py = args.py();
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let alpha_flat: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let n: usize = call.extract(3)?;
    let dt: f64 = call.extract(4)?;
    let n_transient: usize = call.extract(5)?;
    let n_measure: usize = call.extract(6)?;
    let n_samples: usize = call.extract(7)?;
    let r_threshold: f64 = call.extract(8)?;
    let seed: u64 = call.extract(9)?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = basin_stability::try_basin_stability(
        o,
        k,
        a,
        n,
        dt,
        n_transient,
        n_measure,
        n_samples,
        r_threshold,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        result.s_b,
        PyArray1::from_vec(py, result.r_finals),
        result.n_converged,
    ))
}

/// Seeded basin sampling and synchronization-transition review measurements.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(phases_init, omegas, knm_flat, alpha_flat, n, k_scale, dt, n_transient, n_measure)")]
pub(crate) fn steady_state_r_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<f64> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "phases_init",
            "omegas",
            "knm_flat",
            "alpha_flat",
            "n",
            "k_scale",
            "dt",
            "n_transient",
            "n_measure",
        ],
        "steady_state_r_rust",
    )?;
    reject_boolean_scalars(&call, &[4, 5, 6, 7, 8])?;
    let phases_init: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let alpha_flat: PyReadonlyArray1<'py, f64> = call.extract(3)?;
    let n: usize = call.extract(4)?;
    let k_scale: f64 = call.extract(5)?;
    let dt: f64 = call.extract(6)?;
    let n_transient: usize = call.extract(7)?;
    let n_measure: usize = call.extract(8)?;
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
    bifurcation::try_steady_state_r(p, o, k, a, n, k_scale, dt, n_transient, n_measure)
        .map_err(PyValueError::new_err)
}

/// Seeded basin sampling and synchronization-transition review measurements.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(omegas, knm_flat, alpha_flat, n, phases_init, k_min, k_max, n_points, dt, n_transient, n_measure)")]
pub(crate) fn trace_sync_transition_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<ArraysWithScalar<'py>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "omegas",
            "knm_flat",
            "alpha_flat",
            "n",
            "phases_init",
            "k_min",
            "k_max",
            "n_points",
            "dt",
            "n_transient",
            "n_measure",
        ],
        "trace_sync_transition_rust",
    )?;
    reject_boolean_scalars(&call, &[3, 5, 6, 7, 8, 9, 10])?;
    let py = args.py();
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let alpha_flat: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let n: usize = call.extract(3)?;
    let phases_init: PyReadonlyArray1<'py, f64> = call.extract(4)?;
    let k_min: f64 = call.extract(5)?;
    let k_max: f64 = call.extract(6)?;
    let n_points: usize = call.extract(7)?;
    let dt: f64 = call.extract(8)?;
    let n_transient: usize = call.extract(9)?;
    let n_measure: usize = call.extract(10)?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (kv, rv, kc) = bifurcation::try_trace_sync_transition(
        o,
        k,
        a,
        n,
        p,
        k_min,
        k_max,
        n_points,
        dt,
        n_transient,
        n_measure,
    )
    .map_err(PyValueError::new_err)?;
    Ok((PyArray1::from_vec(py, kv), PyArray1::from_vec(py, rv), kc))
}

/// Seeded basin sampling and synchronization-transition review measurements.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(omegas, knm_flat, alpha_flat, n, phases_init, dt, n_transient, n_measure, tol)")]
pub(crate) fn find_critical_coupling_bif_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<f64> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "omegas",
            "knm_flat",
            "alpha_flat",
            "n",
            "phases_init",
            "dt",
            "n_transient",
            "n_measure",
            "tol",
        ],
        "find_critical_coupling_bif_rust",
    )?;
    reject_boolean_scalars(&call, &[3, 5, 6, 7, 8])?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let knm_flat: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let alpha_flat: PyReadonlyArray1<'py, f64> = call.extract(2)?;
    let n: usize = call.extract(3)?;
    let phases_init: PyReadonlyArray1<'py, f64> = call.extract(4)?;
    let dt: f64 = call.extract(5)?;
    let n_transient: usize = call.extract(6)?;
    let n_measure: usize = call.extract(7)?;
    let tol: f64 = call.extract(8)?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    bifurcation::try_find_critical_coupling(o, k, a, n, p, dt, n_transient, n_measure, tol)
        .map_err(PyValueError::new_err)
}
