// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Python FFI bindings

//! Checked Python boundary for fractal-dimension measurements.

use crate::measurement_types::{PlainU64, PlainUsize};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::dimension;

/// Count strict-radius pairs from contiguous finite measurements and plain integer controls.
///
/// # Errors
/// Refuses noncontiguous arrays, invalid counts, shape overflow or nonfinite measurements.
#[pyfunction]
#[pyo3(signature = (trajectory, t, d, epsilons, max_pairs = PlainUsize(50000), seed = PlainU64(42)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn correlation_integral_rust<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray1<'py, f64>,
    t: PlainUsize,
    d: PlainUsize,
    epsilons: PyReadonlyArray1<'py, f64>,
    max_pairs: PlainUsize,
    seed: PlainU64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let traj = trajectory
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let eps = epsilons
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = dimension::correlation_integral(traj, t.0, d.0, eps, max_pairs.0, seed.0)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

/// Compute a dimension from a contiguous finite spectrum without modifying caller storage.
///
/// # Errors
/// Refuses noncontiguous or nonfinite measurements.
#[pyfunction]
pub(crate) fn kaplan_yorke_dimension_rust(exponents: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let le = exponents
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if le.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "exponents must contain only finite values",
        ));
    }
    Ok(dimension::kaplan_yorke_dimension(le))
}
