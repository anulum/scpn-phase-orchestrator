// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ordinal entropy Python boundary

//! Typed measurement and original integer controls for ordinal entropy.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::measurement_types::PlainUsize;

/// Validate the supported embedding parameters and finite real samples.
fn validate(series: &[f64], dimension: usize, delay: usize) -> PyResult<()> {
    if !(2..=7).contains(&dimension) {
        return Err(PyValueError::new_err("dimension must lie in [2, 7]"));
    }
    if delay == 0 {
        return Err(PyValueError::new_err("delay must be positive"));
    }
    if series.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "series must contain only finite values",
        ));
    }
    Ok(())
}

/// Return stable Lehmer codes from a typed float64 series and plain integer controls.
#[pyfunction]
#[pyo3(signature = (series, dimension = PlainUsize(3), delay = PlainUsize(1)))]
pub(crate) fn ordinal_pattern_sequence<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'_, f64>,
    dimension: PlainUsize,
    delay: PlainUsize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let values = series
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    validate(values, dimension.0, delay.0)?;
    let codes = if delay.0 > values.len() / (dimension.0 - 1) {
        Vec::new()
    } else {
        spo_engine::opt_entropy::ordinal_pattern_sequence(values, dimension.0, delay.0)
    };
    Ok(PyArray1::from_vec(py, codes))
}

/// Return normalised transition entropy with original integer parameter validation.
#[pyfunction]
#[pyo3(signature = (series, dimension = PlainUsize(3), delay = PlainUsize(1)))]
pub(crate) fn transition_entropy(
    series: PyReadonlyArray1<'_, f64>,
    dimension: PlainUsize,
    delay: PlainUsize,
) -> PyResult<f64> {
    let values = series
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    validate(values, dimension.0, delay.0)?;
    if delay.0 > values.len() / (dimension.0 - 1) {
        return Ok(0.0);
    }
    Ok(spo_engine::opt_entropy::transition_entropy(
        values,
        dimension.0,
        delay.0,
    ))
}
