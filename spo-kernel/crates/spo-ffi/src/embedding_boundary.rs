// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Embedding Python boundary

//! Original integer controls and finite typed samples for attractor embedding.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::embedding;

use crate::measurement_types::{PlainReal, PlainUsize};

/// Reject nonfinite samples before ordering, histogramming or distance computation.
fn finite_samples(values: &[f64]) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "signal must contain only finite values",
        ));
    }
    Ok(())
}

/// Embed finite float64 samples with plain positive integer parameters.
#[pyfunction]
pub(crate) fn delay_embed_rust<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    delay: PlainUsize,
    dimension: PlainUsize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    finite_samples(values)?;
    if delay.0 == 0 || dimension.0 == 0 {
        return Err(PyValueError::new_err(
            "delay and dimension must be positive",
        ));
    }
    let needed = (dimension.0 - 1)
        .checked_mul(delay.0)
        .ok_or_else(|| PyValueError::new_err("embedding window overflows usize"))?;
    let rows = values
        .len()
        .checked_sub(needed)
        .filter(|rows| *rows > 0)
        .ok_or_else(|| PyValueError::new_err("signal is too short for delay embedding"))?;
    rows.checked_mul(dimension.0)
        .ok_or_else(|| PyValueError::new_err("embedding output size overflows usize"))?;
    let result =
        embedding::delay_embed(values, delay.0, dimension.0).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

/// Select a delay with plain positive lag and histogram controls.
#[pyfunction]
#[pyo3(signature = (signal, max_lag = PlainUsize(100), n_bins = PlainUsize(32)))]
pub(crate) fn optimal_delay_rust(
    signal: PyReadonlyArray1<'_, f64>,
    max_lag: PlainUsize,
    n_bins: PlainUsize,
) -> PyResult<usize> {
    let values = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    finite_samples(values)?;
    if max_lag.0 == 0 || n_bins.0 < 2 {
        return Err(PyValueError::new_err(
            "max_lag must be positive and n_bins >= 2",
        ));
    }
    n_bins
        .0
        .checked_mul(n_bins.0)
        .ok_or_else(|| PyValueError::new_err("histogram size overflows usize"))?;
    Ok(embedding::optimal_delay(values, max_lag.0, n_bins.0))
}

/// Select embedding dimension with plain metadata and finite nonnegative tolerances.
#[pyfunction]
#[pyo3(signature = (signal, delay, max_dim = PlainUsize(10), rtol = PlainReal(15.0), atol = PlainReal(2.0)))]
pub(crate) fn optimal_dimension_rust(
    signal: PyReadonlyArray1<'_, f64>,
    delay: PlainUsize,
    max_dim: PlainUsize,
    rtol: PlainReal,
    atol: PlainReal,
) -> PyResult<usize> {
    let values = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    finite_samples(values)?;
    if delay.0 == 0 || max_dim.0 == 0 {
        return Err(PyValueError::new_err("delay and max_dim must be positive"));
    }
    if !rtol.0.is_finite() || !atol.0.is_finite() || rtol.0 < 0.0 || atol.0 < 0.0 {
        return Err(PyValueError::new_err(
            "rtol and atol must be finite and nonnegative",
        ));
    }
    if delay.0 >= values.len() {
        return Ok(1);
    }
    Ok(embedding::optimal_dimension(
        values, delay.0, max_dim.0, rtol.0, atol.0,
    ))
}
