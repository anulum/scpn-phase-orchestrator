// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Transfer-entropy coupling boundary

//! Source metadata and physical matrix checks for adaptive coupling.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::measurement_types::{PlainReal, PlainUsize};

/// Validate finite non-negative matrix values and negligible self-coupling.
fn validate_matrix(values: &[f64], n: usize, name: &str, tolerance: f64) -> PyResult<()> {
    if values.iter().any(|v| !v.is_finite() || *v < -tolerance) {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and non-negative"
        )));
    }
    if (0..n).any(|i| values[i * n + i].abs() > 1e-12) {
        return Err(PyValueError::new_err(format!(
            "{name} diagonal must be zero"
        )));
    }
    Ok(())
}

/// Update a typed coupling matrix with validated transfer-entropy scores.
///
/// Reject metadata aliases, invalid cardinality, non-finite matrices and
/// controls outside the public learning-rate and decay domains before update.
#[pyfunction]
pub(crate) fn te_adapt_coupling_rust(
    py: Python<'_>,
    knm: PyReadonlyArray1<'_, f64>,
    te: PyReadonlyArray1<'_, f64>,
    n: PlainUsize,
    lr: PlainReal,
    decay: PlainReal,
) -> PyResult<Py<PyArray1<f64>>> {
    let n = n.0;
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("n*n overflows usize"))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = te
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if k.len() != expected || t.len() != expected {
        return Err(PyValueError::new_err("knm and te lengths must match n*n"));
    }
    if !lr.0.is_finite() || lr.0 < 0.0 {
        return Err(PyValueError::new_err("lr must be finite and non-negative"));
    }
    if !decay.0.is_finite() || !(0.0..=1.0).contains(&decay.0) {
        return Err(PyValueError::new_err("decay must be finite and in [0, 1]"));
    }
    validate_matrix(k, n, "knm", 0.0)?;
    validate_matrix(t, n, "transfer entropy", 1e-12)?;
    let scores: Vec<f64> = t.iter().map(|v| v.max(0.0)).collect();
    let result = spo_engine::te_adaptive::te_adapt_coupling(k, &scores, n, lr.0, decay.0);
    if result.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("adapted coupling must be finite"));
    }
    Ok(PyArray1::from_vec(py, result).into())
}
