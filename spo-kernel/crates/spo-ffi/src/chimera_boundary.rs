// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Chimera Python boundary

//! Finite phase/coupling arrays and original integer counts for chimera detection.

use crate::measurement_types::PlainUsize;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::chimera;

type ChimeraOutput<'py> = (Vec<usize>, Vec<usize>, f64, Bound<'py, PyArray1<f64>>);

/// Detect chimera states from finite typed float64 arrays with a zero diagonal.
///
/// Returns coherent indices, incoherent indices, boundary fraction and local order.
/// Counts must be plain nonnegative integers; flattened coupling has exactly n*n entries.
#[pyfunction]
pub(crate) fn detect_chimera_rust<'py>(
    py: Python<'py>,
    phases: PyReadonlyArray1<'py, f64>,
    knm: PyReadonlyArray1<'py, f64>,
    n: PlainUsize,
) -> PyResult<ChimeraOutput<'py>> {
    let n = n.0;
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("n*n overflows usize"))?;
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if p.len() != n || k.len() != expected {
        return Err(PyValueError::new_err(
            "phases or knm cardinality does not match n",
        ));
    }
    if p.iter().chain(k.iter()).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "phases and knm must contain only finite values",
        ));
    }
    if (0..n).any(|i| k[i * n + i].abs() > 1e-15) {
        return Err(PyValueError::new_err(
            "knm self-coupling diagonal must be zero",
        ));
    }
    let result = chimera::detect_chimera(p, k, n);
    Ok((
        result.coherent_indices,
        result.incoherent_indices,
        result.chimera_index,
        PyArray1::from_vec(py, result.local_order),
    ))
}
