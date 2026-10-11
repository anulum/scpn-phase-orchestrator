// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hodge Python boundary

//! Checked cardinalities and simplices for the actual Python-callable Hodge core.

use crate::measurement_types::PlainUsize;
use crate::return_types::ArrayTriple;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::hodge;

/// Decompose finite contiguous float64 measurements on a valid simplicial complex.
///
/// Counts retain their original non-boolean integer types. Edges and triangles
/// are contiguous int64 arrays of unique ascending pairs and triples; every
/// triangle is supported by its three supplied edges. Shapes and counts are
/// checked even when `n` is zero, before the numerical kernel runs.
///
/// # Errors
///
/// Refuses noncontiguous arrays, inconsistent cardinalities, nonfinite values,
/// invalid simplices and nonfinite computed components.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hodge_decomposition_rust<'py>(
    py: Python<'py>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    phases: PyReadonlyArray1<'py, f64>,
    n: PlainUsize,
    edges_flat: PyReadonlyArray1<'py, i64>,
    n_edges: PlainUsize,
    tris_flat: PyReadonlyArray1<'py, i64>,
    n_tris: PlainUsize,
) -> PyResult<ArrayTriple<'py>> {
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let edges = edges_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tris = tris_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (g, c, h) = hodge::try_hodge_decomposition(k, p, n.0, edges, n_edges.0, tris, n_tris.0)
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, g),
        PyArray1::from_vec(py, c),
        PyArray1::from_vec(py, h),
    ))
}
