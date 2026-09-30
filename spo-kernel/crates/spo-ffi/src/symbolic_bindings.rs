// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Symbolic extraction bindings

//! Python symbolic phase and transition-quality entry points.

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use spo_oscillators::symbolic;

enum StateValues {
    Signed(Vec<i64>),
    Unsigned(Vec<u64>),
}

/// Copy aligned 64-bit labels without reinterpreting signedness, in logical order.
fn state_values(state_indices: &Bound<'_, PyAny>) -> PyResult<StateValues> {
    if let Ok(indices) = state_indices.extract::<PyReadonlyArray1<'_, i64>>() {
        if !indices.is_aligned() {
            return Err(PyValueError::new_err("state_indices must be aligned"));
        }
        return Ok(StateValues::Signed(
            indices.as_array().iter().copied().collect(),
        ));
    }
    if let Ok(indices) = state_indices.extract::<PyReadonlyArray1<'_, u64>>() {
        if !indices.is_aligned() {
            return Err(PyValueError::new_err("state_indices must be aligned"));
        }
        return Ok(StateValues::Unsigned(
            indices.as_array().iter().copied().collect(),
        ));
    }
    Err(PyTypeError::new_err(
        "state_indices must be a one-dimensional int64 or uint64 array",
    ))
}

/// Map an unsigned state index onto a ring with the requested vocabulary size.
#[pyfunction]
pub(crate) fn ring_phase(state_index: usize, n_states: usize) -> f64 {
    symbolic::ring_phase(state_index, n_states)
}

/// Map a one-dimensional int64 or uint64 view, preserving logical stride order.
#[pyfunction]
pub(crate) fn ring_phases_rust<'py>(
    py: Python<'py>,
    state_indices: &Bound<'_, PyAny>,
    n_states: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indices = state_values(state_indices)?;
    let phases = match indices {
        StateValues::Signed(values) => symbolic::ring_phases(&values, n_states),
        StateValues::Unsigned(values) => symbolic::ring_phases_unsigned(&values, n_states),
    };
    Ok(PyArray1::from_vec(py, phases))
}

/// Normalise an unsigned cumulative position by an unsigned total walk length.
#[pyfunction]
pub(crate) fn graph_walk_phase(position: usize, walk_length: usize) -> f64 {
    symbolic::graph_walk_phase(position, walk_length)
}

/// Extract graph phases from int64 or uint64 labels, including strided views.
#[pyfunction]
pub(crate) fn graph_walk_phases_rust<'py>(
    py: Python<'py>,
    state_indices: &Bound<'_, PyAny>,
    n_states: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indices = state_values(state_indices)?;
    let phases = match indices {
        StateValues::Signed(values) => symbolic::graph_walk_phases(&values, n_states),
        StateValues::Unsigned(values) => symbolic::graph_walk_phases_unsigned(&values, n_states),
    };
    Ok(PyArray1::from_vec(py, phases))
}

/// Score an unsigned linear transition distance relative to the vocabulary size.
#[pyfunction]
pub(crate) fn transition_quality(step_size: usize, n_states: usize) -> f64 {
    symbolic::transition_quality(step_size, n_states)
}

/// Score int64 or uint64 views without imposing a contiguous-buffer requirement.
#[pyfunction]
pub(crate) fn transition_qualities_rust<'py>(
    py: Python<'py>,
    state_indices: &Bound<'_, PyAny>,
    n_states: usize,
    initial_quality: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !initial_quality.is_finite() {
        return Err(PyValueError::new_err(
            "initial_quality must be a finite float",
        ));
    }
    let indices = state_values(state_indices)?;
    let qualities = match indices {
        StateValues::Signed(values) => {
            symbolic::transition_qualities(&values, n_states, initial_quality)
        }
        StateValues::Unsigned(values) => {
            symbolic::transition_qualities_unsigned(&values, n_states, initial_quality)
        }
    };
    Ok(PyArray1::from_vec(py, qualities))
}
