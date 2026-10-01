// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — E/I native boundary

//! Native E/I coupling admission and Python result conversion.
use crate::measurement_types::{PlainReal, PlainUsize};
use crate::return_types::EiBalanceMetrics;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::ei_balance;

/// Reject negative indices while ignoring non-negative out-of-range values.
fn indices(values: &[i64], n: usize, name: &str) -> PyResult<Vec<usize>> {
    let mut result = Vec::new();
    for &value in values {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{name} indices must be non-negative"
            )));
        }
        if let Ok(value) = usize::try_from(value) {
            if value < n {
                result.push(value);
            }
        }
    }
    Ok(result)
}

// ─── E/I Balance ──────────────────────────────────────────────────

/// Compute the validated E/I summary through the installed native boundary.
#[pyfunction]
pub(crate) fn compute_ei_balance_rust(
    knm_flat: PyReadonlyArray1<'_, f64>,
    n: PlainUsize,
    excitatory_indices: PyReadonlyArray1<'_, i64>,
    inhibitory_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<EiBalanceMetrics> {
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_raw = excitatory_indices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let i_raw = inhibitory_indices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_idx = indices(e_raw, n.0, "excitatory")?;
    let i_idx = indices(i_raw, n.0, "inhibitory")?;
    let r = ei_balance::compute_ei_balance(k, n.0, &e_idx, &i_idx).map_err(crate::spo_err)?;
    Ok((
        r.ratio,
        r.excitatory_strength,
        r.inhibitory_strength,
        r.is_balanced,
        r.e_to_e,
        r.e_to_i,
        r.i_to_e,
        r.i_to_i,
    ))
}

/// Return a finite independently owned inhibitory-row adjustment.
#[pyfunction]
#[pyo3(signature = (knm_flat, n, excitatory_indices, inhibitory_indices, target_ratio = PlainReal(1.0)))]
pub(crate) fn adjust_ei_ratio_rust<'py>(
    py: Python<'py>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    n: PlainUsize,
    excitatory_indices: PyReadonlyArray1<'py, i64>,
    inhibitory_indices: PyReadonlyArray1<'py, i64>,
    target_ratio: PlainReal,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = knm_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_raw = excitatory_indices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let i_raw = inhibitory_indices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_idx = indices(e_raw, n.0, "excitatory")?;
    let i_idx = indices(i_raw, n.0, "inhibitory")?;
    let result = ei_balance::adjust_ei_ratio(k, n.0, &e_idx, &i_idx, target_ratio.0)
        .map_err(crate::spo_err)?;
    Ok(PyArray1::from_vec(py, result))
}
