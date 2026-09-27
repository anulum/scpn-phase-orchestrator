// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Entropy production Python boundary

//! Finite real measurement and scalar contracts for entropy production.

use crate::measurement_types::PlainReal;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::entropy_prod;

/// Compute dissipation from finite typed float64 buffers and plain real controls.
///
/// Frequencies have the phase cardinality and coupling has exactly n*n entries.
/// The timestep is nonnegative; empty systems and zero timesteps return zero.
#[pyfunction]
pub(crate) fn entropy_production_rate(
    phases: PyReadonlyArray1<'_, f64>,
    omegas: PyReadonlyArray1<'_, f64>,
    knm: PyReadonlyArray1<'_, f64>,
    alpha: PlainReal,
    dt: PlainReal,
) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let expected = p
        .len()
        .checked_mul(p.len())
        .ok_or_else(|| PyValueError::new_err("n*n overflows usize"))?;
    if o.len() != p.len() || k.len() != expected {
        return Err(PyValueError::new_err(
            "omegas or knm cardinality does not match phases",
        ));
    }
    if p.iter()
        .chain(o.iter())
        .chain(k.iter())
        .any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "phases, omegas and knm must contain only finite values",
        ));
    }
    if !alpha.0.is_finite() || !dt.0.is_finite() || dt.0 < 0.0 {
        return Err(PyValueError::new_err(
            "alpha must be finite and dt finite and nonnegative",
        ));
    }
    let rate = entropy_prod::entropy_production_rate(p, o, k, alpha.0, dt.0);
    if !rate.is_finite() || rate < 0.0 {
        return Err(PyValueError::new_err(
            "entropy production must be finite and nonnegative",
        ));
    }
    Ok(rate)
}
