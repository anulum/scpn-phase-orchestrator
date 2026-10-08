// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Ethical-cost Python boundary

//! Typed ethical-cost arrays, original scalar types and dimension metadata.

use crate::measurement_types::{PlainReal, PlainUsize};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::ethical;

/// Compute the diagnostic from contiguous float64 arrays and plain real parameters.
///
/// Shapes, finite values and representable derived arithmetic are checked by
/// the public Rust core. Dimension metadata must be a non-boolean integer.
/// Boolean, text, complex and temporal parameter aliases are refused before
/// coercion; finite signed parameter values remain admissible.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ethical_cost_rust(
    phases: PyReadonlyArray1<f64>,
    knm: PyReadonlyArray1<f64>,
    n: PlainUsize,
    alpha_r: PlainReal,
    beta_k: PlainReal,
    gamma_q: PlainReal,
    nu_s: PlainReal,
    kappa: PlainReal,
    r_min: PlainReal,
    connectivity_min: PlainReal,
    max_coupling: PlainReal,
) -> PyResult<(f64, f64, f64, usize)> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    ethical::compute_ethical_cost(
        p,
        k,
        n.0,
        alpha_r.0,
        beta_k.0,
        gamma_q.0,
        nu_s.0,
        kappa.0,
        r_min.0,
        connectivity_min.0,
        max_coupling.0,
    )
    .map_err(PyValueError::new_err)
}
