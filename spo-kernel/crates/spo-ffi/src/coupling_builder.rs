// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Coupling builder Python boundary

//! Original measurement types at the coupling-construction Python boundary.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spo_engine::coupling::{project_knm, CouplingBuilder};
use spo_types::CouplingConfig;

use crate::measurement_types::{real_values, PlainReal, PlainUsize};
use crate::spo_err;

/// Construct and project typed coupling matrices for Python consumers.
#[pyclass(name = "PyCouplingBuilder")]
pub(crate) struct PyCouplingBuilder;

#[pymethods]
impl PyCouplingBuilder {
    /// Construct a stateless coupling builder.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Build a coupling matrix after validating original scalar metadata types.
    fn build<'py>(
        &self,
        py: Python<'py>,
        n: PlainUsize,
        base_strength: PlainReal,
        decay_alpha: PlainReal,
    ) -> PyResult<Bound<'py, PyDict>> {
        n.0.checked_mul(n.0).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("n*n overflows usize for coupling")
        })?;
        let config = CouplingConfig {
            base_strength: base_strength.0,
            decay_alpha: decay_alpha.0,
        };
        let cs = CouplingBuilder::build(n.0, &config).map_err(spo_err)?;
        let dict = PyDict::new(py);
        dict.set_item("knm", cs.knm)?;
        dict.set_item("alpha", cs.alpha)?;
        dict.set_item("n", cs.n)?;
        Ok(dict)
    }

    /// Project original real values to symmetric non-negative zero-diagonal coupling.
    /// Finite extreme means and subnormals are preserved; an empty matrix is valid.
    /// Invalid types, non-finite values and wrong cardinality are refused before projection.
    #[staticmethod]
    fn project(knm: &Bound<'_, PyAny>, n: PlainUsize) -> PyResult<Vec<f64>> {
        let mut knm = real_values(knm, "knm")?;
        project_knm(&mut knm, n.0).map_err(spo_err)?;
        Ok(knm)
    }
}
