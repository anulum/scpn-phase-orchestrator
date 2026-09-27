// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase-lag Python boundary

//! Original-source validation for the phase-lag Python interface.

use pyo3::prelude::*;
use spo_engine::lags::LagModel;

use crate::measurement_types::{real_values, PlainReal, PlainUsize};
use crate::spo_err;

/// Phase-lag matrix constructed from validated physical distances.
#[pyclass(name = "PyLagModel")]
pub(crate) struct PyLagModel {
    inner: LagModel,
}

#[pymethods]
impl PyLagModel {
    /// Estimate a row-major antisymmetric phase-lag matrix.
    ///
    /// Reject boolean, textual and temporal source aliases before conversion;
    /// the engine validates distance shape, finiteness and physical symmetry.
    #[staticmethod]
    fn estimate(distances: &Bound<'_, PyAny>, n: PlainUsize, speed: PlainReal) -> PyResult<Self> {
        let distances = real_values(distances, "distances")?;
        let inner = LagModel::estimate_from_distances(&distances, n.0, speed.0).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Construct a zero-lag matrix from a genuine unsigned integer count.
    #[staticmethod]
    fn zeros(n: PlainUsize) -> Self {
        Self {
            inner: LagModel::zeros(n.0),
        }
    }

    /// Return an owned row-major copy of the phase-lag values.
    #[getter]
    fn alpha(&self) -> Vec<f64> {
        self.inner.alpha.clone()
    }

    /// Return the oscillator count associated with the matrix.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
}
