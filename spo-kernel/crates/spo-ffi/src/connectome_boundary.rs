// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Synthetic connectome FFI boundary

//! Original synthetic generator with integer admission and checked allocation.

use crate::measurement_types::{PlainU64, PlainUsize};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::connectome::try_load_hcp_connectome;

/// Return flat row-major synthetic weights or refuse invalid dense storage.
#[pyfunction]
pub(crate) fn load_hcp_connectome_rust(
    py: Python<'_>,
    n_regions: PlainUsize,
    seed: PlainU64,
) -> PyResult<Py<PyArray1<f64>>> {
    if n_regions.0 < 2 {
        return Err(PyValueError::new_err("n_regions must be >= 2"));
    }
    let result = try_load_hcp_connectome(n_regions.0, seed.0).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result).into())
}
