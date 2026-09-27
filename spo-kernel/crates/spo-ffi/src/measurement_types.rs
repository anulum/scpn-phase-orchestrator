// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase quality Python boundary

//! Source-type validation shared by Python-callable numeric Rust boundaries.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

/// Extract a real scalar without admitting boolean or temporal aliases.
fn real_scalar(
    value: &Bound<'_, PyAny>,
    real_type: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only plain real numbers"
        )));
    }
    // Ordinary Python scalars need no attribute probing or ABC dispatch.
    if value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>() {
        return value.extract::<f64>().map_err(|err| {
            PyValueError::new_err(format!("{name} must contain real numbers: {err}"))
        });
    }
    let temporal = value
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract::<String>())
        .is_ok_and(|kind| kind == "m" || kind == "M");
    if temporal || !value.is_instance(real_type)? {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only plain real numbers"
        )));
    }
    value
        .extract::<f64>()
        .map_err(|err| PyValueError::new_err(format!("{name} must contain real numbers: {err}")))
}

/// Convert a measurement sequence after validating each original element.
pub(crate) fn real_values(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let real_type = value.py().import("numbers")?.getattr("Real")?;
    value
        .try_iter()?
        .map(|item| real_scalar(&item?, &real_type, name))
        .collect()
}

/// Constructor argument retaining source checks and numeric default values.
pub(crate) struct PlainReal(pub(crate) f64);

impl<'a, 'py> FromPyObject<'a, 'py> for PlainReal {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let real_type = value.py().import("numbers")?.getattr("Real")?;
        real_scalar(&value, &real_type, "measurement value").map(Self)
    }
}

/// Unsigned metadata count that preserves alias checks before integer coercion.
pub(crate) struct PlainUsize(pub(crate) usize);

/// Require a genuine integer before extracting count or form-code metadata.
fn integer_source(value: &Bound<'_, PyAny>) -> PyResult<()> {
    let integral = value.py().import("numbers")?.getattr("Integral")?;
    let temporal = value
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract::<String>())
        .is_ok_and(|kind| kind == "m" || kind == "M");
    if value.is_instance_of::<PyBool>() || temporal || !value.is_instance(&integral)? {
        return Err(PyValueError::new_err(
            "metadata must be a plain non-boolean integer",
        ));
    }
    Ok(())
}

impl<'a, 'py> FromPyObject<'a, 'py> for PlainUsize {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        integer_source(&value)?;
        value.extract::<usize>().map(Self)
    }
}

/// Signed form code with the same original-type checks as metadata counts.
pub(crate) struct PlainI32(pub(crate) i32);

impl<'a, 'py> FromPyObject<'a, 'py> for PlainI32 {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        integer_source(&value)?;
        value.extract::<i32>().map(Self)
    }
}

/// Signed block-window metadata with original-type checks before coercion.
pub(crate) struct PlainI64(pub(crate) i64);

impl<'a, 'py> FromPyObject<'a, 'py> for PlainI64 {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        integer_source(&value)?;
        value.extract::<i64>().map(Self)
    }
}

/// Unsigned random seed preserving the full u64 domain and original type.
pub(crate) struct PlainU64(pub(crate) u64);

impl<'a, 'py> FromPyObject<'a, 'py> for PlainU64 {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        integer_source(&value)?;
        value.extract::<u64>().map(Self)
    }
}
