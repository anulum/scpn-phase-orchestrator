// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Python named call arguments

//! Bind positional and keyword Python arguments before numerical extraction.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Ordered arguments of a fully bound numerical call.
pub(crate) struct CallArguments<'py> {
    values: Vec<Bound<'py, PyAny>>,
}

impl<'py> CallArguments<'py> {
    /// Preserve required positional-or-keyword parameters and reject ambiguous calls.
    pub(crate) fn bind(
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
        names: &[&str],
        function: &str,
    ) -> PyResult<Self> {
        if args.len() > names.len() {
            return Err(PyTypeError::new_err(format!(
                "{function}() takes {} positional arguments but {} were given",
                names.len(),
                args.len()
            )));
        }
        let mut values: Vec<Option<Bound<'py, PyAny>>> = vec![None; names.len()];
        for (index, value) in args.iter().enumerate() {
            values[index] = Some(value);
        }
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let name: String = key.extract()?;
                let index = names
                    .iter()
                    .position(|&candidate| candidate == name)
                    .ok_or_else(|| {
                        PyTypeError::new_err(format!(
                            "{function}() got an unexpected keyword argument '{name}'"
                        ))
                    })?;
                if values[index].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "{function}() got multiple values for argument '{name}'"
                    )));
                }
                values[index] = Some(value);
            }
        }
        let values = values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    PyTypeError::new_err(format!(
                        "{function}() missing required argument '{}'",
                        names[index]
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { values })
    }

    /// Extract a named parameter through its original PyO3 type contract.
    pub(crate) fn extract<T>(&self, index: usize) -> PyResult<T>
    where
        T: FromPyObjectOwned<'py>,
    {
        self.values[index].extract().map_err(Into::into)
    }
}
