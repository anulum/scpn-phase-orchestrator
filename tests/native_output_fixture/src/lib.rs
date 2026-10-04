// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Native output fault boundary fixture

//! Exercise the installed-extension boundary with deliberately invalid outputs.
//! This module implements no numerical solver and is never a runtime backend.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const MARKER: &str = "SPO_NATIVE_OUTPUT_FAULT_FIXTURE_V1";
type EIMetrics = (f64, f64, f64, bool, f64, f64, f64, f64);
static FAULT: AtomicU8 = AtomicU8::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

/// Select a named invalid output in this test extension only.
#[pyfunction]
fn fixture_select_fault(name: &str) -> PyResult<()> {
    let code = match name {
        "ei-ratio-nan" => 1,
        "ei-strength-inf" => 2,
        "ei-block-nan" => 3,
        "sheaf-dimension" => 4,
        "sheaf-cardinality" => 5,
        "sheaf-nan" => 6,
        "sheaf-inf" => 7,
        "sheaf-negative" => 8,
        "sheaf-upper-bound" => 9,
        "sparse-shape" => 10,
        "sparse-bool" => 11,
        "sparse-complex" => 12,
        "sparse-negative" => 13,
        "sparse-upper-bound" => 14,
        _ => return Err(PyValueError::new_err("unknown native output fault")),
    };
    FAULT.store(code, Ordering::SeqCst);
    Ok(())
}

/// Return the number of calls that actually crossed this extension boundary.
#[pyfunction]
fn fixture_call_count() -> usize {
    CALLS.load(Ordering::SeqCst)
}

/// Return invalid E/I metrics without implementing a numerical computation.
#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
fn compute_ei_balance_rust(
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<EIMetrics> {
    CALLS.fetch_add(1, Ordering::SeqCst);
    let mut result = (1.0, 1.0, 1.0, true, 1.0, 1.0, 1.0, 1.0);
    match FAULT.load(Ordering::SeqCst) {
        1 => result.0 = f64::NAN,
        2 => result.1 = f64::INFINITY,
        3 => result.7 = f64::NAN,
        _ => return Err(PyRuntimeError::new_err("E/I fault was not selected")),
    }
    Ok(result)
}

/// Refuse unsupported numerical adjustment instead of pretending to solve it.
#[pyfunction]
#[pyo3(signature = (*_args, **_kwargs))]
fn adjust_ei_ratio_rust(
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(
        "the fault fixture does not implement numerical adjustment",
    ))
}

/// Form a malformed sheaf return at the real native/Python boundary.
#[cfg(feature = "steppers")]
fn sheaf_output(py: Python<'_>, size: usize) -> PyResult<Bound<'_, PyAny>> {
    CALLS.fetch_add(1, Ordering::SeqCst);
    let numpy = py.import("numpy")?;
    let mut data = vec![0.0; size];
    match FAULT.load(Ordering::SeqCst) {
        4 => return numpy.call_method1("array", (vec![data],)),
        5 => {
            data.pop();
        }
        6 => data[0] = f64::NAN,
        7 => data[0] = f64::INFINITY,
        8 => data[0] = -1.0,
        9 => data[0] = std::f64::consts::TAU,
        _ => return Err(PyRuntimeError::new_err("sheaf fault was not selected")),
    }
    numpy.call_method1("array", (data,))
}

/// Form a malformed sparse return without implementing a numerical solver.
#[cfg(feature = "steppers")]
fn sparse_output(py: Python<'_>, size: usize) -> PyResult<Bound<'_, PyAny>> {
    CALLS.fetch_add(1, Ordering::SeqCst);
    let numpy = py.import("numpy")?;
    let mut data = vec![0.0; size];
    match FAULT.load(Ordering::SeqCst) {
        10 => {
            data.pop();
        }
        11 => return numpy.call_method1("array", (vec![false; size],)),
        12 => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "complex128")?;
            return numpy.call_method("array", (data,), Some(&kwargs));
        }
        13 => data[0] = -1.0,
        14 => data[0] = std::f64::consts::TAU,
        _ => return Err(PyRuntimeError::new_err("sparse fault was not selected")),
    }
    numpy.call_method1("array", (data,))
}

/// A defective stepper whose output is always refused by the public wrapper.
#[cfg(feature = "steppers")]
#[pyclass]
struct PySheafUPDEStepper {
    size: usize,
}

#[cfg(feature = "steppers")]
#[pymethods]
impl PySheafUPDEStepper {
    /// Retain geometry only; this fixture has no timestep or integration state.
    #[new]
    #[pyo3(signature = (n, d, dt, method, *, atol, rtol))]
    fn new(n: usize, d: usize, dt: f64, method: &str, atol: f64, rtol: f64) -> Self {
        let _ = (dt, method, atol, rtol);
        Self { size: n * d }
    }

    /// Emit the selected defective output through the real step call.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        sheaf_output(py, self.size)
    }

    /// Emit the same defective output through the real batch call.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        sheaf_output(py, self.size)
    }
}

/// A sparse boundary producer with no numerical solver implementation.
#[cfg(feature = "steppers")]
#[pyclass]
struct PySparseUPDEStepper {
    size: usize,
}

#[cfg(feature = "steppers")]
#[pymethods]
impl PySparseUPDEStepper {
    /// Retain geometry for invalid-output construction only.
    #[new]
    #[pyo3(signature = (n, dt, method, *, atol, rtol))]
    fn new(n: usize, dt: f64, method: &str, atol: f64, rtol: f64) -> Self {
        let _ = (dt, method, atol, rtol);
        Self { size: n }
    }

    /// Emit the selected defective output through the real step call.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        sparse_output(py, self.size)
    }

    /// Emit the same defective output through the real batch call.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        sparse_output(py, self.size)
    }
}

/// Mark the fixture explicitly and omit stepper classes in the second build.
#[pymodule]
fn spo_kernel(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__SPO_FAULT_FIXTURE__", MARKER)?;
    module.add("__version__", "0.0.0+fault-fixture")?;
    module.add(
        "__SPO_FIXTURE_VARIANT__",
        if cfg!(feature = "steppers") {
            "invalid-outputs"
        } else {
            "missing-classes"
        },
    )?;
    module.add_function(wrap_pyfunction!(fixture_select_fault, module)?)?;
    module.add_function(wrap_pyfunction!(fixture_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(compute_ei_balance_rust, module)?)?;
    module.add_function(wrap_pyfunction!(adjust_ei_ratio_rust, module)?)?;
    #[cfg(feature = "steppers")]
    {
        module.add_class::<PySheafUPDEStepper>()?;
        module.add_class::<PySparseUPDEStepper>()?;
    }
    Ok(())
}
