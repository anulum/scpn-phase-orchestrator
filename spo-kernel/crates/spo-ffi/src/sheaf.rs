// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Sheaf stepper NumPy boundary

//! Stateful cellular-sheaf integration for flat readonly NumPy buffers.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_engine::sheaf_upde::SheafUPDEStepper;
use spo_types::{IntegrationConfig, Method};

use crate::measurement_types::{PlainReal, PlainU64, PlainUsize};
use crate::spo_err;

// ─── PySheafUPDEStepper ───────────────────────────────────────────────────

#[pyclass(name = "PySheafUPDEStepper")]
pub(crate) struct PySheafUPDEStepper {
    inner: SheafUPDEStepper,
}

#[pymethods]
impl PySheafUPDEStepper {
    /// Configure geometry, solver method, outer timestep and RK45 tolerances.
    ///
    /// Counts must be non-boolean integers; numerical controls must be finite
    /// real numbers accepted by the solver configuration.
    ///
    /// Parameters
    /// ----------
    /// n, d : int
    ///     Positive oscillator count and dimension per phase vector.
    /// dt : float, optional
    ///     Positive outer interval advanced by each successful step.
    /// method : str, optional
    ///     "euler", "rk4" or adaptive Dormand-Prince "rk45".
    /// n_substeps : int, optional
    ///     Positive u32 count partitioning each outer interval.
    /// atol, rtol : float, optional
    ///     Positive finite RK45 tolerances, with rtol >= atol for RK45.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     The method is not a string.
    /// ValueError
    ///     Counts or scalars have boolean, textual, temporal, complex or other
    ///     non-real source types; geometry, method, controls or the divided
    ///     timestep are invalid; or n_substeps exceeds u32.
    /// OverflowError
    ///     An integer count is negative or exceeds the native usize range.
    #[new]
    #[pyo3(signature = (n, d, dt = PlainReal(0.01), method = "euler", n_substeps = PlainUsize(1), atol = PlainReal(1e-6), rtol = PlainReal(1e-3)))]
    #[pyo3(text_signature = "(n, d, dt=0.01, method='euler', n_substeps=1, atol=1e-6, rtol=1e-3)")]
    fn new(
        n: PlainUsize,
        d: PlainUsize,
        dt: PlainReal,
        method: &str,
        n_substeps: PlainUsize,
        atol: PlainReal,
        rtol: PlainReal,
    ) -> PyResult<Self> {
        let m = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            "rk45" => Method::RK45,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt: dt.0,
            method: m,
            n_substeps: u32::try_from(n_substeps.0)
                .map_err(|_| PyValueError::new_err("n_substeps exceeds u32"))?,
            atol: atol.0,
            rtol: rtol.0,
        };
        let inner = SheafUPDEStepper::new(n.0, d.0, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Advance one complete outer interval without mutating NumPy inputs.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     A buffer is not a one-dimensional float64 NumPy array.
    /// ValueError
    ///     Buffers are strided, have incorrect lengths or contain non-finite
    ///     values; zeta is not a finite real scalar; or integration refuses.
    ///     Inputs and last_dt remain unchanged on refusal.
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        restriction_maps: PyReadonlyArray1<'py, f64>,
        zeta: PlainReal,
        psi: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let p_w = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let r_m = restriction_maps
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let p_psi = psi
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .step(&mut p_out, p_w, r_m, zeta.0, p_psi)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    /// Advance validated batches, retaining inputs and last_dt on refusal.
    ///
    /// Zero steps still validate all buffers and return an independent copy.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     A buffer is not a one-dimensional float64 NumPy array.
    /// ValueError
    ///     Buffers are strided, have incorrect lengths or contain non-finite
    ///     values; zeta is not a finite real scalar; n_steps is not a plain
    ///     non-boolean integer; or integration refuses.
    /// OverflowError
    ///     n_steps is negative or exceeds the native u64 range.
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        restriction_maps: PyReadonlyArray1<'py, f64>,
        zeta: PlainReal,
        psi: PyReadonlyArray1<'py, f64>,
        n_steps: PlainU64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let p_w = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let r_m = restriction_maps
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let p_psi = psi
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .run(&mut p_out, p_w, r_m, zeta.0, p_psi, n_steps.0)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    /// Return the configured oscillator count.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Return the dimension of each phase vector.
    #[getter]
    fn d(&self) -> usize {
        self.inner.d()
    }

    /// Return the next adaptive substep proposal or fixed configured timestep.
    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
    }
}
