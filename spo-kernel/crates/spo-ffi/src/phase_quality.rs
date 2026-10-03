// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase quality Python boundary

//! Preserve measurement source types before passing values to the Rust scorer.

use crate::measurement_types::{real_values, PlainReal};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spo_oscillators::quality::PhaseQualityScorer;

/// Native quality scorer with the same measurement contract as the public API.
#[pyclass(name = "PyPhaseQualityScorer")]
pub(crate) struct PyPhaseQualityScorer {
    inner: PhaseQualityScorer,
}

#[pymethods]
impl PyPhaseQualityScorer {
    /// Construct a scorer with finite unit-interval plain real thresholds.
    #[new]
    #[pyo3(signature = (collapse_threshold = PlainReal(0.1), min_quality = PlainReal(0.3)), text_signature = "(collapse_threshold=0.1, min_quality=0.3)")]
    fn new(collapse_threshold: PlainReal, min_quality: PlainReal) -> PyResult<Self> {
        for (name, value) in [
            ("collapse_threshold", collapse_threshold.0),
            ("min_quality", min_quality.0),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        Ok(Self {
            inner: PhaseQualityScorer {
                collapse_threshold: collapse_threshold.0,
                min_quality: min_quality.0,
            },
        })
    }

    /// Score original real measurements with overflow-safe amplitude weighting.
    ///
    /// # Arguments
    ///
    /// * `qualities` - Real measurement sequence; usable qualities are clamped.
    /// * `amplitudes` - Real amplitude sequence; weights have a 1e-12 floor.
    ///
    /// # Returns
    ///
    /// The matching-prefix mean, skipping nonfinite pairs and returning zero
    /// when none remain. Finite weights are scaled before Rust accumulation.
    ///
    /// # Errors
    ///
    /// Returns a Python error for noniterable input or a non-real source alias.
    fn score(&self, qualities: &Bound<'_, PyAny>, amplitudes: &Bound<'_, PyAny>) -> PyResult<f64> {
        let qualities = real_values(qualities, "quality")?;
        let amplitudes = real_values(amplitudes, "amplitude")?;
        Ok(self.inner.score(&qualities, &amplitudes))
    }

    /// Count nonfinite real qualities as collapsed after source validation.
    fn is_collapsed(&self, qualities: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_collapsed(&real_values(qualities, "quality")?))
    }

    /// Zero nonfinite or low real qualities after source validation.
    fn downweight_mask(&self, qualities: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        Ok(self
            .inner
            .downweight_mask(&real_values(qualities, "quality")?))
    }
}
