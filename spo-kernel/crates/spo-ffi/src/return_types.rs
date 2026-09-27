// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Python array return types

//! Tuple layouts shared by the Python numerical binding return contracts.

use numpy::PyArray1;
use pyo3::prelude::*;

/// Two contiguous float64 arrays; element meanings are specified by each API.
pub(crate) type ArrayPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
/// Three float64 arrays, preserving the public tuple ordering.
pub(crate) type ArrayTriple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);
/// Final paired state and its paired trajectories, in the public tuple ordering.
pub(crate) type ArrayQuartet<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);
/// Two float64 arrays followed by a scalar diagnostic.
pub(crate) type ArraysWithScalar<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64);
/// Two float64 arrays followed by their reported cardinality.
pub(crate) type ArraysWithCount<'py> =
    (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize);
/// Recurrence rate, determinism, diagonal mean/max/entropy, laminarity, trapping time and vertical max.
pub(crate) type RqaMetrics = (f64, f64, f64, usize, f64, f64, f64, usize);
/// E/I ratio, excitatory/inhibitory totals, balance flag and four block means.
pub(crate) type EiBalanceMetrics = (f64, f64, f64, bool, f64, f64, f64, f64);
/// Phase, amplitude and frequency arrays followed by dominant frequency.
pub(crate) type PhaseExtractionOutput =
    (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64);
