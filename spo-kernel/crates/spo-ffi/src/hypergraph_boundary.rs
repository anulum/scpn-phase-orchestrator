// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Hypergraph boundary

//! Stateful and trajectory Python calls for weighted hypergraph coupling.

use crate::call_arguments::CallArguments;
use crate::measurement_types::{PlainReal, PlainUsize};
use crate::spo_err;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spo_engine::hypergraph;
use spo_types::IntegrationConfig;

// ─── PyHypergraphStepper ──────────────────────────────────────────────────

/// Stateful integrator preserving the model's configured timestep and Python state contract.
#[pyclass(name = "PyHypergraphStepper")]
pub(crate) struct PyHypergraphStepper {
    inner: hypergraph::HypergraphStepper,
}

#[pymethods]
impl PyHypergraphStepper {
    #[new]
    #[pyo3(signature = (n, dt = 0.01))]
    fn new(n: usize, dt: f64) -> PyResult<Self> {
        let config = IntegrationConfig {
            dt,
            ..Default::default()
        };
        let inner = hypergraph::HypergraphStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Evolve validated hypergraph state from the original positional or keyword parameters.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, phases, omegas, edges, knm, alpha, zeta, psi)")]
    fn step<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let call = CallArguments::bind(
            args,
            kwargs,
            &["phases", "omegas", "edges", "knm", "alpha", "zeta", "psi"],
            "step",
        )?;
        let py = args.py();
        let phases: PyReadonlyArray1<'py, f64> = call.extract(0)?;
        let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
        let edges: Vec<(Vec<usize>, f64)> = call.extract(2)?;
        let knm: PyReadonlyArray1<'py, f64> = call.extract(3)?;
        let alpha: PyReadonlyArray1<'py, f64> = call.extract(4)?;
        let zeta = call.extract::<PlainReal>(5)?.0;
        let psi = call.extract::<PlainReal>(6)?.0;
        let mut p = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = knm
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let h_edges: Vec<hypergraph::Hyperedge> = edges
            .into_iter()
            .map(|(nodes, strength)| hypergraph::Hyperedge { nodes, strength })
            .collect();

        self.inner
            .step(&mut p, o, &h_edges, k, a, zeta, psi)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    /// Evolve validated hypergraph state from the original positional or keyword parameters.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, phases, omegas, edges, knm, alpha, zeta, psi, n_steps)")]
    fn run<'py>(
        &mut self,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let call = CallArguments::bind(
            args,
            kwargs,
            &[
                "phases", "omegas", "edges", "knm", "alpha", "zeta", "psi", "n_steps",
            ],
            "run",
        )?;
        let py = args.py();
        let phases: PyReadonlyArray1<'py, f64> = call.extract(0)?;
        let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
        let edges: Vec<(Vec<usize>, f64)> = call.extract(2)?;
        let knm: PyReadonlyArray1<'py, f64> = call.extract(3)?;
        let alpha: PyReadonlyArray1<'py, f64> = call.extract(4)?;
        let zeta = call.extract::<PlainReal>(5)?.0;
        let psi = call.extract::<PlainReal>(6)?.0;
        let n_steps = call.extract::<PlainUsize>(7)?.0;
        let mut p = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = knm
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let h_edges: Vec<hypergraph::Hyperedge> = edges
            .into_iter()
            .map(|(nodes, strength)| hypergraph::Hyperedge { nodes, strength })
            .collect();

        self.inner
            .run(&mut p, o, &h_edges, k, a, zeta, psi, n_steps)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    fn order_parameter(&self) -> (f64, f64) {
        self.inner.order_parameter()
    }
}

/// Stateful and trajectory Python calls for weighted hypergraph coupling.
///
/// Positional, keyword and mixed calls retain the same names and typed numerical inputs.
/// `edge_nodes` concatenates node indices; `edge_offsets` contains one start index
/// per edge, without a terminal offset. `edge_strengths` contains one weight per edge.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(phases, omegas, n, edge_nodes, edge_offsets, edge_strengths, pairwise_knm, alpha, zeta, psi, dt, n_steps)")]
pub(crate) fn hypergraph_run_rust<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let call = CallArguments::bind(
        args,
        kwargs,
        &[
            "phases",
            "omegas",
            "n",
            "edge_nodes",
            "edge_offsets",
            "edge_strengths",
            "pairwise_knm",
            "alpha",
            "zeta",
            "psi",
            "dt",
            "n_steps",
        ],
        "hypergraph_run_rust",
    )?;
    let py = args.py();
    let phases: PyReadonlyArray1<'py, f64> = call.extract(0)?;
    let omegas: PyReadonlyArray1<'py, f64> = call.extract(1)?;
    let n = call.extract::<PlainUsize>(2)?.0;
    let edge_nodes: PyReadonlyArray1<'py, i64> = call.extract(3)?;
    let edge_offsets: PyReadonlyArray1<'py, i64> = call.extract(4)?;
    let edge_strengths: PyReadonlyArray1<'py, f64> = call.extract(5)?;
    let pairwise_knm: PyReadonlyArray1<'py, f64> = call.extract(6)?;
    let alpha: PyReadonlyArray1<'py, f64> = call.extract(7)?;
    let zeta = call.extract::<PlainReal>(8)?.0;
    let psi = call.extract::<PlainReal>(9)?.0;
    let dt = call.extract::<PlainReal>(10)?.0;
    let n_steps = call.extract::<PlainUsize>(11)?.0;
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let en = edge_nodes
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let eo = edge_offsets
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let es = edge_strengths
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let kn = pairwise_knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let al = alpha
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if eo.len() != es.len()
        || (eo.is_empty() && !en.is_empty())
        || (!eo.is_empty() && (en.is_empty() || eo[0] != 0))
        || eo.iter().any(|&v| v < 0 || v as usize >= en.len())
        || eo.windows(2).any(|v| v[0] >= v[1])
        || en.iter().any(|&v| v < 0 || v as usize >= n)
    {
        return Err(PyValueError::new_err("invalid hyperedge flat encoding"));
    }
    let mut edges = Vec::with_capacity(eo.len());
    for i in 0..eo.len() {
        let start = eo[i] as usize;
        let end = eo.get(i + 1).map_or(en.len(), |&v| v as usize);
        let nodes: Vec<usize> = en[start..end].iter().map(|&v| v as usize).collect();
        if nodes.len() < 2
            || nodes
                .iter()
                .enumerate()
                .any(|(j, v)| nodes[..j].contains(v))
        {
            return Err(PyValueError::new_err("hyperedges require distinct nodes"));
        }
        edges.push(hypergraph::Hyperedge {
            nodes,
            strength: es[i],
        });
    }
    let config = IntegrationConfig {
        dt,
        ..Default::default()
    };
    let mut stepper = hypergraph::HypergraphStepper::new(n, config).map_err(spo_err)?;
    let mut result = p.to_vec();
    stepper
        .run(&mut result, o, &edges, kn, al, zeta, psi, n_steps)
        .map_err(spo_err)?;
    Ok(PyArray1::from_vec(py, result).into())
}
