// SCPN Phase Orchestrator — PyO3 FFI Bindings
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
//!
//! Python-callable wrappers around the spo-kernel Rust crates.
//!
//! Install: `cd spo-kernel && maturin develop --release -m crates/spo-ffi/Cargo.toml`

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use spo_engine::{
    coupling::{project_knm, CouplingBuilder},
    imprint::ImprintModel,
    order_params,
    upde::UPDEStepper,
};
use spo_oscillators::{informational, physical, quality::PhaseQualityScorer, symbolic};
use spo_supervisor::{
    boundaries::{BoundaryDef, BoundaryObserver, Severity},
    coherence::CoherenceMonitor,
    projector::ActionProjector,
    regime::RegimeManager,
};
use spo_types::{
    ControlAction, CouplingConfig, IntegrationConfig, Knob, LayerState, Method, Regime, UPDEState,
};

// ─── PyUPDEStepper ──────────────────────────────────────────────────

#[pyclass(name = "PyUPDEStepper")]
struct PyUPDEStepper {
    inner: UPDEStepper,
}

#[pymethods]
impl PyUPDEStepper {
    #[new]
    #[pyo3(signature = (n, dt = 0.01, method = "euler"))]
    fn new(n: usize, dt: f64, method: &str) -> PyResult<Self> {
        let m = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt,
            method: m,
            n_substeps: 1,
        };
        let inner =
            UPDEStepper::new(n, config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Advance phases by one step. Returns new phases list.
    fn step(
        &mut self,
        phases: Vec<f64>,
        omegas: Vec<f64>,
        knm: Vec<f64>,
        zeta: f64,
        psi: f64,
        alpha: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let mut p = phases;
        self.inner
            .step(&mut p, &omegas, &knm, zeta, psi, &alpha)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(p)
    }

    /// Run n_steps. Returns final phases.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        phases: Vec<f64>,
        omegas: Vec<f64>,
        knm: Vec<f64>,
        zeta: f64,
        psi: f64,
        alpha: Vec<f64>,
        n_steps: u64,
    ) -> PyResult<Vec<f64>> {
        let mut p = phases;
        self.inner
            .run(&mut p, &omegas, &knm, zeta, psi, &alpha, n_steps)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(p)
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }
}

// ─── PyCouplingBuilder ──────────────────────────────────────────────

#[pyclass(name = "PyCouplingBuilder")]
struct PyCouplingBuilder;

#[pymethods]
impl PyCouplingBuilder {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Build coupling matrix. Returns dict with knm, alpha, n.
    fn build<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        base_strength: f64,
        decay_alpha: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let config = CouplingConfig {
            base_strength,
            decay_alpha,
        };
        let cs =
            CouplingBuilder::build(n, &config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("knm", cs.knm)?;
        dict.set_item("alpha", cs.alpha)?;
        dict.set_item("n", cs.n)?;
        Ok(dict)
    }

    /// Project Knm to enforce symmetry, non-negative, zero diagonal.
    #[staticmethod]
    fn project(mut knm: Vec<f64>, n: usize) -> Vec<f64> {
        project_knm(&mut knm, n);
        knm
    }
}

// ─── PyRegimeManager ────────────────────────────────────────────────

#[pyclass(name = "PyRegimeManager")]
struct PyRegimeManager {
    inner: RegimeManager,
}

#[pymethods]
impl PyRegimeManager {
    #[new]
    #[pyo3(signature = (hysteresis = 0.05, cooldown_steps = 10))]
    fn new(hysteresis: f64, cooldown_steps: u64) -> Self {
        Self {
            inner: RegimeManager::new(hysteresis, cooldown_steps),
        }
    }

    /// Evaluate regime from layer R values and boundary violations.
    fn evaluate(&self, layer_rs: Vec<f64>, hard_violations: Vec<String>) -> String {
        let state = make_upde_state(&layer_rs);
        let boundary = make_boundary_state(hard_violations);
        let regime = self.inner.evaluate(&state, &boundary);
        regime_to_str(regime)
    }

    /// Apply transition with cooldown/hysteresis. Returns actual regime string.
    fn transition(&mut self, proposed: &str) -> PyResult<String> {
        let p = str_to_regime(proposed)?;
        let actual = self.inner.transition(p);
        Ok(regime_to_str(actual))
    }

    #[getter]
    fn current(&self) -> String {
        regime_to_str(self.inner.current)
    }
}

// ─── PyCoherenceMonitor ─────────────────────────────────────────────

#[pyclass(name = "PyCoherenceMonitor")]
struct PyCoherenceMonitor {
    inner: CoherenceMonitor,
}

#[pymethods]
impl PyCoherenceMonitor {
    #[new]
    fn new(good_layers: Vec<usize>, bad_layers: Vec<usize>) -> Self {
        Self {
            inner: CoherenceMonitor::new(good_layers, bad_layers),
        }
    }

    fn compute_r_good(&self, layer_rs: Vec<f64>) -> f64 {
        let state = make_upde_state(&layer_rs);
        self.inner.compute_r_good(&state)
    }

    fn compute_r_bad(&self, layer_rs: Vec<f64>) -> f64 {
        let state = make_upde_state(&layer_rs);
        self.inner.compute_r_bad(&state)
    }
}

// ─── PyBoundaryObserver ─────────────────────────────────────────────

#[pyclass(name = "PyBoundaryObserver")]
struct PyBoundaryObserver;

#[pymethods]
impl PyBoundaryObserver {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Observe boundary violations. Returns dict with violations, soft_violations, hard_violations.
    #[allow(clippy::type_complexity)]
    fn observe<'py>(
        &self,
        py: Python<'py>,
        defs: Vec<(String, String, Option<f64>, Option<f64>, String)>,
        values: HashMap<String, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let boundary_defs: Vec<BoundaryDef> = defs
            .into_iter()
            .map(|(name, variable, lower, upper, severity)| BoundaryDef {
                name,
                variable,
                lower,
                upper,
                severity: if severity == "hard" {
                    Severity::Hard
                } else {
                    Severity::Soft
                },
            })
            .collect();
        let state = BoundaryObserver::observe(&boundary_defs, &values);
        let dict = PyDict::new(py);
        dict.set_item("violations", state.violations)?;
        dict.set_item("soft_violations", state.soft_violations)?;
        dict.set_item("hard_violations", state.hard_violations)?;
        Ok(dict)
    }
}

// ─── PyImprintModel ─────────────────────────────────────────────────

#[pyclass(name = "PyImprintModel")]
struct PyImprintModel {
    inner: ImprintModel,
}

#[pymethods]
impl PyImprintModel {
    #[new]
    fn new(n: usize, decay_rate: f64, saturation: f64) -> PyResult<Self> {
        let inner = ImprintModel::new(n, decay_rate, saturation)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, exposure: Vec<f64>, dt: f64) {
        self.inner.update(&exposure, dt);
    }

    fn modulate_coupling(&self, mut knm: Vec<f64>) -> Vec<f64> {
        self.inner.modulate_coupling(&mut knm);
        knm
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn m(&self) -> Vec<f64> {
        self.inner.m.clone()
    }
}

// ─── PyActionProjector ──────────────────────────────────────────────

#[pyclass(name = "PyActionProjector")]
struct PyActionProjector {
    inner: ActionProjector,
}

#[pymethods]
impl PyActionProjector {
    #[new]
    fn new(
        rate_limits: HashMap<String, f64>,
        value_bounds: HashMap<String, (f64, f64)>,
    ) -> PyResult<Self> {
        let rl: HashMap<Knob, f64> = rate_limits
            .into_iter()
            .filter_map(|(k, v)| str_to_knob(&k).ok().map(|knob| (knob, v)))
            .collect();
        let vb: HashMap<Knob, (f64, f64)> = value_bounds
            .into_iter()
            .filter_map(|(k, v)| str_to_knob(&k).ok().map(|knob| (knob, v)))
            .collect();
        Ok(Self {
            inner: ActionProjector::new(rl, vb),
        })
    }

    fn project(&self, knob: &str, value: f64, previous_value: f64) -> PyResult<f64> {
        let k = str_to_knob(knob)?;
        let action = ControlAction {
            knob: k,
            scope: "global".into(),
            value,
            ttl_s: 10.0,
            justification: String::new(),
        };
        Ok(self.inner.project(&action, previous_value).value)
    }
}

// ─── PyPhaseQualityScorer ───────────────────────────────────────────

#[pyclass(name = "PyPhaseQualityScorer")]
struct PyPhaseQualityScorer {
    inner: PhaseQualityScorer,
}

#[pymethods]
impl PyPhaseQualityScorer {
    #[new]
    #[pyo3(signature = (collapse_threshold = 0.1, min_quality = 0.3))]
    fn new(collapse_threshold: f64, min_quality: f64) -> Self {
        Self {
            inner: PhaseQualityScorer {
                collapse_threshold,
                min_quality,
            },
        }
    }

    fn score(&self, qualities: Vec<f64>, amplitudes: Vec<f64>) -> f64 {
        self.inner.score(&qualities, &amplitudes)
    }

    fn is_collapsed(&self, qualities: Vec<f64>) -> bool {
        self.inner.is_collapsed(&qualities)
    }

    fn downweight_mask(&self, qualities: Vec<f64>) -> Vec<f64> {
        self.inner.downweight_mask(&qualities)
    }
}

// ─── Free Functions ─────────────────────────────────────────────────

#[pyfunction]
fn order_parameter(phases: Vec<f64>) -> (f64, f64) {
    order_params::compute_order_parameter(&phases)
}

#[pyfunction]
fn plv(phases_a: Vec<f64>, phases_b: Vec<f64>) -> f64 {
    order_params::compute_plv(&phases_a, &phases_b)
}

#[pyfunction]
fn ring_phase(state_index: usize, n_states: usize) -> f64 {
    symbolic::ring_phase(state_index, n_states)
}

#[pyfunction]
fn event_phase(timestamps: Vec<f64>) -> (f64, f64, f64) {
    informational::event_phase(&timestamps)
}

#[pyfunction]
fn physical_extract(real: Vec<f64>, imag: Vec<f64>, sample_rate: f64) -> (f64, f64, f64, f64) {
    physical::extract_from_analytic(&real, &imag, sample_rate)
}

// ─── Helpers ────────────────────────────────────────────────────────

fn make_upde_state(layer_rs: &[f64]) -> UPDEState {
    UPDEState {
        layers: layer_rs
            .iter()
            .map(|&r| LayerState { r, psi: 0.0 })
            .collect(),
        cross_layer_alignment: vec![],
        stability_proxy: 0.0,
        regime: Regime::Nominal,
    }
}

fn make_boundary_state(hard_violations: Vec<String>) -> spo_supervisor::boundaries::BoundaryState {
    spo_supervisor::boundaries::BoundaryState {
        violations: hard_violations.clone(),
        soft_violations: vec![],
        hard_violations,
    }
}

fn regime_to_str(r: Regime) -> String {
    match r {
        Regime::Nominal => "nominal",
        Regime::Degraded => "degraded",
        Regime::Critical => "critical",
        Regime::Recovery => "recovery",
    }
    .into()
}

fn str_to_regime(s: &str) -> PyResult<Regime> {
    match s {
        "nominal" => Ok(Regime::Nominal),
        "degraded" => Ok(Regime::Degraded),
        "critical" => Ok(Regime::Critical),
        "recovery" => Ok(Regime::Recovery),
        _ => Err(PyValueError::new_err(format!("unknown regime: {s}"))),
    }
}

fn str_to_knob(s: &str) -> PyResult<Knob> {
    match s {
        "K" => Ok(Knob::K),
        "alpha" => Ok(Knob::Alpha),
        "zeta" => Ok(Knob::Zeta),
        "Psi" => Ok(Knob::Psi),
        _ => Err(PyValueError::new_err(format!("unknown knob: {s}"))),
    }
}

// ─── Module Registration ────────────────────────────────────────────

#[pymodule]
fn spo_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUPDEStepper>()?;
    m.add_class::<PyCouplingBuilder>()?;
    m.add_class::<PyRegimeManager>()?;
    m.add_class::<PyCoherenceMonitor>()?;
    m.add_class::<PyBoundaryObserver>()?;
    m.add_class::<PyImprintModel>()?;
    m.add_class::<PyActionProjector>()?;
    m.add_class::<PyPhaseQualityScorer>()?;
    m.add_function(wrap_pyfunction!(order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(plv, m)?)?;
    m.add_function(wrap_pyfunction!(ring_phase, m)?)?;
    m.add_function(wrap_pyfunction!(event_phase, m)?)?;
    m.add_function(wrap_pyfunction!(physical_extract, m)?)?;
    Ok(())
}
