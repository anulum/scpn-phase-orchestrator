// SCPN Phase Orchestrator — PyO3 FFI Bindings
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
//!
//! Python-callable wrappers around the spo-kernel Rust crates.
//!
//! Install: `cd spo-kernel && maturin develop --release -m crates/spo-ffi/Cargo.toml`
//!
//! Note: `#![deny(unsafe_code)]` is omitted because PyO3 macros generate
//! unsafe blocks internally. We forbid undocumented unsafe instead.
#![forbid(clippy::undocumented_unsafe_blocks)]

use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use spo_engine::{
    coupling::{project_knm, CouplingBuilder},
    imprint::ImprintModel,
    lags::LagModel,
    order_params, pac,
    stuart_landau::StuartLandauStepper,
    upde::UPDEStepper,
};
use spo_oscillators::{informational, physical, quality::PhaseQualityScorer, symbolic};
use spo_supervisor::{
    boundaries::{BoundaryDef, BoundaryObserver, Severity},
    coherence::CoherenceMonitor,
    policy::SupervisorPolicy,
    projector::ActionProjector,
    regime::RegimeManager,
};
use spo_types::{
    ControlAction, CouplingConfig, IntegrationConfig, Knob, LayerState, Method, Regime, SpoError,
    UPDEState,
};

fn spo_err(e: SpoError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// (name, variable, lower_bound, upper_bound, severity)
type BoundaryDefTuple = Vec<(String, String, Option<f64>, Option<f64>, String)>;

// ─── PyUPDEStepper ──────────────────────────────────────────────────

#[pyclass(name = "PyUPDEStepper")]
struct PyUPDEStepper {
    inner: UPDEStepper,
}

#[pymethods]
impl PyUPDEStepper {
    #[new]
    #[pyo3(signature = (n, dt = 0.01, method = "euler", n_substeps = 1, atol = 1e-6, rtol = 1e-3))]
    fn new(
        n: usize,
        dt: f64,
        method: &str,
        n_substeps: u32,
        atol: f64,
        rtol: f64,
    ) -> PyResult<Self> {
        let m = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            "rk45" => Method::RK45,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt,
            method: m,
            n_substeps,
            atol,
            rtol,
        };
        let inner = UPDEStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Advance phases by one step. Returns new phases as numpy array.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: PyReadonlyArray1<'py, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
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
        self.inner
            .step(&mut p, o, k, zeta, psi, a)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    /// Run n_steps. Returns final phases as numpy array.
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: PyReadonlyArray1<'py, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
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
        self.inner
            .run(&mut p, o, k, zeta, psi, a, n_steps)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
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
        let cs = CouplingBuilder::build(n, &config).map_err(spo_err)?;
        let dict = PyDict::new(py);
        dict.set_item("knm", cs.knm)?;
        dict.set_item("alpha", cs.alpha)?;
        dict.set_item("n", cs.n)?;
        Ok(dict)
    }

    /// Project Knm to enforce symmetry, non-negative, zero diagonal.
    #[staticmethod]
    fn project(mut knm: Vec<f64>, n: usize) -> PyResult<Vec<f64>> {
        project_knm(&mut knm, n).map_err(spo_err)?;
        Ok(knm)
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

    /// Detect phase-locked layer pairs from cross-layer alignment matrix.
    ///
    /// `psi_values`: per-layer mean phase (same length as `layer_rs`).
    /// `cross_layer_alignment`: flattened n×n PLV matrix.
    fn detect_phase_lock(
        &self,
        layer_rs: Vec<f64>,
        psi_values: Vec<f64>,
        cross_layer_alignment: Vec<f64>,
        threshold: f64,
    ) -> Vec<(usize, usize)> {
        let state = make_upde_state_full(&layer_rs, &psi_values, &cross_layer_alignment);
        self.inner.detect_phase_lock(&state, threshold)
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
    fn observe<'py>(
        &self,
        py: Python<'py>,
        defs: BoundaryDefTuple,
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
        let inner = ImprintModel::new(n, decay_rate, saturation).map_err(spo_err)?;
        Ok(Self { inner })
    }

    fn update(&mut self, exposure: Vec<f64>, dt: f64) {
        self.inner.update(&exposure, dt);
    }

    fn modulate_coupling(&self, mut knm: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.modulate_coupling(&mut knm).map_err(spo_err)?;
        Ok(knm)
    }

    fn modulate_lag(&self, mut alpha: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.modulate_lag(&mut alpha).map_err(spo_err)?;
        Ok(alpha)
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

// ─── PyLagModel ─────────────────────────────────────────────────────

#[pyclass(name = "PyLagModel")]
struct PyLagModel {
    inner: LagModel,
}

#[pymethods]
impl PyLagModel {
    #[staticmethod]
    fn estimate(distances: Vec<f64>, n: usize, speed: f64) -> PyResult<Self> {
        let inner = LagModel::estimate_from_distances(&distances, n, speed).map_err(spo_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn zeros(n: usize) -> Self {
        Self {
            inner: LagModel::zeros(n),
        }
    }

    #[getter]
    fn alpha(&self) -> Vec<f64> {
        self.inner.alpha.clone()
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
}

// ─── PySupervisorPolicy ─────────────────────────────────────────────

#[pyclass(name = "PySupervisorPolicy")]
struct PySupervisorPolicy {
    inner: SupervisorPolicy,
}

#[pymethods]
impl PySupervisorPolicy {
    #[new]
    #[pyo3(signature = (hysteresis = 0.05, cooldown_steps = 10))]
    fn new(hysteresis: f64, cooldown_steps: u64) -> Self {
        Self {
            inner: SupervisorPolicy::new(RegimeManager::new(hysteresis, cooldown_steps)),
        }
    }

    fn decide<'py>(
        &mut self,
        py: Python<'py>,
        layer_rs: Vec<f64>,
        hard_violations: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let state = make_upde_state(&layer_rs);
        let boundary = make_boundary_state(hard_violations);
        let actions = self.inner.decide(&state, &boundary);
        actions
            .into_iter()
            .map(|a| {
                let d = PyDict::new(py);
                d.set_item("knob", knob_to_str(a.knob))?;
                d.set_item("scope", a.scope)?;
                d.set_item("value", a.value)?;
                d.set_item("ttl_s", a.ttl_s)?;
                d.set_item("justification", a.justification)?;
                Ok(d)
            })
            .collect()
    }
}

// ─── PyStuartLandauStepper ─────────────────────────────────────────

#[pyclass(name = "PyStuartLandauStepper")]
struct PyStuartLandauStepper {
    inner: StuartLandauStepper,
}

#[pymethods]
impl PyStuartLandauStepper {
    #[new]
    #[pyo3(signature = (n, dt = 0.01, method = "euler", n_substeps = 1, atol = 1e-6, rtol = 1e-3))]
    fn new(
        n: usize,
        dt: f64,
        method: &str,
        n_substeps: u32,
        atol: f64,
        rtol: f64,
    ) -> PyResult<Self> {
        let m = match method {
            "euler" => Method::Euler,
            "rk4" => Method::RK4,
            "rk45" => Method::RK45,
            _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
        };
        let config = IntegrationConfig {
            dt,
            method: m,
            n_substeps,
            atol,
            rtol,
        };
        let inner = StuartLandauStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    /// Advance state [θ; r] by one step. Returns new state list (2N).
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        state: Vec<f64>,
        omegas: PyReadonlyArray1<'_, f64>,
        mu: PyReadonlyArray1<'_, f64>,
        knm: PyReadonlyArray1<'_, f64>,
        knm_r: PyReadonlyArray1<'_, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'_, f64>,
        epsilon: f64,
    ) -> PyResult<Vec<f64>> {
        let mut s = state;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let m = mu
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = knm
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let kr = knm_r
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .step(&mut s, o, m, k, kr, zeta, psi, a, epsilon)
            .map_err(spo_err)?;
        Ok(s)
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
    }
}

// ─── PAC Functions ──────────────────────────────────────────────────

#[pyfunction]
fn pac_modulation_index(
    theta_low: PyReadonlyArray1<'_, f64>,
    amp_high: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<f64> {
    let t = theta_low
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = amp_high
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pac::modulation_index(t, a, n_bins))
}

#[pyfunction]
fn pac_matrix_compute(
    phases: PyReadonlyArray1<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    t: usize,
    n: usize,
    n_bins: usize,
) -> PyResult<Vec<f64>> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = amplitudes
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pac::pac_matrix(p, a, t, n, n_bins))
}

// ─── Free Functions ─────────────────────────────────────────────────

#[pyfunction]
fn order_parameter(phases: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    let s = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(order_params::compute_order_parameter(s))
}

#[pyfunction]
fn plv(phases_a: PyReadonlyArray1<'_, f64>, phases_b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let a = phases_a
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = phases_b
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    order_params::compute_plv(a, b).map_err(spo_err)
}

#[pyfunction]
fn ring_phase(state_index: usize, n_states: usize) -> f64 {
    symbolic::ring_phase(state_index, n_states)
}

#[pyfunction]
fn event_phase(timestamps: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64, f64)> {
    let s = timestamps
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(informational::event_phase(s))
}

#[pyfunction]
fn physical_extract(
    real: PyReadonlyArray1<'_, f64>,
    imag: PyReadonlyArray1<'_, f64>,
    sample_rate: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let r = real
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let i = imag
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(physical::extract_from_analytic(r, i, sample_rate))
}

#[pyfunction]
fn graph_walk_phase(position: usize, walk_length: usize) -> f64 {
    symbolic::graph_walk_phase(position, walk_length)
}

#[pyfunction]
fn transition_quality(step_size: usize, n_states: usize) -> f64 {
    symbolic::transition_quality(step_size, n_states)
}

#[pyfunction]
fn layer_coherence(phases: PyReadonlyArray1<'_, f64>, indices: Vec<usize>) -> PyResult<f64> {
    let s = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(order_params::compute_layer_coherence(s, &indices))
}

// ─── Helpers ────────────────────────────────────────────────────────

fn make_upde_state(layer_rs: &[f64]) -> UPDEState {
    make_upde_state_full(layer_rs, &[], &[])
}

fn make_upde_state_full(layer_rs: &[f64], psi_values: &[f64], cla: &[f64]) -> UPDEState {
    let layers: Vec<LayerState> = layer_rs
        .iter()
        .enumerate()
        .map(|(i, &r)| LayerState {
            r,
            psi: psi_values.get(i).copied().unwrap_or(0.0),
        })
        .collect();
    let stability_proxy = if layers.is_empty() {
        0.0
    } else {
        layers.iter().map(|l| l.r).sum::<f64>() / layers.len() as f64
    };
    UPDEState {
        layers,
        cross_layer_alignment: cla.to_vec(),
        stability_proxy,
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

fn knob_to_str(k: Knob) -> &'static str {
    match k {
        Knob::K => "K",
        Knob::Alpha => "alpha",
        Knob::Zeta => "zeta",
        Knob::Psi => "Psi",
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
    m.add_class::<PyLagModel>()?;
    m.add_class::<PySupervisorPolicy>()?;
    m.add_class::<PyStuartLandauStepper>()?;
    m.add_function(wrap_pyfunction!(pac_modulation_index, m)?)?;
    m.add_function(wrap_pyfunction!(pac_matrix_compute, m)?)?;
    m.add_function(wrap_pyfunction!(order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(plv, m)?)?;
    m.add_function(wrap_pyfunction!(ring_phase, m)?)?;
    m.add_function(wrap_pyfunction!(event_phase, m)?)?;
    m.add_function(wrap_pyfunction!(physical_extract, m)?)?;
    m.add_function(wrap_pyfunction!(graph_walk_phase, m)?)?;
    m.add_function(wrap_pyfunction!(transition_quality, m)?)?;
    m.add_function(wrap_pyfunction!(layer_coherence, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_upde_state_layer_count() {
        let s = make_upde_state(&[0.5, 0.6, 0.7]);
        assert_eq!(s.layers.len(), 3);
        assert!((s.layers[0].r - 0.5).abs() < 1e-12);
    }

    #[test]
    fn make_upde_state_r_values() {
        let s = make_upde_state(&[0.1, 0.9]);
        assert!((s.mean_r() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn make_upde_state_full_psi_and_cla() {
        let s = make_upde_state_full(&[0.5, 0.8], &[1.0, 2.0], &[0.0, 0.9, 0.9, 0.0]);
        assert!((s.layers[0].psi - 1.0).abs() < 1e-12);
        assert!((s.layers[1].psi - 2.0).abs() < 1e-12);
        assert_eq!(s.cross_layer_alignment.len(), 4);
        assert!((s.stability_proxy - 0.65).abs() < 1e-12);
    }

    #[test]
    fn make_boundary_state_hard_violations() {
        let b = make_boundary_state(vec!["v1".into(), "v2".into()]);
        assert_eq!(b.hard_violations.len(), 2);
        assert!(b.soft_violations.is_empty());
    }

    #[test]
    fn regime_to_str_roundtrip() {
        for (regime, expected) in [
            (Regime::Nominal, "nominal"),
            (Regime::Degraded, "degraded"),
            (Regime::Critical, "critical"),
            (Regime::Recovery, "recovery"),
        ] {
            assert_eq!(regime_to_str(regime), expected);
            assert_eq!(str_to_regime(expected).unwrap(), regime);
        }
    }

    #[test]
    fn str_to_regime_error() {
        assert!(str_to_regime("invalid").is_err());
    }

    #[test]
    fn knob_roundtrip() {
        for (knob, expected) in [
            (Knob::K, "K"),
            (Knob::Alpha, "alpha"),
            (Knob::Zeta, "zeta"),
            (Knob::Psi, "Psi"),
        ] {
            assert_eq!(knob_to_str(knob), expected);
            assert_eq!(str_to_knob(expected).unwrap(), knob);
        }
    }

    #[test]
    fn str_to_knob_error() {
        assert!(str_to_knob("invalid").is_err());
    }
}
