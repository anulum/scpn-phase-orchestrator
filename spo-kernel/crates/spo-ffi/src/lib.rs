// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Python FFI bindings

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
    chimera,
    coupling::{project_knm, CouplingBuilder},
    basin_stability, bifurcation, delay, dimension, ei_balance, embedding, entropy_prod,
    free_energy, hodge, inertial, itpc, market, poincare, psychedelic, ssgf_costs,
    swarmalator,
    imprint::ImprintModel,
    lags::LagModel,
    lif_ensemble::{LIFEnsemble, LIFParams},
    lyapunov, order_params, pac,
    plasticity::PlasticityModel,
    recurrence,
    sheaf_upde::SheafUPDEStepper,
    sparse_upde::SparseUPDEStepper,
    spectral,
    stuart_landau::StuartLandauStepper,
    transfer_entropy,
    upde::UPDEStepper,
    winding,
};
use spo_oscillators::{informational, physical, quality::PhaseQualityScorer, symbolic};
use spo_supervisor::{
    active_inference::ActiveInferenceAgent,
    boundaries::{BoundaryDef, BoundaryObserver, Severity},
    coherence::CoherenceMonitor,
    petri_net,
    policy::SupervisorPolicy,
    projector::ActionProjector,
    regime::RegimeManager,
    rule_engine,
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

// ─── PyActiveInferenceAgent ──────────────────────────────────────────────

#[pyclass(name = "PyActiveInferenceAgent")]
struct PyActiveInferenceAgent {
    inner: ActiveInferenceAgent,
}

#[pymethods]
impl PyActiveInferenceAgent {
    #[new]
    #[pyo3(signature = (n_hidden = 4, target_r = 0.5, lr = 1.0))]
    fn new(n_hidden: usize, target_r: f64, lr: f64) -> PyResult<Self> {
        let inner = ActiveInferenceAgent::new(n_hidden, target_r, lr).map_err(spo_err)?;
        Ok(Self { inner })
    }

    fn control(&mut self, r_obs: f64, psi_obs: f64, dt: f64) -> (f64, f64) {
        self.inner.control(r_obs, psi_obs, dt)
    }

    #[getter]
    fn target_r(&self) -> f64 {
        self.inner.target_r
    }

    #[setter]
    fn set_target_r(&mut self, val: f64) {
        self.inner.target_r = val;
    }
}

// ─── PySheafUPDEStepper ───────────────────────────────────────────────────

#[pyclass(name = "PySheafUPDEStepper")]
struct PySheafUPDEStepper {
    inner: SheafUPDEStepper,
}

#[pymethods]
impl PySheafUPDEStepper {
    #[new]
    #[pyo3(signature = (n, d, dt = 0.01, method = "euler", n_substeps = 1, atol = 1e-6, rtol = 1e-3))]
    fn new(
        n: usize,
        d: usize,
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
        let inner = SheafUPDEStepper::new(n, d, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        restriction_maps: PyReadonlyArray1<'py, f64>,
        zeta: f64,
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
            .step(&mut p_out, p_w, r_m, zeta, p_psi)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        restriction_maps: PyReadonlyArray1<'py, f64>,
        zeta: f64,
        psi: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
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
            .run(&mut p_out, p_w, r_m, zeta, p_psi, n_steps)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d()
    }

    #[getter]
    fn last_dt(&self) -> f64 {
        self.inner.last_dt()
    }
}

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
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        knm: Bound<'py, PyArray1<f64>>,
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
        let mut k_bound = knm.readwrite();
        let k = k_bound
            .as_slice_mut()
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
        knm: Bound<'py, PyArray1<f64>>,
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
        let mut k_bound = knm.readwrite();
        let k = k_bound
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .run(&mut p, o, k, zeta, psi, a, n_steps)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, p))
    }

    #[pyo3(signature = (lr, decay = 0.0, modulator = 1.0))]
    fn set_plasticity(&mut self, lr: f64, decay: f64, modulator: f64) -> PyResult<()> {
        self.inner.plasticity = Some(PlasticityModel::new(lr, decay).map_err(spo_err)?);
        self.inner.modulator = modulator;
        Ok(())
    }

    fn disable_plasticity(&mut self) {
        self.inner.plasticity = None;
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
    #[pyo3(signature = (hysteresis = 0.05, cooldown_steps = 10, hysteresis_hold_steps = 0))]
    fn new(hysteresis: f64, cooldown_steps: u64, hysteresis_hold_steps: u64) -> Self {
        Self {
            inner: RegimeManager::new(hysteresis, cooldown_steps)
                .with_hold_steps(hysteresis_hold_steps),
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

    /// Bypass cooldown and hysteresis hold — event-driven triggers.
    fn force_transition(&mut self, regime: &str) -> PyResult<String> {
        let r = str_to_regime(regime)?;
        let actual = self.inner.force_transition(r);
        Ok(regime_to_str(actual))
    }

    #[getter]
    fn current(&self) -> String {
        regime_to_str(self.inner.current)
    }

    #[getter]
    fn transition_log(&self) -> Vec<(u64, String, String)> {
        self.inner
            .transition_log
            .iter()
            .map(|(step, prev, new)| (*step, regime_to_str(*prev), regime_to_str(*new)))
            .collect()
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
        py: Python<'_>,
        rate_limits: HashMap<String, f64>,
        value_bounds: HashMap<String, (f64, f64)>,
    ) -> PyResult<Self> {
        let warnings = py.import("warnings")?;
        let rl: HashMap<Knob, f64> = rate_limits
            .into_iter()
            .filter_map(|(k, v)| match str_to_knob(&k) {
                Ok(knob) => Some((knob, v)),
                Err(_) => {
                    let _ = warnings.call_method1(
                        "warn",
                        (format!(
                            "PyActionProjector: ignoring unknown knob {k:?} in rate_limits"
                        ),),
                    );
                    None
                }
            })
            .collect();
        let vb: HashMap<Knob, (f64, f64)> = value_bounds
            .into_iter()
            .filter_map(|(k, v)| match str_to_knob(&k) {
                Ok(knob) => Some((knob, v)),
                Err(_) => {
                    let _ = warnings.call_method1(
                        "warn",
                        (format!(
                            "PyActionProjector: ignoring unknown knob {k:?} in value_bounds"
                        ),),
                    );
                    None
                }
            })
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

// ─── PySparseUPDEStepper ──────────────────────────────────────────────────

#[pyclass(name = "PySparseUPDEStepper")]
struct PySparseUPDEStepper {
    inner: SparseUPDEStepper,
}

#[pymethods]
impl PySparseUPDEStepper {
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
        let inner = SparseUPDEStepper::new(n, config).map_err(spo_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (lr, decay = 0.0, modulator = 1.0))]
    fn set_plasticity(&mut self, lr: f64, decay: f64, modulator: f64) -> PyResult<()> {
        self.inner.plasticity = Some(PlasticityModel::new(lr, decay).map_err(spo_err)?);
        self.inner.modulator = modulator;
        Ok(())
    }

    fn disable_plasticity(&mut self) {
        self.inner.plasticity = None;
    }

    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        row_ptr: PyReadonlyArray1<'py, usize>,
        col_indices: PyReadonlyArray1<'py, usize>,
        knm_values: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha_values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let p_w = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rp = row_ptr
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ci = col_indices
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut kv_bound = knm_values.readwrite();
        let kv = kv_bound
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let av = alpha_values
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .step(&mut p_out, p_w, rp, ci, kv, zeta, psi, av)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'py, f64>,
        omegas: PyReadonlyArray1<'py, f64>,
        row_ptr: PyReadonlyArray1<'py, usize>,
        col_indices: PyReadonlyArray1<'py, usize>,
        knm_values: Bound<'py, PyArray1<f64>>,
        zeta: f64,
        psi: f64,
        alpha_values: PyReadonlyArray1<'py, f64>,
        n_steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut p_out = phases
            .to_vec()
            .map_err(|_| PyValueError::new_err("phases not contiguous"))?;
        let p_w = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rp = row_ptr
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ci = col_indices
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut kv_bound = knm_values.readwrite();
        let kv = kv_bound
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let av = alpha_values
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.inner
            .run(&mut p_out, p_w, rp, ci, kv, zeta, psi, av, n_steps)
            .map_err(spo_err)?;

        Ok(PyArray1::from_vec(py, p_out))
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

    /// Advance state [θ; r] by one step. Returns new state array (2N).
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        state: PyReadonlyArray1<'_, f64>,
        omegas: PyReadonlyArray1<'_, f64>,
        mu: PyReadonlyArray1<'_, f64>,
        knm: PyReadonlyArray1<'_, f64>,
        knm_r: PyReadonlyArray1<'_, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'_, f64>,
        epsilon: f64,
    ) -> PyResult<Vec<f64>> {
        let mut s = state
            .to_vec()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let m = mu
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut k = knm
            .to_vec()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let kr = knm_r
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .step(&mut s, o, m, &mut k, kr, zeta, psi, a, epsilon)
            .map_err(spo_err)?;
        Ok(s)
    }

    /// Run n_steps. Returns final state list (2N).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, n_steps))]
    fn run(
        &mut self,
        state: PyReadonlyArray1<'_, f64>,
        omegas: PyReadonlyArray1<'_, f64>,
        mu: PyReadonlyArray1<'_, f64>,
        knm: PyReadonlyArray1<'_, f64>,
        knm_r: PyReadonlyArray1<'_, f64>,
        zeta: f64,
        psi: f64,
        alpha: PyReadonlyArray1<'_, f64>,
        epsilon: f64,
        n_steps: u64,
    ) -> PyResult<Vec<f64>> {
        let mut s = state
            .to_vec()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let o = omegas
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let m = mu
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut k = knm
            .to_vec()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let kr = knm_r
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .run(&mut s, o, m, &mut k, kr, zeta, psi, a, epsilon, n_steps)
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

// ─── PyRuleEngine ───────────────────────────────────────────────

/// (metric, op_str, threshold)
type CondTuple = (String, String, f64);
/// (knob, scope, value, ttl_s)
type ActionTuple = (String, String, f64, f64);
/// (name, regimes, conditions, logic, actions, cooldown_s, max_fires)
type RuleTuple = (
    String,
    Vec<String>,
    Vec<CondTuple>,
    String,
    Vec<ActionTuple>,
    f64,
    u32,
);

fn parse_op(s: &str) -> PyResult<petri_net::GuardOp> {
    match s {
        ">" => Ok(petri_net::GuardOp::Gt),
        ">=" => Ok(petri_net::GuardOp::Ge),
        "<" => Ok(petri_net::GuardOp::Lt),
        "<=" => Ok(petri_net::GuardOp::Le),
        "==" => Ok(petri_net::GuardOp::Eq),
        _ => Err(PyValueError::new_err(format!("unknown op: {s}"))),
    }
}

#[pyclass(name = "PyRuleEngine")]
struct PyRuleEngine {
    inner: rule_engine::RuleEngine,
}

#[pymethods]
impl PyRuleEngine {
    #[new]
    fn new(rules: Vec<RuleTuple>) -> PyResult<Self> {
        let parsed: Vec<rule_engine::PolicyRule> = rules
            .into_iter()
            .map(
                |(name, regimes, conds, logic, actions, cooldown_s, max_fires)| {
                    let conditions: Vec<rule_engine::Condition> = conds
                        .into_iter()
                        .map(|(metric, op, threshold)| {
                            Ok(rule_engine::Condition {
                                metric,
                                op: parse_op(&op)?,
                                threshold,
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    let condition = if conditions.len() == 1 {
                        rule_engine::RuleCondition::Single(
                            conditions.into_iter().next().expect("len==1"),
                        )
                    } else {
                        let logic = match logic.to_uppercase().as_str() {
                            "OR" => rule_engine::Logic::Or,
                            _ => rule_engine::Logic::And,
                        };
                        rule_engine::RuleCondition::Compound { conditions, logic }
                    };
                    let rule_actions: Vec<rule_engine::RuleAction> = actions
                        .into_iter()
                        .map(|(knob, scope, value, ttl_s)| rule_engine::RuleAction {
                            knob,
                            scope,
                            value,
                            ttl_s,
                        })
                        .collect();
                    Ok(rule_engine::PolicyRule {
                        name,
                        regimes: regimes.into_iter().map(|r| r.to_uppercase()).collect(),
                        condition,
                        actions: rule_actions,
                        cooldown_s,
                        max_fires,
                    })
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: rule_engine::RuleEngine::new(parsed),
        })
    }

    /// Evaluate rules. Returns list of (knob, scope, value, ttl_s, rule_name).
    fn evaluate(
        &mut self,
        regime: &str,
        ctx: HashMap<String, f64>,
    ) -> Vec<(String, String, f64, f64, String)> {
        self.inner
            .evaluate(regime, &ctx)
            .into_iter()
            .map(|a| (a.knob, a.scope, a.value, a.ttl_s, a.rule_name))
            .collect()
    }

    fn advance_clock(&mut self, dt: f64) {
        self.inner.advance_clock(dt);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn clock(&self) -> f64 {
        self.inner.clock()
    }
}

// ─── PyPetriNet ─────────────────────────────────────────────────

/// (place_name, weight)
type ArcTuple = (String, u32);
/// (name, inputs, outputs, guard_text_or_none)
type TransitionTuple = (String, Vec<ArcTuple>, Vec<ArcTuple>, Option<String>);

#[pyclass(name = "PyPetriNet")]
struct PyPetriNet {
    inner: petri_net::PetriNet,
}

#[pymethods]
impl PyPetriNet {
    #[new]
    fn new(places: Vec<String>, transitions: Vec<TransitionTuple>) -> PyResult<Self> {
        let ts: Vec<petri_net::Transition> = transitions
            .into_iter()
            .map(|(name, inputs, outputs, guard_text)| {
                let guard =
                    match guard_text {
                        Some(text) => Some(petri_net::parse_guard(&text).map_err(|e| {
                            PyValueError::new_err(format!("guard parse error: {e}"))
                        })?),
                        None => None,
                    };
                Ok(petri_net::Transition {
                    name,
                    inputs: inputs
                        .into_iter()
                        .map(|(p, w)| petri_net::Arc {
                            place: p,
                            weight: w,
                        })
                        .collect(),
                    outputs: outputs
                        .into_iter()
                        .map(|(p, w)| petri_net::Arc {
                            place: p,
                            weight: w,
                        })
                        .collect(),
                    guard,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = petri_net::PetriNet::new(places, ts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Step: fire first enabled transition. Returns (tokens_dict, fired_name_or_none).
    fn step<'py>(
        &self,
        py: Python<'py>,
        tokens: HashMap<String, u32>,
        ctx: HashMap<String, f64>,
    ) -> PyResult<(Bound<'py, PyDict>, Option<String>)> {
        let mut marking = petri_net::Marking::default();
        for (p, n) in &tokens {
            marking.set(p, *n);
        }
        let (new_marking, fired_idx) = self.inner.step(&marking, &ctx);
        let dict = PyDict::new(py);
        for p in self.inner.place_names() {
            let n = new_marking.get(p);
            if n > 0 {
                dict.set_item(p, n)?;
            }
        }
        let fired_name = fired_idx.map(|i| self.inner.transitions()[i].name.clone());
        Ok((dict, fired_name))
    }

    /// Return names of all enabled transitions.
    fn enabled(&self, tokens: HashMap<String, u32>, ctx: HashMap<String, f64>) -> Vec<String> {
        let mut marking = petri_net::Marking::default();
        for (p, n) in &tokens {
            marking.set(p, *n);
        }
        self.inner
            .enabled(&marking, &ctx)
            .into_iter()
            .map(|i| self.inner.transitions()[i].name.clone())
            .collect()
    }

    #[getter]
    fn place_names(&self) -> Vec<String> {
        self.inner.place_names().to_vec()
    }
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

// ─── PyLIFEnsemble ──────────────────────────────────────────────────

#[pyclass(name = "PyLIFEnsemble")]
struct PyLIFEnsemble {
    inner: LIFEnsemble,
}

#[pymethods]
impl PyLIFEnsemble {
    #[new]
    #[pyo3(signature = (n_layers, neurons_per_layer, noise_std = 0.0))]
    fn new(n_layers: usize, neurons_per_layer: usize, noise_std: f64) -> PyResult<Self> {
        let params = LIFParams {
            noise_std,
            ..LIFParams::default()
        };
        let inner = LIFEnsemble::new(n_layers, neurons_per_layer, params).map_err(spo_err)?;
        Ok(Self { inner })
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        currents: PyReadonlyArray1<'py, f64>,
        n_substeps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let currents_slice = currents
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("currents array not contiguous: {e}")))?;
        let rates = self
            .inner
            .step(currents_slice, n_substeps)
            .map_err(spo_err)?;
        Ok(PyArray1::from_vec(py, rates))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn n_total(&self) -> usize {
        self.inner.n_total()
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    #[getter]
    fn neurons_per_layer(&self) -> usize {
        self.inner.neurons_per_layer()
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn get_neuron_states(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut states = Vec::with_capacity(self.inner.n_total());
        for i in 0..self.inner.n_total() {
            if let Some((v, refr)) = self.inner.neuron_state(i) {
                let d = PyDict::new(py);
                d.set_item("v", v)?;
                d.set_item("refractory", refr)?;
                states.push(d.unbind());
            }
        }
        Ok(states)
    }

    fn voltages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.voltages())
    }

    fn spike_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_slice(py, self.inner.spike_counts_slice())
    }
}

// ─── PID (Redundancy / Synergy) ─────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (phases, group_a, group_b, n_bins = 32))]
fn pid_redundancy(
    phases: PyReadonlyArray1<'_, f64>,
    group_a: Vec<usize>,
    group_b: Vec<usize>,
    n_bins: usize,
) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spo_engine::pid::redundancy(p, &group_a, &group_b, n_bins))
}

#[pyfunction]
#[pyo3(signature = (phases, group_a, group_b, n_bins = 32))]
fn pid_synergy(
    phases: PyReadonlyArray1<'_, f64>,
    group_a: Vec<usize>,
    group_b: Vec<usize>,
    n_bins: usize,
) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spo_engine::pid::synergy(p, &group_a, &group_b, n_bins))
}

// ─── Normalized Persistent Entropy ──────────────────────────────────

#[pyfunction]
#[pyo3(signature = (phases, max_radius = std::f64::consts::PI))]
fn compute_npe(phases: PyReadonlyArray1<'_, f64>, max_radius: f64) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spo_engine::npe::compute_npe(p, max_radius))
}

#[pyfunction]
fn phase_distance_matrix<'py>(
    py: Python<'py>,
    phases: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dist = spo_engine::npe::phase_distance_matrix(p);
    Ok(PyArray1::from_slice(py, &dist))
}

// ─── Entropy Production Rate ────────────────────────────────────────

#[pyfunction]
fn entropy_production_rate(
    phases: PyReadonlyArray1<'_, f64>,
    omegas: PyReadonlyArray1<'_, f64>,
    knm: PyReadonlyArray1<'_, f64>,
    alpha: f64,
    dt: f64,
) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(entropy_prod::entropy_production_rate(p, o, k, alpha, dt))
}

// ─── Winding Numbers ────────────────────────────────────────────────

#[pyfunction]
fn winding_numbers<'py>(
    py: Python<'py>,
    phases_history: PyReadonlyArray1<'_, f64>,
    n_oscillators: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let flat = phases_history
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if n_oscillators == 0 {
        return Ok(PyArray1::from_slice(py, &[]));
    }
    let t = flat.len() / n_oscillators;
    let result = winding::winding_numbers(flat, t, n_oscillators);
    Ok(PyArray1::from_slice(py, &result))
}

// ─── Lyapunov Spectrum ──────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (phases_init, omegas, knm, alpha, dt = 0.01, n_steps = 1000, qr_interval = 10, zeta = 0.0, psi = 0.0))]
#[allow(clippy::too_many_arguments)]
fn lyapunov_spectrum_rust<'py>(
    py: Python<'py>,
    phases_init: PyReadonlyArray1<'py, f64>,
    omegas: PyReadonlyArray1<'py, f64>,
    knm: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    dt: f64,
    n_steps: usize,
    qr_interval: usize,
    zeta: f64,
    psi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases_init
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = lyapunov::lyapunov_spectrum(p, o, k, a, dt, n_steps, qr_interval, zeta, psi)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

// ─── Fractal Dimension ──────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (trajectory, t, d, epsilons, max_pairs = 50000, seed = 42))]
#[allow(clippy::too_many_arguments)]
fn correlation_integral_rust<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray1<'py, f64>,
    t: usize,
    d: usize,
    epsilons: PyReadonlyArray1<'py, f64>,
    max_pairs: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let traj = trajectory
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let eps = epsilons
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = dimension::correlation_integral(traj, t, d, eps, max_pairs, seed)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn kaplan_yorke_dimension_rust(exponents: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let le = exponents
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(dimension::kaplan_yorke_dimension(le))
}

// ─── Chimera Detection ──────────────────────────────────────────────

/// Returns (coherent_indices, incoherent_indices, chimera_index, local_order).
#[pyfunction]
fn detect_chimera_rust<'py>(
    py: Python<'py>,
    phases: PyReadonlyArray1<'py, f64>,
    knm: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Vec<usize>, Vec<usize>, f64, Bound<'py, PyArray1<f64>>)> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = chimera::detect_chimera(p, k, n);
    Ok((
        result.coherent_indices,
        result.incoherent_indices,
        result.chimera_index,
        PyArray1::from_vec(py, result.local_order),
    ))
}

// ─── Spectral Analysis ──────────────────────────────────────────────

#[pyfunction]
fn fiedler_value_rust(knm: PyReadonlyArray1<'_, f64>, n: usize) -> PyResult<f64> {
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spectral::fiedler_value(k, n))
}

#[pyfunction]
fn fiedler_vector_rust<'py>(
    py: Python<'py>,
    knm: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v = spectral::fiedler_vector(k, n);
    Ok(PyArray1::from_vec(py, v))
}

#[pyfunction]
fn spectral_gap_rust(knm: PyReadonlyArray1<'_, f64>, n: usize) -> PyResult<f64> {
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spectral::spectral_gap(k, n))
}

#[pyfunction]
fn critical_coupling_rust(
    omegas: PyReadonlyArray1<'_, f64>,
    knm: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> PyResult<f64> {
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spectral::critical_coupling(o, k, n))
}

#[pyfunction]
fn sync_convergence_rate_rust(
    knm: PyReadonlyArray1<'_, f64>,
    omegas: PyReadonlyArray1<'_, f64>,
    n: usize,
    gamma_max: f64,
) -> PyResult<f64> {
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(spectral::sync_convergence_rate(k, o, n, gamma_max))
}

// ─── Transfer Entropy ───────────────────────────────────────────────

#[pyfunction]
fn phase_transfer_entropy_rust(
    source: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<f64> {
    let s = source
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = target
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(transfer_entropy::phase_transfer_entropy(s, t, n_bins))
}

#[pyfunction]
fn transfer_entropy_matrix_rust<'py>(
    py: Python<'py>,
    phase_series: PyReadonlyArray1<'py, f64>,
    n_osc: usize,
    n_time: usize,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ps = phase_series
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = transfer_entropy::transfer_entropy_matrix(ps, n_osc, n_time, n_bins)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

// ─── Recurrence Analysis ────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (trajectory, t, d, epsilon, angular = false))]
fn recurrence_matrix_rust<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray1<'py, f64>,
    t: usize,
    d: usize,
    epsilon: f64,
    angular: bool,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let traj = trajectory
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = recurrence::recurrence_matrix(traj, t, d, epsilon, angular)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (traj_a, traj_b, t, d, epsilon, angular = false))]
fn cross_recurrence_matrix_rust<'py>(
    py: Python<'py>,
    traj_a: PyReadonlyArray1<'py, f64>,
    traj_b: PyReadonlyArray1<'py, f64>,
    t: usize,
    d: usize,
    epsilon: f64,
    angular: bool,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let a = traj_a
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = traj_b
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = recurrence::cross_recurrence_matrix(a, b, t, d, epsilon, angular)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Returns (rr, det, avg_diag, max_diag, ent_diag, lam, trap_time, max_vert).
#[pyfunction]
#[pyo3(signature = (recurrence_flat, t, l_min = 2, v_min = 2, exclude_main_diagonal = true))]
fn rqa_rust(
    recurrence_flat: PyReadonlyArray1<'_, u8>,
    t: usize,
    l_min: usize,
    v_min: usize,
    exclude_main_diagonal: bool,
) -> PyResult<(f64, f64, f64, usize, f64, f64, f64, usize)> {
    let r = recurrence_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = recurrence::rqa(r, t, l_min, v_min, exclude_main_diagonal)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok((
        result.recurrence_rate,
        result.determinism,
        result.avg_diagonal,
        result.max_diagonal,
        result.entropy_diagonal,
        result.laminarity,
        result.trapping_time,
        result.max_vertical,
    ))
}

// ─── SSGF Costs ───────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (w_flat, phases, n, w1 = 1.0, w2 = 0.5, w3 = 0.1, w4 = 0.1))]
fn compute_ssgf_costs_rust(
    w_flat: PyReadonlyArray1<'_, f64>,
    phases: PyReadonlyArray1<'_, f64>,
    n: usize,
    w1: f64,
    w2: f64,
    w3: f64,
    w4: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let w = w_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let r = ssgf_costs::compute_ssgf_costs(w, p, n, (w1, w2, w3, w4));
    Ok((r.c1_sync, r.c2_spectral_gap, r.c3_sparsity, r.c4_symmetry, r.u_total))
}

// ─── Swarmalator ──────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (pos_init, phases_init, omegas, n, dim, dt, a, b, j, k, n_steps))]
fn swarmalator_run_rust<'py>(
    py: Python<'py>,
    pos_init: PyReadonlyArray1<'py, f64>,
    phases_init: PyReadonlyArray1<'py, f64>,
    omegas: PyReadonlyArray1<'py, f64>,
    n: usize,
    dim: usize,
    dt: f64,
    a: f64,
    b: f64,
    j: f64,
    k: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let p = pos_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ph = phases_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (fp, fph, pt, pht) = swarmalator::swarmalator_run(p, ph, o, n, dim, dt, a, b, j, k, n_steps);
    Ok((
        PyArray1::from_vec(py, fp),
        PyArray1::from_vec(py, fph),
        PyArray1::from_vec(py, pt),
        PyArray1::from_vec(py, pht),
    ))
}

// ─── Delayed Kuramoto ─────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    phases_init, omegas, knm_flat, alpha_flat, n,
    zeta, psi, dt, delay_steps, n_steps
))]
fn delayed_kuramoto_run_rust<'py>(
    py: Python<'py>,
    phases_init: PyReadonlyArray1<'py, f64>,
    omegas: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    alpha_flat: PyReadonlyArray1<'py, f64>,
    n: usize,
    zeta: f64,
    psi: f64,
    dt: f64,
    delay_steps: usize,
    n_steps: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = delay::delayed_kuramoto_run(p, o, k, a, n, zeta, psi, dt, delay_steps, n_steps);
    Ok(PyArray1::from_vec(py, result))
}

// ─── Free Energy ──────────────────────────────────────────────────

#[pyfunction]
fn boltzmann_weight_rust(u_total: f64, temperature: f64) -> f64 {
    free_energy::boltzmann_weight(u_total, temperature)
}

#[pyfunction]
fn effective_temperature_rust(costs: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let c = costs.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(free_energy::effective_temperature(c))
}

#[pyfunction]
fn add_langevin_noise_rust<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    temperature: f64,
    dt: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let zv = z.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, free_energy::add_langevin_noise(zv, temperature, dt, seed)))
}

// ─── Hodge Decomposition ──────────────────────────────────────────

#[pyfunction]
fn hodge_decomposition_rust<'py>(
    py: Python<'py>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    phases: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (g, c, h) = hodge::hodge_decomposition(k, p, n);
    Ok((
        PyArray1::from_vec(py, g),
        PyArray1::from_vec(py, c),
        PyArray1::from_vec(py, h),
    ))
}

// ─── E/I Balance ──────────────────────────────────────────────────

#[pyfunction]
fn compute_ei_balance_rust(
    knm_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    excitatory_indices: PyReadonlyArray1<'_, i64>,
    inhibitory_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<(f64, f64, f64, bool)> {
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_raw = excitatory_indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let i_raw = inhibitory_indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_idx: Vec<usize> = e_raw.iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect();
    let i_idx: Vec<usize> = i_raw.iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect();
    let r = ei_balance::compute_ei_balance(k, n, &e_idx, &i_idx);
    Ok((r.ratio, r.excitatory_strength, r.inhibitory_strength, r.is_balanced))
}

#[pyfunction]
#[pyo3(signature = (knm_flat, n, excitatory_indices, inhibitory_indices, target_ratio = 1.0))]
fn adjust_ei_ratio_rust<'py>(
    py: Python<'py>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    n: usize,
    excitatory_indices: PyReadonlyArray1<'py, i64>,
    inhibitory_indices: PyReadonlyArray1<'py, i64>,
    target_ratio: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_raw = excitatory_indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let i_raw = inhibitory_indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let e_idx: Vec<usize> = e_raw.iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect();
    let i_idx: Vec<usize> = i_raw.iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect();
    let result = ei_balance::adjust_ei_ratio(k, n, &e_idx, &i_idx, target_ratio);
    Ok(PyArray1::from_vec(py, result))
}

// ─── Inertial Kuramoto (Swing Equation) ───────────────────────────

#[pyfunction]
#[pyo3(signature = (theta, omega_dot, power, knm_flat, inertia, damping, n, dt))]
fn inertial_step_rust<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega_dot: PyReadonlyArray1<'py, f64>,
    power: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    inertia: PyReadonlyArray1<'py, f64>,
    damping: PyReadonlyArray1<'py, f64>,
    n: usize,
    dt: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let th = theta.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let od = omega_dot.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pw = power.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let km = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let in_ = inertia.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dm = damping.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (new_th, new_od) = inertial::inertial_step(th, od, pw, km, in_, dm, n, dt);
    Ok((PyArray1::from_vec(py, new_th), PyArray1::from_vec(py, new_od)))
}

#[pyfunction]
#[pyo3(signature = (theta, omega_dot, power, knm_flat, inertia, damping, n, dt, n_steps))]
fn inertial_run_rust<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega_dot: PyReadonlyArray1<'py, f64>,
    power: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    inertia: PyReadonlyArray1<'py, f64>,
    damping: PyReadonlyArray1<'py, f64>,
    n: usize,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let th = theta.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let od = omega_dot.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pw = power.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let km = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let in_ = inertia.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dm = damping.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (f_th, f_od, t_th, t_od) =
        inertial::inertial_run(th, od, pw, km, in_, dm, n, dt, n_steps);
    Ok((
        PyArray1::from_vec(py, f_th),
        PyArray1::from_vec(py, f_od),
        PyArray1::from_vec(py, t_th),
        PyArray1::from_vec(py, t_od),
    ))
}

// ─── Market Synchronisation ───────────────────────────────────────

#[pyfunction]
fn market_order_parameter_rust<'py>(
    py: Python<'py>,
    phases_flat: PyReadonlyArray1<'py, f64>,
    t: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, market::market_order_parameter(p, t, n)))
}

#[pyfunction]
#[pyo3(signature = (phases_flat, t, n, window = 50))]
fn market_plv_rust<'py>(
    py: Python<'py>,
    phases_flat: PyReadonlyArray1<'py, f64>,
    t: usize,
    n: usize,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, market::market_plv(p, t, n, window)))
}

#[pyfunction]
#[pyo3(signature = (r, sync_threshold = 0.7, desync_threshold = 0.3))]
fn detect_regimes_rust<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    sync_threshold: f64,
    desync_threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let rv = r.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(
        py,
        market::detect_regimes(rv, sync_threshold, desync_threshold),
    ))
}

// ─── Basin Stability (Menck et al. 2013) ──────────────────────────

#[pyfunction]
#[pyo3(signature = (
    omegas, knm_flat, alpha_flat, n,
    dt, n_transient, n_measure, n_samples, r_threshold, seed
))]
fn basin_stability_rust<'py>(
    py: Python<'py>,
    omegas: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    alpha_flat: PyReadonlyArray1<'py, f64>,
    n: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    n_samples: usize,
    r_threshold: f64,
    seed: u64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, usize)> {
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = basin_stability::basin_stability(
        o, k, a, n, dt, n_transient, n_measure, n_samples, r_threshold, seed,
    );
    Ok((result.s_b, PyArray1::from_vec(py, result.r_finals), result.n_converged))
}

// ─── Bifurcation (Kuramoto 1975) ──────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    phases_init, omegas, knm_flat, alpha_flat, n,
    k_scale, dt, n_transient, n_measure
))]
fn steady_state_r_rust(
    phases_init: PyReadonlyArray1<'_, f64>,
    omegas: PyReadonlyArray1<'_, f64>,
    knm_flat: PyReadonlyArray1<'_, f64>,
    alpha_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    k_scale: f64,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> PyResult<f64> {
    let p = phases_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(bifurcation::steady_state_r(p, o, k, a, n, k_scale, dt, n_transient, n_measure))
}

#[pyfunction]
#[pyo3(signature = (
    omegas, knm_flat, alpha_flat, n, phases_init,
    k_min, k_max, n_points, dt, n_transient, n_measure
))]
fn trace_sync_transition_rust<'py>(
    py: Python<'py>,
    omegas: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    alpha_flat: PyReadonlyArray1<'py, f64>,
    n: usize,
    phases_init: PyReadonlyArray1<'py, f64>,
    k_min: f64,
    k_max: f64,
    n_points: usize,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (kv, rv, kc) = bifurcation::trace_sync_transition(
        o, k, a, n, p, k_min, k_max, n_points, dt, n_transient, n_measure,
    );
    Ok((PyArray1::from_vec(py, kv), PyArray1::from_vec(py, rv), kc))
}

#[pyfunction]
#[pyo3(signature = (
    omegas, knm_flat, alpha_flat, n, phases_init,
    dt, n_transient, n_measure, tol
))]
fn find_critical_coupling_bif_rust(
    omegas: PyReadonlyArray1<'_, f64>,
    knm_flat: PyReadonlyArray1<'_, f64>,
    alpha_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    phases_init: PyReadonlyArray1<'_, f64>,
    dt: f64,
    n_transient: usize,
    n_measure: usize,
    tol: f64,
) -> PyResult<f64> {
    let o = omegas.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let k = knm_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = alpha_flat.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = phases_init.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(bifurcation::find_critical_coupling(o, k, a, n, p, dt, n_transient, n_measure, tol))
}

// ─── Psychedelic Entropy ───────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (phases, n_bins = 36))]
fn entropy_from_phases_rust(
    phases: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<f64> {
    let p = phases
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(psychedelic::entropy_from_phases(p, n_bins))
}

#[pyfunction]
fn reduce_coupling_rust<'py>(
    py: Python<'py>,
    knm: PyReadonlyArray1<'py, f64>,
    reduction_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = knm
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, psychedelic::reduce_coupling(k, reduction_factor)))
}

// ─── Poincaré Section ──────────────────────────────────────────────

/// Returns (crossings_flat, crossing_times, n_crossings).
#[pyfunction]
#[pyo3(signature = (traj_flat, t, d, normal, offset = 0.0, direction = "positive"))]
fn poincare_section_rust<'py>(
    py: Python<'py>,
    traj_flat: PyReadonlyArray1<'py, f64>,
    t: usize,
    d: usize,
    normal: PyReadonlyArray1<'py, f64>,
    offset: f64,
    direction: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let tr = traj_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = normal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dir = match direction {
        "positive" => poincare::CrossingDirection::Positive,
        "negative" => poincare::CrossingDirection::Negative,
        "both" => poincare::CrossingDirection::Both,
        _ => return Err(PyValueError::new_err("direction must be 'positive', 'negative', or 'both'")),
    };
    let result = poincare::poincare_section(tr, t, d, n, offset, dir)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok((
        PyArray1::from_vec(py, result.crossings),
        PyArray1::from_vec(py, result.crossing_times),
        result.n_crossings,
    ))
}

#[pyfunction]
#[pyo3(signature = (phases_flat, t, n, oscillator_idx = 0, section_phase = 0.0))]
fn phase_poincare_rust<'py>(
    py: Python<'py>,
    phases_flat: PyReadonlyArray1<'py, f64>,
    t: usize,
    n: usize,
    oscillator_idx: usize,
    section_phase: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let p = phases_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = poincare::phase_poincare(p, t, n, oscillator_idx, section_phase)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok((
        PyArray1::from_vec(py, result.crossings),
        PyArray1::from_vec(py, result.crossing_times),
        result.n_crossings,
    ))
}

// ─── Embedding (Takens 1981) ───────────────────────────────────────

#[pyfunction]
fn delay_embed_rust<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    delay: usize,
    dimension: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = embedding::delay_embed(s, delay, dimension)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (signal, max_lag = 100, n_bins = 32))]
fn optimal_delay_rust(
    signal: PyReadonlyArray1<'_, f64>,
    max_lag: usize,
    n_bins: usize,
) -> PyResult<usize> {
    let s = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(embedding::optimal_delay(s, max_lag, n_bins))
}

#[pyfunction]
#[pyo3(signature = (signal, delay, max_dim = 10, rtol = 15.0, atol = 2.0))]
fn optimal_dimension_rust(
    signal: PyReadonlyArray1<'_, f64>,
    delay: usize,
    max_dim: usize,
    rtol: f64,
    atol: f64,
) -> PyResult<usize> {
    let s = signal
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(embedding::optimal_dimension(s, delay, max_dim, rtol, atol))
}

// ─── ITPC (Lachaux et al. 1999) ────────────────────────────────────

#[pyfunction]
fn compute_itpc_rust<'py>(
    py: Python<'py>,
    phases_flat: PyReadonlyArray1<'py, f64>,
    n_trials: usize,
    n_timepoints: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = phases_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = itpc::compute_itpc(p, n_trials, n_timepoints);
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn itpc_persistence_rust(
    phases_flat: PyReadonlyArray1<'_, f64>,
    n_trials: usize,
    n_timepoints: usize,
    pause_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<f64> {
    let p = phases_flat
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let idx_raw = pause_indices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Convert i64 indices to usize, filtering negatives
    let idx: Vec<usize> = idx_raw
        .iter()
        .filter(|&&v| v >= 0)
        .map(|&v| v as usize)
        .collect();
    Ok(itpc::itpc_persistence(p, n_trials, n_timepoints, &idx))
}

// ─── Module Registration ────────────────────────────────────────────

#[pymodule]
fn spo_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUPDEStepper>()?;
    m.add_class::<PySheafUPDEStepper>()?;
    m.add_class::<PyActiveInferenceAgent>()?;
    m.add_class::<PySparseUPDEStepper>()?;
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
    m.add_class::<PyPetriNet>()?;
    m.add_class::<PyRuleEngine>()?;
    m.add_class::<PyLIFEnsemble>()?;
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
    m.add_function(wrap_pyfunction!(pid_redundancy, m)?)?;
    m.add_function(wrap_pyfunction!(pid_synergy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_npe, m)?)?;
    m.add_function(wrap_pyfunction!(phase_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_production_rate, m)?)?;
    m.add_function(wrap_pyfunction!(winding_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(lyapunov_spectrum_rust, m)?)?;
    m.add_function(wrap_pyfunction!(recurrence_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cross_recurrence_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rqa_rust, m)?)?;
    m.add_function(wrap_pyfunction!(phase_transfer_entropy_rust, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_entropy_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fiedler_value_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fiedler_vector_rust, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_gap_rust, m)?)?;
    m.add_function(wrap_pyfunction!(critical_coupling_rust, m)?)?;
    m.add_function(wrap_pyfunction!(sync_convergence_rate_rust, m)?)?;
    m.add_function(wrap_pyfunction!(detect_chimera_rust, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_integral_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kaplan_yorke_dimension_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_itpc_rust, m)?)?;
    m.add_function(wrap_pyfunction!(itpc_persistence_rust, m)?)?;
    m.add_function(wrap_pyfunction!(delay_embed_rust, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_delay_rust, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_dimension_rust, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_section_rust, m)?)?;
    m.add_function(wrap_pyfunction!(phase_poincare_rust, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_from_phases_rust, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_coupling_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ssgf_costs_rust, m)?)?;
    m.add_function(wrap_pyfunction!(swarmalator_run_rust, m)?)?;
    m.add_function(wrap_pyfunction!(delayed_kuramoto_run_rust, m)?)?;
    m.add_function(wrap_pyfunction!(boltzmann_weight_rust, m)?)?;
    m.add_function(wrap_pyfunction!(effective_temperature_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add_langevin_noise_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hodge_decomposition_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ei_balance_rust, m)?)?;
    m.add_function(wrap_pyfunction!(adjust_ei_ratio_rust, m)?)?;
    m.add_function(wrap_pyfunction!(inertial_step_rust, m)?)?;
    m.add_function(wrap_pyfunction!(inertial_run_rust, m)?)?;
    m.add_function(wrap_pyfunction!(market_order_parameter_rust, m)?)?;
    m.add_function(wrap_pyfunction!(market_plv_rust, m)?)?;
    m.add_function(wrap_pyfunction!(detect_regimes_rust, m)?)?;
    m.add_function(wrap_pyfunction!(basin_stability_rust, m)?)?;
    m.add_function(wrap_pyfunction!(steady_state_r_rust, m)?)?;
    m.add_function(wrap_pyfunction!(trace_sync_transition_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_critical_coupling_bif_rust, m)?)?;
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
