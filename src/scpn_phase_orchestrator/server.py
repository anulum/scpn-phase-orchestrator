# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Web dashboard server

"""FastAPI server for real-time simulation monitoring.

Usage:
    spo serve binding_spec.yaml --port 8080
    # Then open http://localhost:8080 in a browser.

    # With API key authentication:
    SPO_API_KEY=mysecret spo serve binding_spec.yaml --port 8080
    # Mutable endpoints (step, reset) require X-API-Key header.
    # When SPO_API_KEY is unset, mutable endpoints are unrestricted
    # (development mode — do NOT deploy without setting the key).

Endpoints:
    GET  /             HTML dashboard
    GET  /api/state    Current UPDE state (JSON)
    POST /api/step     Advance one step (auth required if SPO_API_KEY set)
    POST /api/reset    Reset simulation (auth required if SPO_API_KEY set)
    GET  /api/config   Binding spec summary
    WS   /ws/stream    Real-time WebSocket stream (read-only observer)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    GeometryConstraint,
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

try:
    from fastapi import (  # pragma: no cover
        Response,
        WebSocket,
        WebSocketDisconnect,  # pragma: no cover
    )
except ImportError:  # pragma: no cover
    Response = None  # type: ignore[assignment,misc]  # pragma: no cover
    WebSocket = None  # type: ignore[assignment,misc]  # pragma: no cover
    WebSocketDisconnect = None  # type: ignore[assignment,misc]  # pragma: no cover

__all__ = ["create_app", "SimulationState"]

TWO_PI = 2.0 * np.pi


class SimulationState:
    """Mutable simulation state shared across API endpoints."""

    def __init__(self, spec: BindingSpec) -> None:
        self.spec = spec
        # threading.Lock so FastAPI async handlers and the gRPC servicer
        # (thread-pool worker) serialise against the *same* mutex. An
        # asyncio.Lock would only protect the event loop and leave gRPC
        # threads free to race on the shared engine state.
        self._lock = threading.Lock()
        self.n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        self.coupling = CouplingBuilder().build(
            self.n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        self.omegas = np.array(spec.get_omegas(), dtype=np.float64)
        self.phases = extract_initial_phases(spec, self.omegas)
        self.engine = UPDEEngine(self.n_osc, dt=spec.sample_period_s)
        self.event_bus = EventBus()
        self.boundary_observer = BoundaryObserver(spec.boundaries)
        self.boundary_observer.set_event_bus(self.event_bus)
        self.regime_manager = RegimeManager(event_bus=self.event_bus)
        self.step_count = 0
        self.amplitude_mode = spec.amplitude is not None
        self.sl_engine: StuartLandauEngine | None = None
        self.sl_state: np.ndarray | None = None
        self.mu: np.ndarray | None = None

        self.imprint_model: ImprintModel | None = None
        self.imprint_state: ImprintState | None = None
        if spec.imprint_model is not None:
            self.imprint_model = ImprintModel(
                spec.imprint_model.decay_rate, spec.imprint_model.saturation
            )
            self.imprint_state = ImprintState(m_k=np.zeros(self.n_osc), last_update=0.0)

        self.geo_constraints: list[GeometryConstraint] = []
        if spec.geometry_prior is not None:
            ct = spec.geometry_prior.constraint_type.lower()
            if "symmetric" in ct:
                self.geo_constraints.append(SymmetryConstraint())
            if "non_negative" in ct or "nonneg" in ct:
                self.geo_constraints.append(NonNegativeConstraint())

        if self.amplitude_mode and spec.amplitude is not None:
            amp = spec.amplitude
            self.sl_engine = StuartLandauEngine(
                self.n_osc,
                dt=spec.sample_period_s,
            )
            self.mu = np.full(self.n_osc, amp.mu)
            self.coupling = CouplingBuilder().build_with_amplitude(
                self.n_osc,
                spec.coupling.base_strength,
                spec.coupling.decay_alpha,
                amp.amp_coupling_strength,
                amp.amp_coupling_decay,
            )
            self.sl_state = np.concatenate(
                [
                    self.phases,
                    np.sqrt(np.maximum(self.mu, 0.0)),
                ]
            )

    def step(self) -> dict:
        """Advance one timestep, return state snapshot."""
        eff_knm = self.coupling.knm
        eff_alpha = self.coupling.alpha
        if self.imprint_model is not None and self.imprint_state is not None:
            eff_knm = self.imprint_model.modulate_coupling(eff_knm, self.imprint_state)
            eff_alpha = self.imprint_model.modulate_lag(eff_alpha, self.imprint_state)
        if self.geo_constraints:
            eff_knm = project_knm(eff_knm, self.geo_constraints)

        if (
            self.amplitude_mode
            and self.sl_engine is not None
            and self.mu is not None
            and self.sl_state is not None
            and self.coupling.knm_r is not None
            and self.spec.amplitude is not None
        ):
            self.sl_state = self.sl_engine.step(
                self.sl_state,
                self.omegas,
                self.mu,
                eff_knm,
                self.coupling.knm_r,
                0.0,
                0.0,
                eff_alpha,
                epsilon=self.spec.amplitude.epsilon,
            )
            self.phases = self.sl_state[: self.n_osc]
        else:
            self.phases = self.engine.step(
                self.phases,
                self.omegas,
                eff_knm,
                0.0,
                0.0,
                eff_alpha,
            )
        self.step_count += 1

        layer_states = []
        idx = 0
        for layer in self.spec.layers:
            n = len(layer.oscillator_ids)
            r, psi = compute_order_parameter(self.phases[idx : idx + n])
            layer_states.append(LayerState(R=r, psi=psi))
            idx += n
        if self.imprint_model is not None and self.imprint_state is not None:
            exposure = np.array(
                [
                    layer_states[i].R
                    for i, layer in enumerate(self.spec.layers)
                    for _ in layer.oscillator_ids
                ]
            )
            self.imprint_state = self.imprint_model.update(
                self.imprint_state, exposure, self.spec.sample_period_s
            )

        r_global, _ = compute_order_parameter(self.phases)
        upde_state = UPDEState(
            layers=layer_states,
            cross_layer_alignment=np.eye(len(self.spec.layers)),
            stability_proxy=r_global,
            regime_id=self.regime_manager.current_regime.value,
        )
        obs_values: dict[str, float] = {"R": r_global}
        for i, ls in enumerate(layer_states):
            obs_values[f"R_{i}"] = ls.R
        boundary_state = self.boundary_observer.observe(
            obs_values, step=self.step_count
        )
        proposed = self.regime_manager.evaluate(upde_state, boundary_state)
        self.regime_manager.transition(proposed)

        return self.snapshot()

    def snapshot(self) -> dict:
        """Current state as JSON-serializable dict."""
        layer_map = {}
        idx = 0
        for layer in self.spec.layers:
            n = len(layer.oscillator_ids)
            layer_map[layer.name] = list(range(idx, idx + n))
            idx += n

        layers = []
        for layer in self.spec.layers:
            ids = layer_map[layer.name]
            r, psi = compute_order_parameter(self.phases[ids])
            layers.append(
                {
                    "name": layer.name,
                    "R": round(float(r), 4),
                    "psi": round(float(psi), 4),
                }
            )

        r_global, _ = compute_order_parameter(self.phases)
        result = {
            "step": self.step_count,
            "R_global": round(float(r_global), 4),
            "regime": self.regime_manager.current_regime.value,
            "layers": layers,
            "n_oscillators": self.n_osc,
            "amplitude_mode": self.amplitude_mode,
        }

        if self.amplitude_mode and self.sl_state is not None:
            amps = self.sl_state[self.n_osc :]
            result["mean_amplitude"] = round(float(np.mean(amps)), 4)

        return result

    def reset(self) -> dict:
        """Reset to initial state."""
        self.phases = extract_initial_phases(self.spec, self.omegas)
        if self.amplitude_mode and self.mu is not None:
            self.sl_state = np.concatenate(
                [
                    self.phases,
                    np.sqrt(np.maximum(self.mu, 0.0)),
                ]
            )
        self.step_count = 0
        self.regime_manager = RegimeManager(event_bus=self.event_bus)
        if self.imprint_model is not None:
            self.imprint_state = ImprintState(m_k=np.zeros(self.n_osc), last_update=0.0)
        return self.snapshot()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SPO Dashboard</title>
<style>
body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 2em; }
h1 { color: #7b2ff7; }
.stat { display: inline-block; margin: 0.5em 1em; padding: 0.5em 1em;
  background: #16213e; border-radius: 6px; min-width: 120px; text-align: center; }
.stat .value { font-size: 2em; color: #7b2ff7; }
.stat .label { font-size: 0.8em; color: #888; }
.layers { margin-top: 1em; }
.layer-bar { margin: 4px 0; }
.bar { display: inline-block; height: 16px; background: #7b2ff7; border-radius: 3px; }
#controls { margin: 1em 0; }
button { background: #7b2ff7; color: white; border: none; padding: 8px 16px;
  border-radius: 4px; cursor: pointer; margin-right: 8px; font-family: monospace; }
button:hover { background: #5a1fb0; }
</style>
</head>
<body>
<h1>SCPN Phase Orchestrator</h1>
<div id="stats"></div>
<div id="controls">
  <button onclick="step()">Step</button>
  <button onclick="run()">Run 100</button>
  <button onclick="reset()">Reset</button>
  <button onclick="toggleStream()">Stream</button>
</div>
<div id="layers" class="layers"></div>
<script>
let streaming = false, ws = null;

async function fetchState() {
  const r = await fetch('/api/state');
  return r.json();
}

function render(s) {
  document.getElementById('stats').innerHTML = [
    stat('Step', s.step),
    stat('R_global', s.R_global),
    stat('Regime', s.regime),
    s.amplitude_mode ? stat('Amplitude', s.mean_amplitude || '-') : '',
  ].join('');
  let h = '';
  s.layers.forEach(function(l) {
    let w = Math.round(l.R * 300);
    h += '<div class="layer-bar">';
    h += '<span style="width:80px;display:inline-block">';
    h += l.name + '</span>';
    h += '<span class="bar" style="width:' + w + 'px">';
    h += '</span> ' + l.R + '</div>';
  });
  document.getElementById('layers').innerHTML = h;
}

function stat(label, value) {
  return '<div class="stat">'
    + '<div class="value">' + value + '</div>'
    + '<div class="label">' + label + '</div></div>';
}

async function step() {
  let r = await fetch('/api/step', {method:'POST'});
  render(await r.json());
}
async function run() {
  for(let i = 0; i < 100; i++) await step();
}
async function reset() {
  let r = await fetch('/api/reset', {method:'POST'});
  render(await r.json());
}

function toggleStream() {
  if (streaming && ws) { ws.close(); streaming = false; return; }
  ws = new WebSocket(`ws://${location.host}/ws/stream`);
  ws.onmessage = e => render(JSON.parse(e.data));
  streaming = true;
}

fetchState().then(render);
</script>
</body>
</html>"""


def create_app(spec_path: str | Path) -> object:  # pragma: no cover
    """Create FastAPI app for the given binding spec."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        msg = "fastapi not installed. pip install fastapi uvicorn"
        raise ImportError(msg) from exc

    spec = load_binding_spec(spec_path)
    sim = SimulationState(spec)

    @asynccontextmanager
    async def _lifespan(_app: "FastAPI"):
        """Release engine resources when the process shuts down.

        The simulation state itself is held in-process (numpy arrays), but
        future integrations (gRPC channels, database handles, external
        adapters) register cleanup here so a graceful shutdown never
        leaks a descriptor.
        """
        logger.info(
            "spo server startup: spec=%s n_osc=%d amplitude_mode=%s",
            spec.name,
            sim.n_osc,
            sim.amplitude_mode,
        )
        try:
            yield
        finally:
            logger.info("spo server shutdown: spec=%s", spec.name)
            with sim._lock:
                sim.event_bus.clear()

    app = FastAPI(title="SPO Dashboard", version="0.5.0", lifespan=_lifespan)

    _api_key = os.environ.get("SPO_API_KEY")

    async def _require_auth(x_api_key: str | None = Header(None)) -> None:
        if _api_key is None:
            return
        if x_api_key != _api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        """Handle GET / — serve the HTML dashboard."""
        return DASHBOARD_HTML

    @app.get("/api/state")
    async def get_state() -> dict:
        """Handle GET /api/state — return current simulation snapshot."""
        with sim._lock:
            return sim.snapshot()

    @app.post("/api/step", dependencies=[Depends(_require_auth)])
    async def post_step() -> dict:
        """Handle POST /api/step — advance simulation one tick."""
        with sim._lock:
            snap = sim.step()
        logger.debug(
            "api.step: step=%d R_global=%.4f regime=%s",
            snap.get("step", -1),
            snap.get("R_global", float("nan")),
            snap.get("regime", ""),
        )
        return snap

    @app.post("/api/reset", dependencies=[Depends(_require_auth)])
    async def post_reset() -> dict:
        """Handle POST /api/reset — reset simulation to initial state."""
        with sim._lock:
            snap = sim.reset()
        logger.info("api.reset: step=%d regime=%s", snap.get("step", 0), snap.get("regime", ""))
        return snap

    @app.get("/api/config")
    async def get_config() -> dict:
        """Handle GET /api/config — return engine configuration."""
        return {
            "name": spec.name,
            "n_oscillators": sim.n_osc,
            "n_layers": len(spec.layers),
            "amplitude_mode": sim.amplitude_mode,
            "sample_period_s": spec.sample_period_s,
            "control_period_s": spec.control_period_s,
        }

    @app.get("/api/metrics")
    async def get_metrics() -> Response:  # pragma: no cover
        """Handle GET /api/metrics — export Prometheus-format metrics."""
        from fastapi.responses import PlainTextResponse

        from scpn_phase_orchestrator.adapters.metrics_exporter import (
            MetricsExporter,
        )

        with sim._lock:
            snap = sim.snapshot()
        exporter = MetricsExporter()
        upde_state = UPDEState(
            layers=[
                LayerState(R=ly["R"], psi=ly.get("psi", 0.0)) for ly in snap["layers"]
            ],
            cross_layer_alignment=np.eye(len(snap["layers"])),
            stability_proxy=snap["R_global"],
            regime_id=snap["regime"],
        )
        text = exporter.export(upde_state, snap["regime"], 0.0)
        return PlainTextResponse(text, media_type="text/plain")

    @app.get("/api/health")
    async def health() -> dict:
        """Deep health check — verifies engine, monitor, and regime subsystems."""
        checks: dict[str, str] = {}
        try:
            with sim._lock:
                snap = sim.snapshot()
            checks["engine"] = "ok" if snap.get("step", -1) >= 0 else "degraded"
            r_val = snap.get("R_global", float("nan"))
            checks["R_finite"] = "ok" if np.isfinite(r_val) else "error"
            checks["regime"] = "ok" if snap.get("regime") else "unknown"
        except Exception as exc:
            checks["engine"] = f"error: {exc}"

        healthy = all(v == "ok" for v in checks.values())
        return {"status": "healthy" if healthy else "degraded", "checks": checks}

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        """Read-only observer: streams snapshots without advancing simulation."""
        await websocket.accept()
        try:
            while True:
                with sim._lock:
                    state = sim.snapshot()
                await websocket.send_text(json.dumps(state))
                await asyncio.sleep(spec.sample_period_s)
        except WebSocketDisconnect:
            pass

    return app
