# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Web dashboard server

"""FastAPI server for real-time simulation monitoring.

Usage:
    spo serve binding_spec.yaml --port 8080
    # Then open http://localhost:8080 in a browser.

Endpoints:
    GET  /             HTML dashboard
    GET  /api/state    Current UPDE state (JSON)
    POST /api/step     Advance one step
    POST /api/reset    Reset simulation
    GET  /api/config   Binding spec summary
    WS   /ws/stream    Real-time WebSocket stream
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

__all__ = ["create_app", "SimulationState"]

TWO_PI = 2.0 * np.pi


class SimulationState:
    """Mutable simulation state shared across API endpoints."""

    def __init__(self, spec: BindingSpec) -> None:
        self.spec = spec
        self.n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        self.coupling = CouplingBuilder().build(
            self.n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        self.omegas = np.array(
            [
                1.0 + 0.1 * layer.index
                for layer in spec.layers
                for _ in layer.oscillator_ids
            ]
        )
        self.phases = extract_initial_phases(spec, self.omegas)
        self.engine = UPDEEngine(self.n_osc, dt=spec.sample_period_s)
        self.regime_manager = RegimeManager()
        self.step_count = 0
        self.amplitude_mode = spec.amplitude is not None
        self.sl_engine: StuartLandauEngine | None = None
        self.sl_state: np.ndarray | None = None
        self.mu: np.ndarray | None = None

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
                self.coupling.knm,
                self.coupling.knm_r,
                0.0,
                0.0,
                self.coupling.alpha,
                epsilon=self.spec.amplitude.epsilon,
            )
            self.phases = self.sl_state[: self.n_osc]
        else:
            self.phases = self.engine.step(
                self.phases,
                self.omegas,
                self.coupling.knm,
                0.0,
                0.0,
                self.coupling.alpha,
            )
        self.step_count += 1
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
        self.regime_manager = RegimeManager()
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


def create_app(spec_path: str | Path):  # type: ignore[no-untyped-def]
    """Create FastAPI app for the given binding spec."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        msg = "fastapi not installed. pip install fastapi uvicorn"
        raise ImportError(msg) from exc

    spec = load_binding_spec(spec_path)
    sim = SimulationState(spec)
    app = FastAPI(title="SPO Dashboard", version="0.4.1")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/api/state")
    async def get_state() -> dict:
        return sim.snapshot()

    @app.post("/api/step")
    async def post_step() -> dict:
        return sim.step()

    @app.post("/api/reset")
    async def post_reset() -> dict:
        return sim.reset()

    @app.get("/api/config")
    async def get_config() -> dict:
        return {
            "name": spec.name,
            "n_oscillators": sim.n_osc,
            "n_layers": len(spec.layers),
            "amplitude_mode": sim.amplitude_mode,
            "sample_period_s": spec.sample_period_s,
            "control_period_s": spec.control_period_s,
        }

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                state = sim.step()
                await websocket.send_text(json.dumps(state))
                await asyncio.sleep(spec.sample_period_s)
        except WebSocketDisconnect:
            pass

    return app
