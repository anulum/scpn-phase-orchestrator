# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Web server tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


class TestSimulationState:
    def test_init_cardiac(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "cardiac_rhythm" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        assert sim.n_osc == 10
        assert sim.amplitude_mode is True
        assert sim.step_count == 0

    def test_step_advances(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        state = sim.step()
        assert state["step"] == 1
        assert "R_global" in state
        assert "layers" in state

    def test_multiple_steps(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        for _ in range(50):
            state = sim.step()
        assert state["step"] == 50

    def test_reset(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        sim.step()
        sim.step()
        state = sim.reset()
        assert state["step"] == 0

    def test_snapshot_format(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "neuroscience_eeg" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        snap = sim.snapshot()
        assert isinstance(snap["R_global"], float)
        assert isinstance(snap["layers"], list)
        assert len(snap["layers"]) == 6
        for layer in snap["layers"]:
            assert "name" in layer
            assert "R" in layer

    def test_amplitude_mode_snapshot(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "cardiac_rhythm" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        sim.step()
        snap = sim.snapshot()
        assert snap["amplitude_mode"] is True
        assert "mean_amplitude" in snap

    def test_phase_only_no_amplitude(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        snap = sim.snapshot()
        assert isinstance(snap["amplitude_mode"], bool)

    def test_geometry_constraints_applied(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "neuroscience_eeg" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        assert len(sim.geo_constraints) == 2
        sim.step()
        assert sim.step_count == 1

    def test_geometry_constraints_absent_when_no_prior(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        assert len(sim.geo_constraints) == 0

    def test_boundary_observer_wired_to_regime(self):
        spec = load_binding_spec(
            DOMAINPACK_DIR / "neuroscience_eeg" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        assert sim.boundary_observer is not None
        assert sim.event_bus is not None
        assert sim.boundary_observer._event_bus is sim.event_bus
        for _ in range(5):
            sim.step()
        assert sim.step_count == 5


try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestFastAPIEndpoints:
    @pytest.fixture()
    def client(self):
        from scpn_phase_orchestrator.server import create_app

        app = create_app(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
        return TestClient(app)

    def test_dashboard(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SPO Dashboard" in r.text

    def test_get_state(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert "R_global" in data

    def test_post_step(self, client):
        r = client.post("/api/step")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 1

    def test_post_reset(self, client):
        client.post("/api/step")
        r = client.post("/api/reset")
        assert r.status_code == 200
        assert r.json()["step"] == 0

    def test_get_config(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert "n_oscillators" in data
