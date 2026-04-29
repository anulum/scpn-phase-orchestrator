# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — structured-logging regression tests

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.server_grpc import PhaseStreamServicer
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


def _extra(record: logging.LogRecord, key: str) -> Any:
    return record.__dict__[key]


def _make_state(regime_id: str, stability: float = 0.5) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=stability, psi=0.0) for _ in range(3)],
        cross_layer_alignment=np.eye(3),
        stability_proxy=stability,
        regime_id=regime_id,
    )


class _DegradedRegimeManager(RegimeManager):
    def evaluate(self, *_args: Any, **_kwargs: Any) -> Regime:
        return Regime.DEGRADED

    def transition(self, proposed: Regime) -> Regime:
        return proposed


class TestSupervisorPolicyLogging:
    def test_decide_logs_regime_and_action_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = SupervisorPolicy(RegimeManager())
        caplog.set_level(
            logging.INFO,
            logger="scpn_phase_orchestrator.supervisor.policy",
        )
        state = _make_state("nominal", stability=0.9)
        boundary = BoundaryState(violations=[])

        actions = policy.decide(state, boundary)

        records = [r for r in caplog.records if r.name.endswith("supervisor.policy")]
        assert records, "supervisor.decide did not emit a log record"
        rec = records[-1]
        assert "regime=" in rec.getMessage()
        assert _extra(rec, "n_actions") == len(actions)
        assert _extra(rec, "regime") in {r.value for r in Regime}
        assert "n_violations" in rec.__dict__

    def test_decide_extra_includes_knob_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Structured extras must include the knob list so downstream log
        aggregators can index decisions by knob.
        """
        policy = SupervisorPolicy(RegimeManager())
        caplog.set_level(
            logging.INFO,
            logger="scpn_phase_orchestrator.supervisor.policy",
        )
        state = _make_state("degraded", stability=0.2)
        policy._regime_manager = _DegradedRegimeManager()

        actions = policy.decide(state, BoundaryState(violations=[]))
        assert all(isinstance(a, ControlAction) for a in actions)

        records = [r for r in caplog.records if r.name.endswith("supervisor.policy")]
        assert records
        rec = records[-1]
        assert _extra(rec, "knobs") == [a.knob for a in actions]


class TestServerGrpcLogging:
    def test_get_state_emits_structured_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        servicer = PhaseStreamServicer(sim)
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.server_grpc")

        response = servicer.GetState(request=None, context=None)

        records = [
            r
            for r in caplog.records
            if r.name.endswith("server_grpc") and getattr(r, "rpc", "") == "GetState"
        ]
        assert records, "GetState RPC did not emit a structured INFO log"
        rec = records[-1]
        assert _extra(rec, "step") == response.step
        assert _extra(rec, "regime") == response.regime
        assert _extra(rec, "status") == "ok"

    def test_reset_emits_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        servicer = PhaseStreamServicer(sim)
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.server_grpc")

        servicer.Reset(request=None, context=None)

        records = [r for r in caplog.records if r.name.endswith("server_grpc")]
        reset_records = [r for r in records if getattr(r, "rpc", "") == "Reset"]
        assert reset_records, "Reset RPC did not emit an INFO-level structured log"
        rec = reset_records[-1]
        assert "grpc.Reset" in rec.getMessage()
        assert _extra(rec, "status") == "ok"

    def test_step_log_includes_request_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        servicer = PhaseStreamServicer(sim)
        request = type("StepRequest", (), {"n_steps": 3})()
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.server_grpc")

        response = servicer.Step(request=request, context=None)

        records = [
            r
            for r in caplog.records
            if r.name.endswith("server_grpc") and getattr(r, "rpc", "") == "Step"
        ]
        assert records
        rec = records[-1]
        assert _extra(rec, "n_steps") == 3
        assert _extra(rec, "step") == response.step
        assert response.step == 3


class TestServerHttpLogging:
    def test_http_middleware_logs_path_without_query(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        testclient = pytest.importorskip("fastapi.testclient")
        from scpn_phase_orchestrator.server import create_app

        app = create_app(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
        client = testclient.TestClient(app)
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.server")

        response = client.get("/api/config?token=sensitive-value")

        assert response.status_code == 200
        records = [
            r
            for r in caplog.records
            if r.name.endswith("server")
            and getattr(r, "http_path", "") == "/api/config"
        ]
        assert records, "HTTP request middleware did not emit a structured log"
        rec = records[-1]
        assert _extra(rec, "http_method") == "GET"
        assert _extra(rec, "status_code") == 200
        assert _extra(rec, "duration_ms") >= 0.0
        assert "sensitive-value" not in rec.getMessage()
        assert "token" not in rec.getMessage()


# Pipeline wiring: observability is the only tool an operator has when a
# deployment misbehaves in production. These tests make sure at least the
# lifecycle / decision events reach the logging subsystem with structured
# extras — a silent supervisor is a debugging dead-end.
