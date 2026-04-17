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


def _make_state(regime_id: str, stability: float = 0.5) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=stability, psi=0.0) for _ in range(3)],
        cross_layer_alignment=np.eye(3),
        stability_proxy=stability,
        regime_id=regime_id,
    )


class TestSupervisorPolicyLogging:
    def test_decide_logs_regime_and_action_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = SupervisorPolicy(RegimeManager())
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.supervisor.policy")
        state = _make_state("nominal", stability=0.9)
        boundary = BoundaryState(violations=[])

        actions = policy.decide(state, boundary)

        records = [
            r for r in caplog.records if r.name.endswith("supervisor.policy")
        ]
        assert records, "supervisor.decide did not emit a log record"
        rec = records[-1]
        assert "regime=" in rec.getMessage()
        assert rec.n_actions == len(actions)
        assert rec.regime in {r.value for r in Regime}
        assert "n_violations" in rec.__dict__

    def test_decide_extra_includes_knob_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Structured extras must include the knob list so downstream log
        aggregators can index decisions by knob.
        """
        policy = SupervisorPolicy(RegimeManager())
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.supervisor.policy")
        state = _make_state("degraded", stability=0.2)
        # Force DEGRADED: we stub the regime manager to emit degraded.
        policy._regime_manager.evaluate = lambda *_a, **_k: Regime.DEGRADED  # type: ignore[assignment]
        policy._regime_manager.transition = lambda r: r  # type: ignore[assignment]

        actions = policy.decide(state, BoundaryState(violations=[]))
        assert all(isinstance(a, ControlAction) for a in actions)

        records = [
            r for r in caplog.records if r.name.endswith("supervisor.policy")
        ]
        assert records
        rec = records[-1]
        assert rec.knobs == [a.knob for a in actions]


class TestServerGrpcLogging:
    def test_reset_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec = load_binding_spec(
            DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        servicer = PhaseStreamServicer(sim)
        caplog.set_level(logging.INFO, logger="scpn_phase_orchestrator.server_grpc")

        servicer.Reset(request=None, context=None)

        records = [r for r in caplog.records if r.name.endswith("server_grpc")]
        assert any("grpc.Reset" in r.getMessage() for r in records), (
            "Reset RPC did not emit an INFO-level structured log"
        )


# Pipeline wiring: observability is the only tool an operator has when a
# deployment misbehaves in production. These tests make sure at least the
# lifecycle / decision events reach the logging subsystem with structured
# extras — a silent supervisor is a debugging dead-end.
