# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC streaming service tests

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.grpc_gen.spo_pb2 import StateResponse, StreamRequest
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.server_grpc import PhaseStreamServicer

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


class _FakeContext:
    def __init__(self, active: bool = True) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active


def _make_servicer():
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    return PhaseStreamServicer(sim)


def test_stream_yields_responses():
    svc = _make_servicer()
    req = StreamRequest(max_steps=5, interval_s=0.0)
    results = list(svc.StreamPhases(req, _FakeContext()))
    assert len(results) == 5
    for resp in results:
        assert isinstance(resp, StateResponse)
        assert isinstance(resp.R_global, float)
        assert resp.step > 0
        assert resp.regime != ""


def test_stream_stops_on_inactive_context():
    svc = _make_servicer()
    ctx = MagicMock()
    ctx.is_active.side_effect = [True, True, False]
    req = StreamRequest(max_steps=100, interval_s=0.0)
    results = list(svc.StreamPhases(req, ctx))
    assert len(results) == 2


def test_stream_has_layer_data():
    svc = _make_servicer()
    req = StreamRequest(max_steps=3, interval_s=0.0)
    results = list(svc.StreamPhases(req, _FakeContext()))
    assert len(results) == 3
    assert len(results[0].layers) > 0
    assert results[0].layers[0].name != ""


def test_stream_with_none_context():
    svc = _make_servicer()
    req = StreamRequest(max_steps=2, interval_s=0.0)
    results = list(svc.StreamPhases(req, None))
    assert len(results) == 2
