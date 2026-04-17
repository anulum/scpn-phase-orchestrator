# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC streaming service tests

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.grpc_gen import StateResponse, StreamRequest
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


def test_servicer_reuses_simulation_state_lock():
    """T6 regression: gRPC servicer must share SimulationState._lock so HTTP
    and gRPC paths serialise against the same mutex. A separate lock leaves
    shared engine state vulnerable to concurrent mutation.
    """
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    servicer = PhaseStreamServicer(sim)
    assert servicer._lock is sim._lock


def test_simulation_state_lock_is_thread_safe():
    """T6 regression: SimulationState._lock must be a threading primitive
    (not asyncio.Lock) so it can be held from gRPC thread pools.
    """
    import threading

    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    # Must be acquirable from a plain thread without an event loop.
    acquired = threading.Event()

    def worker() -> None:
        with sim._lock:
            acquired.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2.0)
    assert acquired.is_set(), "lock not acquirable from worker thread"


# Pipeline wiring: StreamPhases streams SimulationState.step() which
# drives UPDEEngine internally. Tests above verify R_global and layers.
