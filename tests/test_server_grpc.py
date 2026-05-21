# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC streaming service tests

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.runtime.grpc_gen import StateResponse, StreamRequest
from scpn_phase_orchestrator.runtime.server import SimulationState
from scpn_phase_orchestrator.runtime.server_grpc import HAS_GRPC, PhaseStreamServicer

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


class _FakeContext:
    def __init__(
        self,
        active: bool = True,
        metadata: tuple[tuple[Any, Any], ...] = (),
    ) -> None:
        self._active = active
        self._metadata = metadata

    def is_active(self) -> bool:
        return self._active

    def invocation_metadata(self) -> tuple[tuple[Any, Any], ...]:
        return self._metadata

    def abort(self, _code, detail: str) -> None:
        raise PermissionError(detail)


class _RecordingContext(_FakeContext):
    def __init__(
        self,
        active: bool = True,
        metadata: tuple[tuple[Any, Any], ...] = (),
    ) -> None:
        super().__init__(active=active, metadata=metadata)
        self.abort_calls: list[tuple[Any, str]] = []

    def abort(self, code: Any, detail: str) -> None:
        self.abort_calls.append((code, detail))
        raise PermissionError(detail)


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


def test_stream_rejects_invalid_max_steps():
    svc = _make_servicer()
    req = StreamRequest(max_steps=-1, interval_s=0.0)
    with pytest.raises(PermissionError, match="max_steps"):
        list(svc.StreamPhases(req, _FakeContext()))


def test_stream_rejects_zero_max_steps():
    svc = _make_servicer()
    req = StreamRequest(max_steps=0, interval_s=0.0)
    with pytest.raises(PermissionError, match="max_steps"):
        list(svc.StreamPhases(req, _FakeContext()))


def test_stream_rejects_invalid_interval():
    svc = _make_servicer()
    req = StreamRequest(max_steps=1, interval_s=float("nan"))
    with pytest.raises(PermissionError, match="interval_s"):
        list(svc.StreamPhases(req, _FakeContext()))


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


def test_grpc_production_requires_api_key_env(monkeypatch):
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    monkeypatch.setenv("SPO_GRPC_ENV", "production")
    monkeypatch.delenv("SPO_GRPC_API_KEY", raising=False)
    monkeypatch.delenv("SPO_API_KEY", raising=False)

    try:
        with pytest.raises(RuntimeError, match="SPO_GRPC_API_KEY"):
            PhaseStreamServicer(sim)
    finally:
        monkeypatch.delenv("SPO_GRPC_ENV", raising=False)


def test_grpc_production_rejects_missing_metadata(monkeypatch):
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    monkeypatch.setenv("SPO_GRPC_ENV", "production")
    monkeypatch.setenv("SPO_GRPC_API_KEY", "test-key")
    svc = PhaseStreamServicer(sim)

    with pytest.raises(PermissionError, match="x-api-key"):
        svc.GetState(None, _FakeContext())


def test_grpc_production_rate_limits_metadata_identity(monkeypatch):
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    monkeypatch.setenv("SPO_GRPC_ENV", "production")
    monkeypatch.setenv("SPO_GRPC_API_KEY", "test-key")
    monkeypatch.setenv("SPO_GRPC_RATE_LIMIT_PER_MINUTE", "1")
    svc = PhaseStreamServicer(sim)
    ctx = _FakeContext(metadata=(("x-api-key", "test-key"),))
    ctx2 = _FakeContext(metadata=(("x-api-key", "other-key"),))

    assert isinstance(svc.GetState(None, ctx), StateResponse)
    with pytest.raises(PermissionError, match="Invalid or missing x-api-key"):
        svc.GetState(None, ctx2)
    with pytest.raises(PermissionError, match="Rate limit"):
        svc.GetState(None, ctx)


def test_grpc_production_accepts_metadata_variants(monkeypatch):
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    monkeypatch.setenv("SPO_GRPC_ENV", "production")
    monkeypatch.setenv("SPO_API_KEY", "test-key")

    svc = PhaseStreamServicer(sim)
    variants: list[tuple[tuple[Any, Any], ...]] = [
        (("X-API-KEY", "test-key"),),
        ((b"x-api-key", b"test-key"),),
    ]
    for metadata in variants:
        assert isinstance(
            svc.GetState(None, _FakeContext(metadata=metadata)), StateResponse
        )


def test_grpc_context_abort_recorded_code(monkeypatch):
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    monkeypatch.setenv("SPO_GRPC_ENV", "production")
    monkeypatch.setenv("SPO_GRPC_API_KEY", "test-key")

    svc = PhaseStreamServicer(sim)
    ctx = _RecordingContext(metadata=(("x-api-key", "wrong-key"),))
    with pytest.raises(PermissionError, match="Invalid or missing x-api-key"):
        svc.GetState(None, ctx)
    code, _detail = ctx.abort_calls[0]
    if HAS_GRPC:
        assert code is not None
    else:
        assert code is None


def test_step_rejects_negative_n_steps():
    svc = _make_servicer()
    req = type("StepRequest", (), {"n_steps": -1})()
    with pytest.raises(PermissionError, match="n_steps"):
        svc.Step(req, _FakeContext())


# Pipeline wiring: StreamPhases streams SimulationState.step() which
# drives UPDEEngine internally. Tests above verify R_global and layers.
