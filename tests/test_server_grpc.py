# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC streaming service tests

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from scpn_phase_orchestrator.server_grpc import PhaseStreamServicer


class _FakeContext:
    def __init__(self, active: bool = True) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active


def test_stream_yields_responses():
    counter = 0

    def source():
        nonlocal counter
        counter += 1
        return {"R": 0.9, "regime": "nominal"}

    svc = PhaseStreamServicer(source, max_steps=5, interval_s=0.0)
    results = list(svc.StreamPhases(None, _FakeContext()))
    assert len(results) == 5
    for resp in results:
        data = json.loads(resp.payload)
        assert "R" in data
        assert "step" in data
        assert "timestamp" in data


def test_stream_stops_on_inactive_context():
    call_count = 0

    def source():
        nonlocal call_count
        call_count += 1
        return {"R": 0.5}

    ctx = MagicMock()
    ctx.is_active.side_effect = [True, True, False]

    svc = PhaseStreamServicer(source, max_steps=100, interval_s=0.0)
    results = list(svc.StreamPhases(None, ctx))
    assert len(results) == 2


def test_stream_handles_dataclass_state():
    import numpy as np

    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    state = UPDEState(
        layers=[LayerState(R=0.8, psi=0.1)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.8,
        regime_id="nominal",
    )

    svc = PhaseStreamServicer(lambda: state, max_steps=3, interval_s=0.0)
    results = list(svc.StreamPhases(None, _FakeContext()))
    assert len(results) == 3
    data = json.loads(results[0].payload)
    assert "stability_proxy" in data
    assert data["regime_id"] == "nominal"


def test_stream_with_none_context():
    svc = PhaseStreamServicer(
        lambda: {"v": 1}, max_steps=2, interval_s=0.0
    )
    results = list(svc.StreamPhases(None, None))
    assert len(results) == 2
