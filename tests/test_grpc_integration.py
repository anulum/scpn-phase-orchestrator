# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC integration test

"""In-process gRPC servicer test using mocked context.

grpcio is optional — these tests work without it by mocking the
context object (same pattern as server_grpc.py docstring).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.server_grpc import PhaseStreamServicer

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


def _make_servicer() -> PhaseStreamServicer:
    spec = load_binding_spec(DOMAINPACK_DIR / "minimal_domain" / "binding_spec.yaml")
    sim = SimulationState(spec)
    return PhaseStreamServicer(sim)


class TestGRPCServicer:
    def test_get_state(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        resp = servicer.GetState(MagicMock(), ctx)
        assert resp.step == 0
        assert 0.0 <= resp.R_global <= 1.0

    def test_step_advances(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        resp = servicer.Step(MagicMock(n_steps=1), ctx)
        assert resp.step == 1
        assert np.isfinite(resp.R_global)

    def test_multiple_steps(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        for _ in range(5):
            resp = servicer.Step(MagicMock(n_steps=1), ctx)
        assert resp.step == 5

    def test_reset(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        servicer.Step(MagicMock(n_steps=1), ctx)
        servicer.Step(MagicMock(n_steps=1), ctx)
        resp = servicer.Reset(MagicMock(), ctx)
        assert resp.step == 0

    def test_get_config(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        resp = servicer.GetConfig(MagicMock(), ctx)
        assert resp.name == "minimal_domain"
        assert resp.n_oscillators > 0
        assert resp.n_layers >= 2

    def test_state_has_layers(self) -> None:
        servicer = _make_servicer()
        ctx = MagicMock()
        resp = servicer.GetState(MagicMock(), ctx)
        assert len(resp.layers) >= 2
        for layer in resp.layers:
            assert 0.0 <= layer.R <= 1.0
