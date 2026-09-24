# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — the live server loop integrates the spec's drive

"""``SimulationState`` (dashboard, gRPC stream, Studio replay) matches ``simulate``.

The live loop passed ``zeta = 0`` and ``Psi = 0`` to the engines, dropping the
drive every shipped domainpack declares, so the served dynamics differed from
``spo run`` for the same spec (``swarm_robotics``: R 0.72 vs 0.53 after 300
steps). Both loops now resolve the drive through one function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState
from scpn_phase_orchestrator.runtime.simulation import (
    SimulationScenarioContext,
    _spec_drive,
    simulate,
)

_PACKS = sorted(
    (Path(__file__).resolve().parents[1] / "domainpacks").glob("*/binding_spec.yaml")
)


@pytest.mark.parametrize("path", _PACKS, ids=[p.parent.name for p in _PACKS])
def test_open_loop_server_matches_the_simulation_core(path: Path) -> None:
    spec = load_binding_spec(path)
    served = SimulationState(spec)
    start = served.phases.copy()

    def same_start(context: SimulationScenarioContext) -> None:
        if context.step == 0:
            context.phases = start.copy()

    core = simulate(
        spec, steps=50, seed=0, policy_enabled=False, scenario_hook=same_start
    )
    for _ in range(50):
        served.step()
    wrapped = np.angle(np.exp(1j * (core.final_phases - served.phases)))
    np.testing.assert_allclose(wrapped, 0.0, atol=1e-9)


def test_every_shipped_pack_declares_a_drive_the_server_now_applies() -> None:
    driven = [p.parent.name for p in _PACKS if _spec_drive(load_binding_spec(p))[0] > 0]
    assert len(driven) == len(_PACKS)
    served = SimulationState(load_binding_spec(_PACKS[0]))
    assert served.zeta > 0.0
