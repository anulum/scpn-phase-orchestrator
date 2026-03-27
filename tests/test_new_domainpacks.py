# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — New domainpack integration tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"

NEW_PACKS = [
    "financial_markets",
    "gene_oscillator",
    "vortex_shedding",
    "robotic_cpg",
    "sleep_architecture",
    "musical_acoustics",
    "brain_connectome",
    "autonomous_vehicles",
    "chemical_reactor",
    "circadian_biology",
    "epidemic_sir",
    "manufacturing_spc",
    "network_security",
    "pll_clock",
    "satellite_constellation",
    "swarm_robotics",
    "traffic_flow",
]


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_loads(pack: str) -> None:
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert spec.name == pack
    assert len(spec.layers) >= 2


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_simulates(pack: str) -> None:
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    sim = SimulationState(spec)
    for _ in range(10):
        state = sim.step()
    assert state["step"] == 10
    assert 0.0 <= state["R_global"] <= 1.0
