# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio replay drive-knob tests

"""Studio's ``zeta`` and ``Psi`` knobs drive the replay as SPO defines them.

Everywhere else in SPO ``zeta`` is the driver strength and ``Psi`` the target
phase of ``zeta * sin(Psi - theta)``. The Studio replay added ``zeta * Psi`` to
the natural frequencies instead, so ``zeta = 2, Psi = 0`` changed nothing and
the replay did not show the dynamics its knob labels describe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    run_binding_spec_replay,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "domainpacks" / "minimal_domain" / "binding_spec.yaml"
STEPS = 120


def _replay_r(knobs: StudioKnobState) -> tuple[float, ...]:
    """Return the Studio replay's global order parameter history."""
    return run_binding_spec_replay(SPEC, steps=STEPS, knobs=knobs).r_history


def _driven_simulation_r(zeta: float, psi: float) -> tuple[float, ...]:
    """Return the history of the runtime simulation driven at ``(zeta, psi)``."""
    sim = SimulationState(load_binding_spec(SPEC))
    sim.zeta = zeta
    sim.psi_target = psi
    sim.psi_driver = None
    history: list[float] = []
    for _ in range(STEPS):
        state = sim.step()
        history.append(float(state["R_global"]))  # type: ignore[arg-type]
    return tuple(history)


@pytest.mark.parametrize(("zeta", "psi"), [(2.0, 0.0), (0.8, 3.0)])
def test_positive_zeta_drives_the_replay_like_the_runtime(
    zeta: float, psi: float
) -> None:
    """A positive ``zeta`` replays the runtime's driven dynamics."""
    replay = _replay_r(StudioKnobState(K=1.0, zeta=zeta, Psi=psi))

    assert replay == pytest.approx(_driven_simulation_r(zeta, psi), abs=1e-12)
    assert replay != pytest.approx(_replay_r(StudioKnobState(K=1.0)), abs=1e-6)


def test_psi_without_zeta_keeps_the_spec_drive() -> None:
    """``Psi`` alone sets a target with no strength and changes nothing."""
    assert _replay_r(StudioKnobState(K=1.0, Psi=3.0)) == pytest.approx(
        _replay_r(StudioKnobState(K=1.0)), abs=1e-12
    )
