# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — chaos faults hold their strength and end on time

"""Chaos faults are overlays: constant within their window, gone after it.

The simulation keeps whatever a scenario hook writes, so a hook that applies
``omegas + drift`` every step ramps the drift and leaves it in place after the
window. These tests drive the hook the way the simulation does, with one
context carried across steps, and through ``simulate`` itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    make_chaos_hook,
)
from scpn_phase_orchestrator.runtime.simulation import (
    SimulationScenarioContext,
    simulate,
)

SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)


def _persistent_context(n: int = 4) -> SimulationScenarioContext:
    knm = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    return SimulationScenarioContext(
        spec_name="fixture",
        step=0,
        sample_period_s=0.01,
        phases=np.linspace(0.0, 1.0, n),
        omegas=np.ones(n, dtype=np.float64),
        coupling=CouplingState(
            knm=knm, alpha=np.zeros((n, n)), active_template="t", knm_r=None
        ),
        zeta=0.2,
        psi_target=0.0,
        layer_osc_ranges={0: list(range(n))},
        rng=np.random.default_rng(0),
    )


def _drive(
    schedule: ChaosSchedule,
    steps: int,
    between_steps: dict[int, object] | None = None,
) -> list[tuple[float, float, float]]:
    """Run the hook over one carried context; record (omega, zeta, K01)."""
    hook = make_chaos_hook(schedule)
    context = _persistent_context()
    seen: list[tuple[float, float, float]] = []
    for step in range(steps):
        context.step = step
        action = (between_steps or {}).get(step)
        if callable(action):
            action(context)
        hook(context)
        seen.append(
            (
                float(context.omegas[0]),
                float(context.zeta),
                float(context.coupling.knm[0, 1]),
            )
        )
    return seen


def test_frequency_drift_is_constant_in_window_and_ends() -> None:
    fault = ChaosFault("frequency_drift", start_step=2, duration_steps=3, magnitude=0.5)
    omegas = [row[0] for row in _drive(ChaosSchedule((fault,)), 8)]
    assert omegas == pytest.approx([1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])


def test_drive_dropout_is_constant_in_window_and_ends() -> None:
    fault = ChaosFault("drive_dropout", start_step=2, duration_steps=3, magnitude=0.5)
    zetas = [row[1] for row in _drive(ChaosSchedule((fault,)), 7)]
    assert zetas == pytest.approx([0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])


@pytest.mark.parametrize("magnitude", [0.25, 1.0])
def test_coupling_drop_is_constant_in_window_and_ends(magnitude: float) -> None:
    fault = ChaosFault(
        "coupling_drop", start_step=1, duration_steps=2, magnitude=magnitude
    )
    k01 = [row[2] for row in _drive(ChaosSchedule((fault,)), 5)]
    dropped = 1.0 - magnitude
    assert k01 == pytest.approx([1.0, dropped, dropped, 1.0, 1.0])


def test_overlapping_faults_combine_and_both_end() -> None:
    schedule = ChaosSchedule(
        (
            ChaosFault(
                "frequency_drift", start_step=1, duration_steps=3, magnitude=0.5
            ),
            ChaosFault(
                "frequency_drift", start_step=2, duration_steps=3, magnitude=0.25
            ),
            ChaosFault("coupling_drop", start_step=1, duration_steps=2, magnitude=0.5),
            ChaosFault("coupling_drop", start_step=2, duration_steps=1, magnitude=0.5),
        )
    )
    seen = _drive(schedule, 6)
    assert [row[0] for row in seen] == pytest.approx([1.0, 1.5, 1.75, 1.75, 1.25, 1.0])
    assert [row[2] for row in seen] == pytest.approx([1.0, 0.5, 0.25, 1.0, 1.0, 1.0])


def _supervisor_boosts_coupling(context: SimulationScenarioContext) -> None:
    coupling = context.coupling
    context.coupling = CouplingState(
        knm=coupling.knm * 1.2,
        alpha=coupling.alpha,
        active_template=coupling.active_template,
        knm_r=coupling.knm_r,
    )


def test_supervisor_coupling_boost_survives_the_drop_window() -> None:
    fault = ChaosFault("coupling_drop", start_step=1, duration_steps=3, magnitude=0.5)
    k01 = [
        row[2]
        for row in _drive(ChaosSchedule((fault,)), 6, {2: _supervisor_boosts_coupling})
    ]
    # commanded coupling 1.0 → 1.2 at step 2; the drop halves whatever is
    # commanded, and after the window the commanded 1.2 is restored.
    assert k01 == pytest.approx([1.0, 0.5, 0.6, 0.6, 1.2, 1.2])


def test_supervisor_zeta_change_and_reset_carry_through_dropout() -> None:
    fault = ChaosFault("drive_dropout", start_step=1, duration_steps=4, magnitude=0.5)

    def add(context: SimulationScenarioContext) -> None:
        context.zeta = context.zeta + 0.1

    def reset(context: SimulationScenarioContext) -> None:
        context.zeta = 0.0

    zetas = [row[1] for row in _drive(ChaosSchedule((fault,)), 7, {2: add, 4: reset})]
    assert zetas == pytest.approx([0.2, 0.1, 0.15, 0.15, 0.0, 0.0, 0.0])


def test_simulated_frequency_drift_ends_with_its_window() -> None:
    spec = load_binding_spec(SPEC_PATH)
    hook = make_chaos_hook(
        ChaosSchedule(
            (
                ChaosFault(
                    "frequency_drift", start_step=5, duration_steps=4, magnitude=0.5
                ),
            )
        )
    )
    mean_omega: dict[int, float] = {}

    def recording_hook(context: SimulationScenarioContext) -> None:
        hook(context)
        mean_omega[context.step] = float(np.mean(context.omegas))

    simulate(spec, steps=12, seed=1, policy_enabled=True, scenario_hook=recording_hook)
    nominal = mean_omega[0]
    assert [mean_omega[s] for s in range(5, 9)] == pytest.approx([nominal + 0.5] * 4)
    assert [mean_omega[s] for s in range(9, 12)] == pytest.approx([nominal] * 3)
