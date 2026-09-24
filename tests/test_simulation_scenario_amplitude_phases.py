# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scenario phase edits reach the Stuart-Landau state

"""A scenario hook's phase edit must reach the amplitude-mode integrator.

``simulate`` integrates ``sl_state`` (phases and amplitudes) in amplitude
mode, built once before the loop, so a hook that rewrote ``context.phases``
was silently ignored: chaos ``sensor_noise`` of 2 rad had no effect on 32 of
the 36 shipped domainpacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.server import SimulationState
from scpn_phase_orchestrator.runtime.simulation import (
    SimulationScenarioContext,
    simulate,
)

_SPEC = (
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)


def test_minimal_domain_runs_in_amplitude_mode() -> None:
    assert load_binding_spec(_SPEC).amplitude is not None


def test_hook_phase_edit_is_integrated_like_an_initial_state() -> None:
    spec = load_binding_spec(_SPEC)
    reference = SimulationState(spec)
    start = reference.phases.copy()
    seen: dict[int, np.ndarray] = {}

    def hook(context: SimulationScenarioContext) -> None:
        if context.step == 0:
            context.phases = start.copy()
        context.zeta = 0.0
        context.psi_target = 0.0
        seen[context.step] = context.phases.copy()

    simulate(spec, steps=2, seed=0, policy_enabled=False, scenario_hook=hook)
    reference.step()  # the same Stuart-Landau step from the same phases, no drive
    np.testing.assert_allclose(seen[1], reference.phases, atol=1e-12)


def test_sensor_noise_perturbs_an_amplitude_mode_run() -> None:
    result = run_resilience_experiment(
        load_binding_spec(_SPEC),
        ChaosSchedule((ChaosFault("sensor_noise", 5, 20, 2.0),)),
        steps=60,
        seed=1,
    )
    assert result.metrics.max_coherence_drop > 0.05
    assert result.perturbed_final_r != pytest.approx(result.nominal_final_r)
