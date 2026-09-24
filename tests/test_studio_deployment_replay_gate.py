# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio deployment replay gate tests

"""Deployment packages state the replay gate from the recorded replay status.

The operator checklist marks "Run local replay" as blocked until the replay
status is ``completed``. The deployment package and the materialisation plan
listed the safety gate "local replay completed" for the same project state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    binding_spec_project_state,
    build_deployment_package,
    build_operator_checklist,
    build_package_materialisation_plan,
    build_runtime_snapshot,
)
from scpn_phase_orchestrator.studio.workflow import StudioProjectState

ROOT = Path(__file__).resolve().parents[1]


def _state(replay_status: str) -> StudioProjectState:
    """Return a minimal-domain project state with ``replay_status``."""
    knobs = StudioKnobState(K=1.0)
    return binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=ROOT / "domainpacks/minimal_domain/binding_spec.yaml",
        knobs=knobs,
        runtime=build_runtime_snapshot(
            final_state={
                "R_global": 0.72,
                "regime": "nominal",
                "layers": [{"name": "layer-a", "R": 0.7}],
            },
            knobs=knobs,
            replay_status=replay_status,
        ),
    )


@pytest.mark.parametrize("replay_status", ["not_started", "failed"])
def test_unfinished_replay_is_not_reported_as_completed(replay_status: str) -> None:
    """Package and plan agree with the checklist that the replay is not done."""
    state = _state(replay_status)
    assert build_operator_checklist(state)[0]["status"] == "blocked"

    package_gates = build_deployment_package(state)["safety_gates"]
    plan_gates = build_package_materialisation_plan(state)["safety_gates"]

    for gates in (package_gates, plan_gates):
        assert isinstance(gates, list)
        assert "local replay completed" not in gates
        assert "local replay not completed" in gates


def test_completed_replay_is_reported_as_completed() -> None:
    """A completed replay keeps the completed gate in package and plan."""
    state = _state("completed")

    assert "local replay completed" in build_deployment_package(state)["safety_gates"]
    assert (
        "local replay completed"
        in (build_package_materialisation_plan(state)["safety_gates"])
    )
