# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio panel record seal tests

"""Studio review panels recompute the seal of each supervisor record.

The strange-loop result record, the information-geometry proposal record, and
the multiverse rollout manifest and risk report each carry a SHA-256 of their
own canonical JSON. A panel that only checked the
digest format rendered an edited record (scores lowered, a trigger verdict
flipped, a distance zeroed) under the original seal. The panels now
recompute the seal and refuse a record that no longer matches it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import evaluate_strange_loop_drift_scenarios
from scpn_phase_orchestrator.supervisor.information_geometry import (
    propose_information_geometry_control,
)
from scpn_phase_orchestrator.supervisor.multiverse import (
    simulate_multiverse_counterfactual_branches,
)
from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    MultiverseRiskThresholds,
    evaluate_multiverse_branch_risk,
)
from tests.sealing import seal


def _strange_loop_records() -> list[dict[str, object]]:
    """Return production strange-loop drift scenario audit records."""
    return [
        cast("dict[str, object]", deepcopy(result.to_audit_record()))
        for result in evaluate_strange_loop_drift_scenarios()
    ]


def _proposal_record() -> dict[str, object]:
    """Return a production information-geometry proposal audit record."""
    record = propose_information_geometry_control(
        [0.16, 0.27, 0.18, 0.39],
        [0.21, 0.23, 0.27, 0.29],
        coupling_gradient=[0.05, -0.02, 0.04, -0.01],
        max_step=0.08,
        knob="K",
        scope="power_grid",
    ).to_audit_record()
    return cast("dict[str, object]", deepcopy(record))


@pytest.mark.parametrize(
    ("field_name", "edited_value"),
    [
        ("passed_expected_trigger", False),
        ("max_drift_score", 0.0),
        ("min_control_coherence", 1.0),
        ("triggered_recommendation_count", 0),
    ],
)
def test_edited_strange_loop_record_is_refused(
    field_name: str, edited_value: object
) -> None:
    """A strange-loop record edited after sealing no longer renders."""
    records = _strange_loop_records()
    target = next(r for r in records if r[field_name] != edited_value)
    target[field_name] = edited_value

    with pytest.raises(ValueError, match="result_hash does not match the record"):
        studio.build_strange_loop_studio_panel(records)


def test_resealed_strange_loop_record_renders() -> None:
    """The refusal comes from the seal: the same edit, resealed, renders."""
    records = _strange_loop_records()
    records[0]["passed_expected_trigger"] = False
    records[0] = seal(records[0], "result_hash")

    panel = studio.build_strange_loop_studio_panel(records)

    assert records[0]["scenario_id"] in panel["failed_scenario_ids"]


def test_edited_information_geometry_proposal_is_refused() -> None:
    """A proposal whose recorded Wasserstein distance was edited is refused."""
    record = _proposal_record()
    record["wasserstein_distance"] = 0.0

    with pytest.raises(ValueError, match="proposal_hash does not match the record"):
        studio.build_information_geometry_studio_panel([record])


def test_resealed_information_geometry_proposal_renders() -> None:
    """The same proposal edit, resealed, renders with the edited value."""
    record = _proposal_record()
    record["wasserstein_distance"] = 0.0
    record = seal(record, "proposal_hash")

    panel = studio.build_information_geometry_studio_panel([record])

    series = panel["series"]
    assert isinstance(series, tuple)
    assert series[0]["wasserstein_distance"] == 0.0


def _multiverse_records() -> tuple[dict[str, object], dict[str, object]]:
    """Return a production rollout manifest and a risk report rejecting both."""
    coupling = np.array([[0.0, 0.15, 0.15], [0.15, 0.0, 0.15], [0.15, 0.15, 0.0]])
    manifest = simulate_multiverse_counterfactual_branches(
        phases=np.array([0.10, 1.20, 2.40]),
        omegas=np.array([0.05, -0.02, 0.01]),
        baseline_k=coupling,
        baseline_alpha=np.zeros((3, 3)),
        branch_action_sets=(
            (),
            (ControlAction("K", "global", 0.25, 1.0, "coupling review"),),
        ),
        horizon=8,
        dt=0.02,
    ).to_audit_record()
    risk = evaluate_multiverse_branch_risk(
        manifest,
        MultiverseRiskThresholds(min_mean_R=0.9, min_final_R=0.0, max_action_count=4),
    ).to_audit_record()
    return (
        cast("dict[str, object]", deepcopy(manifest)),
        cast("dict[str, object]", deepcopy(risk)),
    )


def _approve_every_branch(risk: dict[str, object]) -> None:
    """Rewrite a risk report so that every branch reads as approved."""
    decisions = cast("list[dict[str, object]]", risk["branch_decisions"])
    for decision in decisions:
        decision["approved"] = True
        decision["rejection_reasons"] = []
    risk["approved_count"] = len(decisions)
    risk["rejected_count"] = 0
    risk["rejection_reasons"] = []
    risk["safest_branch_id"] = decisions[0]["branch_id"]
    risk["safest_branch_hash"] = decisions[0]["branch_hash"]


def test_risk_report_rewritten_to_approve_is_refused() -> None:
    """Rejected branches rewritten as approved do not render as approved."""
    manifest, risk = _multiverse_records()
    assert risk["approved_count"] == 0
    _approve_every_branch(risk)

    with pytest.raises(ValueError, match="report_hash does not match the record"):
        studio.build_multiverse_counterfactual_studio_panel(manifest, risk)


def test_resealed_risk_report_renders_as_approved() -> None:
    """The same rewrite, resealed, renders: the refusal is the seal."""
    manifest, risk = _multiverse_records()
    _approve_every_branch(risk)
    risk = seal(risk, "report_hash")

    panel = studio.build_multiverse_counterfactual_studio_panel(manifest, risk)

    assert panel["approved_count"] == 2


def test_edited_rollout_manifest_is_refused() -> None:
    """A manifest whose horizon was edited no longer matches its seal."""
    manifest, risk = _multiverse_records()
    manifest["horizon"] = 80

    with pytest.raises(ValueError, match="manifest_hash does not match the record"):
        studio.build_multiverse_counterfactual_studio_panel(manifest, risk)


def test_resealed_rollout_manifest_renders() -> None:
    """The manifest seal is taken over the record with its seal field blank."""
    manifest, risk = _multiverse_records()
    manifest["horizon"] = 80
    manifest = seal(manifest, "manifest_hash", blanked=True)

    panel = studio.build_multiverse_counterfactual_studio_panel(manifest, risk)

    assert panel["horizon"] == 80
