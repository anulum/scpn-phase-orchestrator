# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio panel record seal tests

"""Studio review panels recompute the seal of each supervisor record.

The strange-loop result record and the information-geometry proposal record
each carry a SHA-256 of their own canonical JSON. A panel that only checked the
digest format rendered an edited record (scores lowered, a trigger verdict
flipped, a distance zeroed) under the original seal. The panels now
recompute the seal and refuse a record that no longer matches it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor import evaluate_strange_loop_drift_scenarios
from scpn_phase_orchestrator.supervisor.information_geometry import (
    propose_information_geometry_control,
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
