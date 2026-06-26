# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio information-geometry panel tests

"""Studio facade contract tests for the information-geometry review panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor.information_geometry import (
    propose_information_geometry_control,
)
from scpn_phase_orchestrator.supervisor.information_geometry_examples import (
    build_information_geometry_control_scenarios,
)


def _proposal_record(
    *,
    backend: str = "numpy",
) -> dict[str, object]:
    """Return a production information-geometry proposal audit record."""
    return propose_information_geometry_control(
        [0.16, 0.27, 0.18, 0.39],
        [0.21, 0.23, 0.27, 0.29],
        coupling_gradient=[0.05, -0.02, 0.04, -0.01],
        max_step=0.08,
        knob="K",
        scope="power_grid",
        backend=backend,
    ).to_audit_record()


def _scenario_record() -> dict[str, object]:
    """Return a production information-geometry scenario audit record."""
    return dict(build_information_geometry_control_scenarios()[0])


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like copy for negative-path mutations."""
    return cast("dict[str, object]", deepcopy(payload))


def _state(record: dict[str, object]) -> dict[str, object]:
    """Return the nested state record with a strict test-time type."""
    return cast("dict[str, object]", record["state"])


def _actions(record: dict[str, object]) -> list[dict[str, object]]:
    """Return the action proposal list with a strict test-time type."""
    return cast("list[dict[str, object]]", record["action_proposals"])


def test_information_geometry_panel_renders_proposal_and_scenario_evidence() -> None:
    """The public Studio facade renders non-actuating geometry review evidence."""
    proposals = [_proposal_record(), _proposal_record(backend="jax")]
    scenarios = build_information_geometry_control_scenarios()

    panel = studio.build_information_geometry_studio_panel(
        proposals,
        scenarios=scenarios,
    )

    assert panel["panel_kind"] == "studio_information_geometry_panel"
    assert panel["claim_boundary"] == (
        "information_geometry_control_not_live_actuation"
    )
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["proposal_count"] == 2
    assert panel["scenario_count"] == len(scenarios)
    assert panel["latest"]["backend"] == "jax_native_information_geometry"
    assert panel["latest"]["scope"] == "power_grid"
    assert panel["backends"] == (
        "jax_native_information_geometry",
        "numpy_jax_compatible_information_geometry",
    )
    assert panel["metric_diagonal_range"]["minimum"] > 0.0
    assert panel["fisher_rao_range"]["maximum"] >= panel["fisher_rao_range"]["minimum"]
    assert panel["candidate_rows"][0]["control_knobs"]
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]
    assert len(decoded_panel["candidate_rows"]) == len(panel["candidate_rows"])


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "record must be a mapping"),
    ],
)
def test_information_geometry_panel_rejects_malformed_record_sequence(
    records: object,
    match: str,
) -> None:
    """Record sequence validation fails closed before rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel(
            cast("list[dict[str, object]]", records)
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("claim_boundary", "live_control", "claim boundary"),
        ("non_actuating", False, "non_actuating"),
        ("execution_disabled", False, "execution_disabled"),
        ("backend", "live_backend", "backend"),
        ("state", "bad", "state"),
    ],
)
def test_information_geometry_panel_rejects_malformed_record_shape(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Top-level proposal schema and boundary violations fail closed."""
    record = _copy_mapping(_proposal_record())
    record[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel([record])


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("target_coordinates", [0.5, 0.5], "target_coordinates"),
        ("tangent_vector", [0.1, 0.2], "tangent_vector"),
        ("curvature_proxy", 0.123, "curvature_proxy"),
        ("geodesic_length", 0.123, "geodesic_length"),
        ("simplex_coordinates", [0.5, -0.1, 0.2, 0.4], "non-negative"),
        ("simplex_coordinates", [0.0, 0.0, 0.0, 0.0], "positive mass"),
        ("simplex_coordinates", [0.2, 0.2, 0.2, 0.2], "normalised"),
        ("metric_tensor", "bad", "square matrix"),
        ("metric_tensor", [[1.0, 0.0], [0.0, 1.0]], "row count"),
        (
            "metric_tensor",
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "column count",
        ),
        (
            "metric_tensor",
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "diagonal",
        ),
        (
            "metric_tensor",
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "symmetric",
        ),
    ],
)
def test_information_geometry_panel_rejects_malformed_state(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Nested information-geometry state validation catches malformed evidence."""
    record = _copy_mapping(_proposal_record())
    state = dict(_state(record))
    state[field_name] = bad_value
    record["state"] = state

    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel([record])


@pytest.mark.parametrize(
    ("actions", "match"),
    [
        ("bad", "action_proposals"),
        ([], "one review action"),
        ([{"knob": "K"}, {"knob": "zeta"}], "one review action"),
        ([42], "entries"),
        ([{"knob": "", "scope": "global", "value": 0.1, "ttl_s": 1.0}], "knob"),
        ([{"knob": "K", "scope": "", "value": 0.1, "ttl_s": 1.0}], "scope"),
        ([{"knob": "K", "scope": "global", "value": True, "ttl_s": 1.0}], "value"),
        ([{"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 0.0}], "ttl_s"),
        (
            [{"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 1.0}],
            "justification",
        ),
    ],
)
def test_information_geometry_panel_rejects_malformed_review_actions(
    actions: object,
    match: str,
) -> None:
    """Review action proposals must be exactly one complete non-executing action."""
    record = _copy_mapping(_proposal_record())
    record["action_proposals"] = actions

    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel([record])


@pytest.mark.parametrize(
    ("scenarios", "match"),
    [
        ({}, "scenarios must be a sequence"),
        ([42], "scenario must be a mapping"),
    ],
)
def test_information_geometry_panel_rejects_malformed_scenario_sequence(
    scenarios: object,
    match: str,
) -> None:
    """Scenario sequence validation fails closed before table rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel(
            [_proposal_record()],
            scenarios=cast("list[dict[str, object]]", scenarios),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("claim_boundary", "live_control", "claim boundary"),
        ("non_actuating", False, "non_actuating"),
        ("execution_disabled", False, "execution_disabled"),
        ("target_distribution", [0.5, 0.5], "target_distribution"),
        ("current_distribution", [0.4, -0.1, 0.7], "non-negative"),
        ("current_distribution", [0.0, 0.0, 0.0], "positive mass"),
        ("current_distribution", [0.2, 0.2, 0.2], "normalised"),
        ("objective_labels", "bad", "objective_labels"),
        ("knob_hints", "bad", "knob_hints"),
        ("control_gradient", "bad", "control_gradient"),
        ("scenario_id", "", "scenario_id"),
        ("domain", "", "domain"),
        ("scenario_hash", "bad", "scenario_hash"),
        ("max_step", 0.0, "max_step"),
    ],
)
def test_information_geometry_panel_rejects_malformed_scenarios(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Scenario validation catches unsafe boundaries and malformed candidates."""
    scenario = _copy_mapping(_scenario_record())
    scenario[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_information_geometry_studio_panel(
            [_proposal_record()],
            scenarios=[scenario],
        )


def test_information_geometry_panel_rejects_malformed_gradient_pair() -> None:
    """Scenario control-gradient entries must be knob/value pairs."""
    scenario = _copy_mapping(_scenario_record())
    scenario["control_gradient"] = [["K", 0.1, 0.2]]

    with pytest.raises(ValueError, match="knob/value pairs"):
        studio.build_information_geometry_studio_panel(
            [_proposal_record()],
            scenarios=[scenario],
        )
