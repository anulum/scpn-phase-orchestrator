# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio sheaf-cohomology panel tests

"""Studio facade contract tests for the sheaf-cohomology review panel."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor import (
    build_sheaf_obstruction_summary,
    propose_sheaf_obstruction_control,
    sheaf_coherence,
)


class _TruthyEmptyFloatSequence(Sequence[float]):
    """A truthy empty sequence for the panel's post-normalisation residual guard."""

    def __len__(self) -> int:
        """Return zero elements while preserving an explicit truth value."""
        return 0

    def __getitem__(self, index: int) -> float:
        """Raise for every item because the sequence is intentionally empty."""
        raise IndexError(index)

    def __bool__(self) -> bool:
        """Return true so shared sequence normalisation reaches panel validation."""
        return True


def _sheaf_state() -> tuple[np.ndarray, np.ndarray]:
    """Return production sheaf states and directed restriction maps."""
    states = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    maps = np.zeros((3, 3, 2, 2), dtype=np.float64)
    for target in range(3):
        for source in range(3):
            if target != source:
                maps[target, source] = np.eye(2, dtype=np.float64)
    return states, maps


def _panel_payload() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return production sheaf result, summary, and proposal audit records."""
    states, maps = _sheaf_state()
    result = sheaf_coherence(states, maps)
    summary = build_sheaf_obstruction_summary(
        result,
        warning_threshold=0.05,
        critical_threshold=0.25,
        top_k=4,
    ).to_audit_record()
    proposal = propose_sheaf_obstruction_control(
        states,
        maps,
        step_size=0.25,
        max_update_norm=0.4,
    ).to_audit_record()
    return result.to_audit_record(), summary, proposal


def _record() -> dict[str, object]:
    """Return one mutable sheaf-cohomology result record."""
    record, _, _ = _panel_payload()
    return _copy_mapping(record)


def _summary() -> dict[str, object]:
    """Return one mutable sheaf-obstruction summary record."""
    _, summary, _ = _panel_payload()
    return _copy_mapping(summary)


def _proposal() -> dict[str, object]:
    """Return one mutable sheaf-control proposal record."""
    _, _, proposal = _panel_payload()
    return _copy_mapping(proposal)


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def _top_edges(summary: dict[str, object]) -> list[dict[str, object]]:
    """Return a summary's top residual edges with strict test-time typing."""
    return cast("list[dict[str, object]]", summary["top_residual_edges"])


def test_sheaf_panel_renders_review_evidence() -> None:
    """The public Studio facade renders passive sheaf-cohomology review evidence."""
    record, summary, proposal = _panel_payload()

    panel = studio.build_sheaf_cohomology_studio_panel(
        [record],
        summaries=[summary],
        control_proposals=[proposal],
    )

    assert panel["panel_kind"] == "studio_sheaf_cohomology_panel"
    assert panel["supervisor"] == "sheaf_cohomology_control"
    assert panel["claim_boundary"] == "sheaf_cohomology_review_not_live_actuation"
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["operator_review_required"] is True
    assert panel["actuation_permitted"] is False
    assert panel["live_merge_permitted"] is False
    assert panel["hot_patch_permitted"] is False
    assert panel["record_count"] == 1
    assert panel["summary_count"] == 1
    assert panel["control_proposal_count"] == 1
    assert panel["accepted_control_proposal_count"] == 1
    assert panel["critical_summary_count"] == 1
    assert len(panel["top_residual_rows"]) == 4
    assert (
        panel["obstruction_range"]["maximum"] >= panel["obstruction_range"]["minimum"]
    )
    assert (
        panel["consistency_energy_range"]["maximum"]
        >= panel["consistency_energy_range"]["minimum"]
    )
    assert panel["cohomology_dimension_range"]["kernel_minimum"] >= 0
    assert panel["control_proposals"][0]["blocked_reasons"] == ()
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]
    assert len(decoded_panel["top_residual_rows"]) == len(panel["top_residual_rows"])


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "record must be a mapping"),
    ],
)
def test_sheaf_panel_rejects_malformed_record_sequence(
    records: object,
    match: str,
) -> None:
    """Record sequence validation fails closed before rendering."""
    summary = _summary()
    proposal = _proposal()

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            cast("list[dict[str, object]]", records),
            summaries=[summary],
            control_proposals=[proposal],
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("method", "live_sheaf_control", "method"),
        ("laplacian_shape", [5, 6], "laplacian_shape"),
        ("residual_shape", [3, 2, 2], "residual_shape"),
        ("residual_shape", [3, 3, 1], "match laplacian_shape"),
        ("laplacian_shape", "bad", "must be a sequence"),
        ("laplacian_shape", [6], "rank 2"),
    ],
)
def test_sheaf_panel_rejects_malformed_record_shapes(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Sheaf result shape validation rejects malformed audit evidence."""
    record = _record()
    record[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [record],
            summaries=[_summary()],
            control_proposals=[_proposal()],
        )


@pytest.mark.parametrize(
    ("summaries", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "summary must be a mapping"),
    ],
)
def test_sheaf_panel_rejects_malformed_summary_sequence(
    summaries: object,
    match: str,
) -> None:
    """Summary sequence validation fails closed before residual flattening."""
    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=cast("list[dict[str, object]]", summaries),
            control_proposals=[_proposal()],
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("severity", "emergency", "severity"),
        ("critical_threshold", 0.01, "critical_threshold"),
        ("top_residual_edges", {}, "top_residual_edges"),
    ],
)
def test_sheaf_panel_rejects_malformed_summary_shape(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Summary-level schema validation rejects unsupported evidence."""
    summary = _summary()
    summary[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=[summary],
            control_proposals=[_proposal()],
        )


@pytest.mark.parametrize(
    ("edge", "match"),
    [
        (42, "must be a mapping"),
        ({"residual": _TruthyEmptyFloatSequence()}, "residual must not be empty"),
    ],
)
def test_sheaf_panel_rejects_malformed_residual_rows(
    edge: object,
    match: str,
) -> None:
    """Residual-row validation rejects malformed or empty residual evidence."""
    summary = _summary()
    edges = _top_edges(summary)
    if isinstance(edge, dict):
        mutated_edge = dict(edges[0])
        mutated_edge.update(edge)
        edges[0] = mutated_edge
    else:
        edges[0] = cast("dict[str, object]", edge)

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=[summary],
            control_proposals=[_proposal()],
        )


@pytest.mark.parametrize(
    ("proposals", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "proposal must be a mapping"),
    ],
)
def test_sheaf_panel_rejects_malformed_proposal_sequence(
    proposals: object,
    match: str,
) -> None:
    """Proposal sequence validation fails closed before rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=[_summary()],
            control_proposals=cast("list[dict[str, object]]", proposals),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("method", "live_sheaf_control", "method"),
        ("non_actuating", False, "non_actuating"),
        ("execution_disabled", False, "execution_disabled"),
        ("operator_review_required", False, "operator_review_required"),
        ("accepted_for_review", "yes", "accepted_for_review"),
        ("projected_obstruction_score", 99.0, "projected obstruction"),
        ("projected_consistency_energy", 99.0, "projected energy"),
        ("update_norm", 1.0, "update_norm exceeds"),
        ("cohomology_dimensions", [], "cohomology_dimensions"),
    ],
)
def test_sheaf_panel_rejects_malformed_proposal_shape(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Proposal-level schema and review-boundary violations fail closed."""
    proposal = _proposal()
    proposal[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=[_summary()],
            control_proposals=[proposal],
        )


@pytest.mark.parametrize(
    ("accepted", "blocked_reasons", "match"),
    [
        (True, ["manual_block"], "accepted proposal"),
        (False, [], "rejected proposal"),
    ],
)
def test_sheaf_panel_rejects_inconsistent_proposal_review_status(
    accepted: bool,
    blocked_reasons: list[str],
    match: str,
) -> None:
    """Accepted and rejected proposal rows must carry coherent block metadata."""
    proposal = _proposal()
    proposal["accepted_for_review"] = accepted
    proposal["blocked_reasons"] = blocked_reasons

    with pytest.raises(ValueError, match=match):
        studio.build_sheaf_cohomology_studio_panel(
            [_record()],
            summaries=[_summary()],
            control_proposals=[proposal],
        )
