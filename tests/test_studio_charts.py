# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio chart payload tests

"""Studio facade contract tests for chart and integrated-information payloads."""

from __future__ import annotations

import json
from copy import deepcopy
from math import log
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio


def _iit_record(
    *,
    include_pairwise: bool = True,
) -> dict[str, object]:
    """Return one valid integrated-information audit record."""
    n_bins = 8
    phi = 0.2
    record: dict[str, object] = {
        "monitor": "integrated_information",
        "phi": phi,
        "normalised_phi": phi / log(n_bins),
        "total_integration": 0.35,
        "minimum_partition": [[0], [1]],
        "n_bins": n_bins,
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
    }
    if include_pairwise:
        record["pairwise_mi"] = [[0.2, 0.05], [0.05, 0.2]]
    return record


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def test_chart_payload_builders_render_series_regimes_and_iit_panel() -> None:
    """The public Studio facade renders deterministic chart payloads."""
    series = studio.build_series_chart_payload("R", [0.1, 0.5, 0.9])
    regimes = studio.build_regime_chart_payload(["critical", "nominal", "custom"])
    records = [_iit_record(), _iit_record(include_pairwise=False)]

    panel = studio.build_integrated_information_panel(records)

    assert series == [
        {"step": 1, "R": 0.1},
        {"step": 2, "R": 0.5},
        {"step": 3, "R": 0.9},
    ]
    assert regimes[0]["regime_level"] == 0.0
    assert regimes[1]["regime_level"] == 2.0
    assert regimes[2]["regime_level"] == 0.0
    assert panel["panel_kind"] == "studio_integrated_information_panel"
    assert panel["monitor"] == "integrated_information"
    assert panel["record_count"] == 2
    assert panel["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
    assert panel["actuation_permitted"] is False
    assert panel["consciousness_claim_permitted"] is False
    assert panel["phi_range"]["min"] == pytest.approx(0.2)
    assert panel["normalised_phi_range"]["max"] == pytest.approx(0.2 / log(8))
    assert panel["strongest_partition"]["minimum_partition"] == [[0], [1]]
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ("bad", "must be a sequence"),
        ([], "must not be empty"),
        ([42], "must be mappings"),
    ],
)
def test_integrated_information_panel_rejects_malformed_record_sequence(
    records: object,
    match: str,
) -> None:
    """Record sequence validation fails closed before rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_integrated_information_panel(
            cast("list[dict[str, object]]", records)
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("monitor", "iit_theory", "monitor tag"),
        ("claim_boundary", "consciousness_claim", "claim boundary"),
        ("normalised_phi", 0.9, "normalised_phi"),
        ("phi", 99.0, "phi"),
        ("total_integration", 99.0, "total_integration"),
        ("total_integration", 0.1, "phi must not exceed"),
    ],
)
def test_integrated_information_panel_rejects_malformed_record_scalars(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Scalar and claim-boundary validation rejects malformed IIT evidence."""
    record = _copy_mapping(_iit_record())
    record[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_integrated_information_panel([record])


@pytest.mark.parametrize(
    ("minimum_partition", "match"),
    [
        ("bad", "two index groups"),
        ([[0]], "two index groups"),
        (["left", [1]], "index sequences"),
        ([[], [1]], "must not be empty"),
        ([[True], [1]], "indices must be integers"),
        ([[-1], [1]], "non-negative"),
        ([[0], [0]], "unique"),
    ],
)
def test_integrated_information_panel_rejects_malformed_partitions(
    minimum_partition: object,
    match: str,
) -> None:
    """Minimum-partition validation rejects malformed node groups."""
    record = _copy_mapping(_iit_record())
    record["minimum_partition"] = minimum_partition

    with pytest.raises(ValueError, match=match):
        studio.build_integrated_information_panel([record])


def test_integrated_information_panel_requires_partition_pairwise_coverage() -> None:
    """Minimum partitions must cover every pairwise-MI node."""
    record = _copy_mapping(_iit_record())
    record["minimum_partition"] = [[0], [2]]

    with pytest.raises(ValueError, match="cover pairwise_mi nodes"):
        studio.build_integrated_information_panel([record])


@pytest.mark.parametrize(
    ("pairwise_mi", "match"),
    [
        ([[True, False], [False, True]], "finite real-valued"),
        ([[0.1j, 0.0], [0.0, 0.1j]], "finite real-valued"),
        ([["x", "y"], ["y", "x"]], "finite real-valued"),
        ([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], "square matrix"),
        ([[0.1]], "at least two oscillators"),
        ([[0.1, float("inf")], [0.0, 0.1]], "finite real-valued"),
        ([[0.1, -0.1], [-0.1, 0.1]], "non-negative"),
        ([[9.0, 0.0], [0.0, 9.0]], "entries must not exceed"),
        ([[0.2, 0.1], [0.0, 0.2]], "symmetric"),
    ],
)
def test_integrated_information_panel_rejects_malformed_pairwise_mi(
    pairwise_mi: object,
    match: str,
) -> None:
    """Pairwise mutual-information validation rejects malformed matrices."""
    record = _copy_mapping(_iit_record())
    record["pairwise_mi"] = pairwise_mi

    with pytest.raises(ValueError, match=match):
        studio.build_integrated_information_panel([record])
