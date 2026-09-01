# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Diagnostic-plan envelope 1.2 depth tests
"""Replay exact TOKAMAK 1.2 bytes and refuse declaration-depth drift."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import scpn_phase_orchestrator.reactor_semantics as rs
import scpn_phase_orchestrator.reactor_semantics.diagnostic_plan_depth as depth

FIXTURES = Path("tests/fixtures/tokamak_diagnostic_plan_v1_2")
HISTORICAL_FIXTURES = Path("tests/fixtures/tokamak_diagnostic_plan")
SOURCE_REVISION = "7402191c43e8fe57cffda1dd5b3cf4319d6d398d"
ARTIFACT_SHA256 = "a0c6ccbf8c398d80ed65f03a82e7a313761d09dea81acb5ab8ad565997cb2720"
MANIFEST_SHA256 = "ed4dd4f86eb7a62bf9674c0bfffa341f3afed42b7754ddcd809bc0b1804a19ab"
FIXTURE_SHA256 = "8e0c0d51f6c7aece428a6e761adf20f820f44aa6946b05921912cc4c87790253"
PLAN_SHA256 = "6a015adfaa2cda7ec1bf04fc685d00d6b7209ca9b78d761fff47ab0919eeec94"


def _compact(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _records(
    fixture_root: Path = FIXTURES,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = json.loads((fixture_root / "plan_envelope_fixture.json").read_text())
    manifest = json.loads((fixture_root / "reactor-domain.json").read_text())
    return manifest, fixture["envelope"], fixture["plan"]


def _source_bytes(
    manifest: dict[str, Any],
    envelope: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[bytes, bytes, bytes]:
    manifest_bytes = _pretty(manifest)
    plan_bytes = _compact(plan)
    envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    envelope["plan_identifier"] = plan["identifier"]
    return manifest_bytes, _compact(envelope), plan_bytes


def _review(
    manifest: dict[str, Any], envelope: dict[str, Any], plan: dict[str, Any]
) -> rs.DeviceDiagnosticPlanReview:
    manifest_bytes, envelope_bytes, plan_bytes = _source_bytes(
        deepcopy(manifest), deepcopy(envelope), deepcopy(plan)
    )
    return rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=ARTIFACT_SHA256,
        manifest_bytes=manifest_bytes,
        envelope_bytes=envelope_bytes,
        plan_bytes=plan_bytes,
    )


def _assert_refusal(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest, envelope, plan = _records()
    mutate(plan)
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest, envelope, plan)
    assert caught.value.code is code


def _add_non_event_timing_marker(plan: dict[str, Any]) -> None:
    channel = next(
        item
        for item in plan["channels"]
        if item["candidate_id"] == "closed.equilibrium_profiles"
    )
    channel["signals"].append(
        {
            "description": "invalid timing marker",
            "identifier": "sig_invalid_marker",
            "quantity": "time",
            "role": "timing_marker",
            "unit": "s",
        }
    )
    channel["signals"].sort(key=lambda item: item["identifier"])


def _add_duplicate_clock_domain(plan: dict[str, Any]) -> None:
    plan["clock_topology"]["domains"].append(
        {
            "identifier": "dom_second",
            "member_clock_identifiers": ["clk_shot"],
            "root_clock_identifier": "clk_shot",
            "scope": "invalid duplicate membership",
        }
    )


def _split_clock_domains_without_reference_relation(plan: dict[str, Any]) -> None:
    plan["clock_topology"]["domains"] = [
        {
            "identifier": "dom_facility",
            "member_clock_identifiers": ["clk_facility"],
            "root_clock_identifier": "clk_facility",
            "scope": "facility reference",
        },
        {
            "identifier": "dom_shot",
            "member_clock_identifiers": ["clk_shot"],
            "root_clock_identifier": "clk_shot",
            "scope": "shot domain",
        },
    ]
    plan["clock_relations"] = []


def _make_frame_transformations_unsorted(plan: dict[str, Any]) -> None:
    plan["frames"].insert(
        0,
        {
            "description": "synthetic Boozer coordinates",
            "identifier": "frm_boozer",
            "kind": "boozer",
        },
    )
    plan["frame_transformations"].append(
        {
            "equilibrium_dependent": True,
            "evidence_claimed": False,
            "kind": "flux_mapping",
            "method": "synthetic declaration only",
            "source_identifier": "frm_flux",
            "target_identifier": "frm_boozer",
        }
    )


def test_exact_tokamak_1_2_fixture_is_byte_identical_and_accepted() -> None:
    fixture_bytes = (FIXTURES / "plan_envelope_fixture.json").read_bytes()
    manifest_bytes = (FIXTURES / "reactor-domain.json").read_bytes()
    fixture = json.loads(fixture_bytes)

    assert depth.validate_diagnostic_plan_depth.__module__.endswith(
        "diagnostic_plan_depth"
    )
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256
    assert hashlib.sha256(_compact(fixture["plan"])).hexdigest() == PLAN_SHA256
    review = rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=ARTIFACT_SHA256,
        manifest_bytes=manifest_bytes,
        envelope_bytes=_compact(fixture["envelope"]),
        plan_bytes=_compact(fixture["plan"]),
    )

    assert review.source_envelope_schema_version == "1.2.0"
    assert (
        review.source_envelope_sha256
        == hashlib.sha256(_compact(fixture["envelope"])).hexdigest()
    )
    assert review.source_plan_sha256 == PLAN_SHA256
    assert review.source_plan_json.encode() == _compact(fixture["plan"])
    assert review.accepted_as_design_declaration
    assert not review.evidence_claimed
    assert not review.observation_claimed
    assert not review.classification_performed
    assert not review.semantic_ingress_declared
    assert not review.control_intent_created
    assert not review.actionable
    assert (
        rs.device_diagnostic_plan_review_from_bytes(
            rs.device_diagnostic_plan_review_to_bytes(review)
        )
        == review
    )


def test_historical_1_1_fixture_remains_accepted_without_shape_coercion() -> None:
    manifest, envelope, plan = _records(HISTORICAL_FIXTURES)
    review = _review(manifest, envelope, plan)

    assert review.source_envelope_schema_version == "1.1.0"
    assert "signals" not in plan["channels"][0]
    assert "frame_transformations" not in plan
    assert "clock_topology" not in plan


def test_envelope_versions_dispatch_to_exact_noninterchangeable_plan_shapes() -> None:
    manifest_1_2, envelope_1_2, plan_1_2 = _records()
    _, envelope_1_1, plan_1_1 = _records(HISTORICAL_FIXTURES)

    envelope_1_2["schema_version"] = "1.1.0"
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest_1_2, envelope_1_2, plan_1_2)
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH
    )

    envelope_1_1["schema_version"] = "1.2.0"
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest_1_2, envelope_1_1, plan_1_1)
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH
    )


def test_non_string_envelope_version_is_a_controlled_schema_refusal() -> None:
    manifest, envelope, plan = _records()
    envelope["schema_version"] = ["1.2.0"]

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest, envelope, plan)

    assert (
        caught.value.code
        is rs.DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA
    )


def test_signal_quantity_text_cannot_override_registered_candidate_or_carrier() -> None:
    manifest, envelope, plan = _records()
    equilibrium = next(
        channel
        for channel in plan["channels"]
        if channel["candidate_id"] == "closed.equilibrium_profiles"
    )
    carrier = next(
        signal for signal in equilibrium["signals"] if signal["role"] == "carrier"
    )
    carrier["quantity"] = "phase"
    carrier["unit"] = "rad"

    review = _review(manifest, envelope, plan)
    reviewed = next(
        item
        for item in review.signal_reviews
        if item.candidate_id == "closed.equilibrium_profiles"
    )
    assert reviewed.observability_class is rs.ObservabilityClass.NONCYCLIC_FEATURE
    assert reviewed.carrier is rs.SemanticCarrier.BOUNDED_FEATURE
    assert not reviewed.observation_claimed


@pytest.mark.parametrize(
    ("code", "mutate"),
    (
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0].pop("signals"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].__setitem__(
                "role", "unknown"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__(
                "signals",
                [
                    signal
                    for signal in plan["channels"][0]["signals"]
                    if signal["role"] != "timing_marker"
                ],
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][-1]["signals"].append(
                {
                    "description": "extra model value",
                    "identifier": "sig_second",
                    "quantity": "phase",
                    "role": "auxiliary",
                    "unit": "rad",
                }
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "evidence_claimed", True
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__("kind", "rigid"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan.__setitem__("frame_transformations", []),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0][
                "member_clock_identifiers"
            ].append("clk_sim"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"].__setitem__("domains", []),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__("scope", ""),
        ),
    ),
)
def test_signal_frame_and_topology_drift_is_refused(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _assert_refusal(code, mutate)


@pytest.mark.parametrize(
    ("code", "mutate"),
    (
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__("signals", {}),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__("signals", ["invalid"]),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].pop("description"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].__setitem__(
                "identifier", "INVALID"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].__setitem__(
                "description", ""
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__("signals", []),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].__setitem__("unit", "rad s"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][0].__setitem__(
                "role", "auxiliary"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["channels"][0]["signals"][1].__setitem__(
                "identifier", plan["channels"][0]["signals"][0]["identifier"]
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            _add_non_event_timing_marker,
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "target_identifier", "frm_machine"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "target_identifier", "frm_unknown"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "kind", "unknown"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "equilibrium_dependent", False
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__("method", ""),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frame_transformations"][0].__setitem__(
                "evidence_claimed", "false"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            _make_frame_transformations_unsorted,
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan.__setitem__("clock_topology", []),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "member_clock_identifiers", []
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "member_clock_identifiers", ["clk_shot"]
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "member_clock_identifiers", ["clk_facility", "clk_unknown"]
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            _add_duplicate_clock_domain,
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "root_clock_identifier", "clk_shot"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"].__setitem__(
                "reference_domain_identifier", "dom_unknown"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "member_clock_identifiers", ["clk_facility"]
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan.__setitem__("clock_relations", []),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            _split_clock_domains_without_reference_relation,
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["clock_topology"]["domains"][0].__setitem__(
                "member_clock_identifiers", ["clk_facility", 1]
            ),
        ),
    ),
)
def test_malformed_declaration_shapes_are_refused_at_public_intake(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _assert_refusal(code, mutate)


def test_reverse_frame_pair_and_clock_relation_cycle_are_refused() -> None:
    def duplicate_pair(plan: dict[str, Any]) -> None:
        original = deepcopy(plan["frame_transformations"][0])
        original["source_identifier"], original["target_identifier"] = (
            original["target_identifier"],
            original["source_identifier"],
        )
        plan["frame_transformations"].append(original)

    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH, duplicate_pair
    )

    def cycle(plan: dict[str, Any]) -> None:
        relation = deepcopy(plan["clock_relations"][0])
        relation["child_identifier"], relation["parent_identifier"] = (
            relation["parent_identifier"],
            relation["child_identifier"],
        )
        plan["clock_relations"].append(relation)
        plan["clock_relations"].sort(
            key=lambda item: (item["child_identifier"], item["parent_identifier"])
        )

    _assert_refusal(rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH, cycle)
