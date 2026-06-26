# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio topos panel tests

"""Studio facade contract tests for the Topos semantic-binding review panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.topos_examples import (
    build_topos_domain_obligation_examples,
)
from scpn_phase_orchestrator.binding.topos_semantic import (
    validate_symbolic_binding_functor,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.topos_policy import (
    validate_policy_composition_category,
)

ToposPayload = tuple[dict[str, object], dict[str, object], dict[str, object]]


@pytest.fixture(scope="module")
def topos_payload() -> ToposPayload:
    """Return production symbolic, policy, and domain-example reports."""
    artifacts = compile_symbolic_binding(
        "1-layer Studio Topos review binding with deterministic evidence morphisms",
        name="studio_topos_panel_contract",
        oscillators_per_layer=1,
        dry_run_steps=1,
    )
    symbolic_report = validate_symbolic_binding_functor(artifacts).to_audit_record()
    policy_report = validate_policy_composition_category(
        [
            PolicyRule(
                name="studio_topos_panel_guard",
                regimes=["DEGRADED"],
                condition=PolicyCondition(
                    metric="R",
                    layer=0,
                    op="<",
                    threshold=0.6,
                ),
                actions=[PolicyAction(knob="K", scope="global", value=0.05, ttl_s=3.0)],
            )
        ]
    ).to_audit_record()
    example = build_topos_domain_obligation_examples()[0]
    return symbolic_report, policy_report, dict(example)


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def _obligations(report: dict[str, object]) -> list[dict[str, object]]:
    """Return report obligation rows with strict test-time typing."""
    return cast("list[dict[str, object]]", report["obligation_records"])


def _objects(report: dict[str, object]) -> list[dict[str, object]]:
    """Return report object rows with strict test-time typing."""
    return cast("list[dict[str, object]]", report["objects"])


def _morphisms(report: dict[str, object]) -> list[dict[str, object]]:
    """Return report morphism rows with strict test-time typing."""
    return cast("list[dict[str, object]]", report["morphisms"])


def test_topos_panel_renders_review_evidence(topos_payload: ToposPayload) -> None:
    """The public Studio facade renders passive Topos validation evidence."""
    symbolic_report, policy_report, example = topos_payload

    panel = studio.build_topos_semantic_binding_studio_panel(
        [_copy_mapping(symbolic_report)],
        [_copy_mapping(policy_report)],
        examples=[_copy_mapping(example)],
    )

    assert panel["panel_kind"] == "studio_topos_semantic_binding_panel"
    assert panel["proof_surface"] == "topos_semantic_binding"
    assert panel["proof_boundary"] == (
        "categorical_validation_prototype_not_formal_topos_proof"
    )
    assert panel["non_actuating"] is True
    assert panel["actuation_permitted"] is False
    assert panel["formal_proof_claim_permitted"] is False
    assert panel["symbolic_report_count"] == 1
    assert panel["policy_report_count"] == 1
    assert panel["example_count"] == 1
    assert panel["passed_symbolic_report_count"] == 1
    assert panel["passed_policy_report_count"] == 1
    assert panel["example_domains"] == (example["domain"],)
    assert panel["object_count_range"]["minimum"] > 0
    assert panel["morphism_count_range"]["minimum"] > 0
    assert panel["example_rows"][0]["obligation_count"] == len(
        cast("list[str]", example["obligation_names"])
    )
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]


def test_topos_panel_reports_failed_validation_hashes(
    topos_payload: ToposPayload,
) -> None:
    """Failed symbolic or policy reports remain passive review evidence."""
    symbolic_report, policy_report, _ = topos_payload
    failed_symbolic = _copy_mapping(symbolic_report)
    failed_policy = _copy_mapping(policy_report)
    failed_symbolic["passed"] = False
    failed_policy["passed"] = False

    panel = studio.build_topos_semantic_binding_studio_panel(
        [failed_symbolic],
        [failed_policy],
    )

    assert panel["passed_symbolic_report_count"] == 0
    assert panel["passed_policy_report_count"] == 0
    assert panel["failed_symbolic_report_hashes"] == [symbolic_report["report_hash"]]
    assert panel["failed_policy_report_hashes"] == [policy_report["report_hash"]]


@pytest.mark.parametrize(
    ("symbolic_reports", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "must be a mapping"),
    ],
)
def test_topos_panel_rejects_malformed_report_sequence(
    topos_payload: ToposPayload,
    symbolic_reports: object,
    match: str,
) -> None:
    """Report sequence validation fails closed before rendering."""
    _, policy_report, _ = topos_payload

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            cast("list[dict[str, object]]", symbolic_reports),
            [_copy_mapping(policy_report)],
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("schema_name", "policy_composition_category", "schema_name"),
        ("proof_boundary", "formal_topos_proof", "proof boundary"),
        ("non_actuating", False, "non_actuating"),
        ("object_count", 0, "object_count"),
        ("morphism_count", 0, "morphism_count"),
    ],
)
def test_topos_panel_rejects_malformed_report_shape(
    topos_payload: ToposPayload,
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Top-level report schema and boundary violations fail closed."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    mutated_report[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


@pytest.mark.parametrize(
    ("obligation_records", "match"),
    [
        ("bad", "must be a sequence"),
        ([], "must not be empty"),
        ([42], "entries must be mappings"),
    ],
)
def test_topos_panel_rejects_malformed_obligation_sequence(
    topos_payload: ToposPayload,
    obligation_records: object,
    match: str,
) -> None:
    """Obligation sequence validation rejects malformed proof evidence."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    mutated_report["obligation_records"] = obligation_records

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


def test_topos_panel_rejects_unsupported_obligation_status(
    topos_payload: ToposPayload,
) -> None:
    """Obligation status values must be explicit pass or fail states."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    obligation_rows = _obligations(mutated_report)
    obligation_rows[0] = {**obligation_rows[0], "status": "unknown"}

    with pytest.raises(ValueError, match="status"):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


@pytest.mark.parametrize(
    ("objects", "match"),
    [
        ("bad", "objects must be a sequence"),
        ([42], "objects entries must be mappings"),
    ],
)
def test_topos_panel_rejects_malformed_named_records(
    topos_payload: ToposPayload,
    objects: object,
    match: str,
) -> None:
    """Topos object validation rejects malformed named records."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    mutated_report["objects"] = objects

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


def test_topos_panel_rejects_empty_named_record_fields(
    topos_payload: ToposPayload,
) -> None:
    """Topos object validation rejects empty optional record metadata."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    object_rows = _objects(mutated_report)
    object_rows[0] = {**object_rows[0], "kind": ""}

    with pytest.raises(ValueError, match="kind"):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


@pytest.mark.parametrize(
    ("morphisms", "match"),
    [
        ("bad", "morphisms must be a sequence"),
        ([42], "morphisms entries must be mappings"),
    ],
)
def test_topos_panel_rejects_malformed_morphism_sequence(
    topos_payload: ToposPayload,
    morphisms: object,
    match: str,
) -> None:
    """Topos morphism validation rejects malformed morphism sequences."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    mutated_report["morphisms"] = morphisms

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


def test_topos_panel_requires_deterministic_morphisms(
    topos_payload: ToposPayload,
) -> None:
    """Topos morphism rows must declare deterministic review evidence."""
    symbolic_report, policy_report, _ = topos_payload
    mutated_report = _copy_mapping(symbolic_report)
    morphism_rows = _morphisms(mutated_report)
    morphism_rows[0] = {**morphism_rows[0], "deterministic": False}

    with pytest.raises(ValueError, match="deterministic"):
        studio.build_topos_semantic_binding_studio_panel(
            [mutated_report],
            [_copy_mapping(policy_report)],
        )


@pytest.mark.parametrize(
    ("examples", "match"),
    [
        ({}, "examples must be a sequence"),
        ([42], "example must be a mapping"),
    ],
)
def test_topos_panel_rejects_malformed_example_sequence(
    topos_payload: ToposPayload,
    examples: object,
    match: str,
) -> None:
    """Example sequence validation fails closed before row flattening."""
    symbolic_report, policy_report, _ = topos_payload

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [_copy_mapping(symbolic_report)],
            [_copy_mapping(policy_report)],
            examples=cast("list[dict[str, object]]", examples),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("proof_boundary", "formal_topos_proof", "proof boundary"),
        ("non_actuating", False, "non_actuating"),
        ("passed", False, "must be passed"),
    ],
)
def test_topos_panel_rejects_malformed_examples(
    topos_payload: ToposPayload,
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Example-level proof-boundary and pass-state violations fail closed."""
    symbolic_report, policy_report, example = topos_payload
    mutated_example = _copy_mapping(example)
    mutated_example[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_topos_semantic_binding_studio_panel(
            [_copy_mapping(symbolic_report)],
            [_copy_mapping(policy_report)],
            examples=[mutated_example],
        )
