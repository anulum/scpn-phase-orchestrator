# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor-semantic serialization tests

"""Real public-facade portfolio and strict serialization tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scpn_phase_orchestrator.reactor_semantics import (
    ACTION_OWNER,
    PLANT_TRUTH_OWNERS,
    REVIEW_ONLY_AUTHORITY,
    SEMANTIC_OWNER,
    PhaseRelationType,
    RelationInterpretation,
    SemanticCarrier,
    build_phase_relation,
    build_reactor_reference_portfolio,
    canonical_json,
    contract_digest,
    contract_from_json,
    contract_from_record,
    contract_to_record,
)

SCHEMA = Path("docs/specs/reactor_semantics_u0.schema.json")


def test_reference_portfolio_spans_nine_families_and_all_carriers() -> None:
    portfolio = build_reactor_reference_portfolio()

    assert tuple(item.slice_id for item in portfolio) == (
        "A1",
        "N1",
        "C1",
        "O1",
        "P1",
        "I1",
        "H1",
        "E1",
        "X1",
    )
    carriers = {record.carrier_type for item in portfolio for record in item.semantics}
    assert carriers == set(SemanticCarrier)
    assert {item.context.configuration for item in portfolio} == {
        "conventional_tokamak",
        "stellarator",
        "field_reversed_configuration",
        "tandem_mirror",
        "z_pinch",
        "laser_icf_indirect_drive",
        "maglif",
        "gridded_iec",
        "fusion_fission_hybrid",
    }
    assert {"SCPN-FUSION-CORE", "SCPN-MIF-CORE"} == PLANT_TRUTH_OWNERS
    assert all(item.regime.semantic_owner == SEMANTIC_OWNER for item in portfolio)
    assert all(item.regime.action_owner == ACTION_OWNER for item in portfolio)
    assert all(item.regime.authority == REVIEW_ONLY_AUTHORITY for item in portfolio)
    assert all(
        item.context.operating_point["production_actuation"] is False
        for item in portfolio
    )
    assert all(item.observable.provenance.attributes for item in portfolio)


def test_all_five_contracts_round_trip_through_deterministic_public_codec() -> None:
    item = build_reactor_reference_portfolio()[0]
    source = item.semantics[0]
    target = replace(source, phase_id="u0.a1.roundtrip_target", phase_rad=0.8)
    relation = build_phase_relation(
        source,
        target,
        relation_id="u0.a1.roundtrip_relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
        identification_method="deterministic_fixture",
        evidence_class=source.evidence_class,
    )
    contracts = (item.context, item.observable, source, relation, item.regime)

    for contract in contracts:
        encoded = canonical_json(contract)
        assert contract_from_json(encoded) == contract
        assert canonical_json(contract_from_json(encoded)) == encoded
        assert contract_digest(contract) == contract_digest(contract)
        assert contract_from_record(contract_to_record(contract)) == contract


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda record: record.update(extra=True), "unknown fields"),
        (lambda record: record.pop("schema_version"), "missing fields"),
        (lambda record: record.update(schema_version="2.0.0"), "unsupported schema"),
        (
            lambda record: record.update(contract_type="invented"),
            "unsupported contract_type",
        ),
    ],
)
def test_codec_refuses_unknown_missing_or_incompatible_envelopes(
    mutation,
    match: str,
) -> None:
    record = contract_to_record(build_reactor_reference_portfolio()[0].context)
    mutation(record)
    with pytest.raises(ValueError, match=match):
        contract_from_record(record)


def test_codec_refuses_payload_version_drift_and_duplicate_json_keys() -> None:
    record = contract_to_record(build_reactor_reference_portfolio()[0].context)
    payload = record["payload"]
    assert isinstance(payload, dict)
    payload["schema_version"] = "0.9.0"
    with pytest.raises(ValueError, match="versions must match"):
        contract_from_record(record)

    duplicate = '{"contract_type":"reactor_context","contract_type":"other"}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        contract_from_json(duplicate)
    with pytest.raises(ValueError, match="invalid"):
        contract_from_json("{")
    with pytest.raises(ValueError, match="non-empty"):
        contract_from_json("")


def test_canonical_json_is_sorted_and_contains_no_nonfinite_values() -> None:
    context = build_reactor_reference_portfolio()[0].context
    encoded = canonical_json(context)
    parsed = json.loads(encoded)

    assert encoded.startswith('{"contract_type"')
    assert parsed["payload"]["operating_point"]["production_actuation"] is False
    with pytest.raises(ValueError, match="non-finite"):
        replace(context, operating_point={"bad": float("nan")})


def test_portable_json_schema_accepts_every_public_contract_kind() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    portfolio = build_reactor_reference_portfolio()
    source = portfolio[0].semantics[0]
    target = replace(source, phase_id="u0.a1.schema_target", phase_rad=0.9)
    relation = build_phase_relation(
        source,
        target,
        relation_id="u0.a1.schema_relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
        identification_method="schema_fixture",
        evidence_class=source.evidence_class,
    )
    contracts = [
        *(item.context for item in portfolio),
        *(item.observable for item in portfolio),
        *(record for item in portfolio for record in item.semantics),
        relation,
        *(item.regime for item in portfolio),
    ]

    for contract in contracts:
        validator.validate(contract_to_record(contract))

    mismatched = contract_to_record(portfolio[0].context)
    mismatched["contract_type"] = "regime_estimate"
    with pytest.raises(ValidationError):
        validator.validate(mismatched)

    regime = contract_to_record(portfolio[0].regime)
    payload = regime["payload"]
    assert isinstance(payload, dict)
    payload["authority"] = "actuate"
    with pytest.raises(ValidationError):
        validator.validate(regime)
