# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor diagnostic-plan portfolio status guards

"""Verify the digest-sealed, fail-closed 20-producer plan status snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

STATUS_PATH = Path(
    "docs/reference/data/reactor_diagnostic_plan_portfolio_status.v1.json"
)
SCHEMA_PATH = Path("docs/specs/reactor_diagnostic_plan_portfolio_status.schema.json")
REFERENCE_PATH = Path("docs/reference/reactor_diagnostic_plan_portfolio_status.md")

EXPECTED_ACCEPTED = {
    "SCPN-DENSE-PLASMA-FOCUS-CORE",
    "SCPN-ICF-BEAM-CORE",
    "SCPN-ICF-IMPACT-CORE",
    "SCPN-ICF-LASER-CORE",
    "SCPN-MIF-LINER-CORE",
    "SCPN-MIF-MAGLIF-CORE",
    "SCPN-MIF-PLASMA-JET-CORE",
    "SCPN-THETA-PINCH-CORE",
    "SCPN-TOKAMAK-CORE",
    "SCPN-Z-PINCH-CORE",
}
EXPECTED_REFUSED = {
    "SCPN-BEAM-TARGET-CORE",
    "SCPN-FRC-CORE",
    "SCPN-FUSION-FISSION-HYBRID-CORE",
    "SCPN-IEC-CORE",
    "SCPN-LEVITATED-DIPOLE-CORE",
    "SCPN-MAGNETIC-CUSP-CORE",
    "SCPN-MIRROR-CORE",
    "SCPN-RFP-CORE",
    "SCPN-SPHEROMAK-CORE",
    "SCPN-STELLARATOR-CORE",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rows() -> list[dict[str, object]]:
    payload = _load(STATUS_PATH)["payload"]
    assert isinstance(payload, dict)
    rows = payload["producers"]
    assert isinstance(rows, list)
    return rows


def test_status_matches_strict_schema_and_payload_seal() -> None:
    status = _load(STATUS_PATH)
    schema = _load(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(status)
    assert (
        status["payload_sha256"]
        == hashlib.sha256(_canonical(status["payload"])).hexdigest()
    )


def test_status_is_an_exact_sorted_twenty_producer_partition() -> None:
    rows = _rows()
    projects = [row["project"] for row in rows]
    accepted = {
        row["project"] for row in rows if row["structural_status"] == "accepted"
    }
    refused = {row["project"] for row in rows if row["structural_status"] == "refused"}

    assert len(rows) == 20
    assert projects == sorted(projects)
    assert len(projects) == len(set(projects))
    assert accepted == EXPECTED_ACCEPTED
    assert refused == EXPECTED_REFUSED
    assert accepted.isdisjoint(refused)


def test_summary_counts_are_derived_from_rows() -> None:
    status = _load(STATUS_PATH)
    payload = status["payload"]
    assert isinstance(payload, dict)
    rows = _rows()
    status_counts = Counter(row["structural_status"] for row in rows)
    custody_counts = Counter(row["custody_state"] for row in rows)

    assert payload["counts"] == {
        "producers": len(rows),
        "structurally_accepted": status_counts["accepted"],
        "structurally_refused": status_counts["refused"],
        "exact_fixture_custody": custody_counts["exact_fixture_custody"],
        "producer_fix_required": custody_counts["producer_fix_required"],
        "qualified_physical_observations": 0,
        "qualified_physical_phases": 0,
    }


def test_accepted_rows_bind_existing_byte_identical_custody() -> None:
    for row in _rows():
        if row["structural_status"] != "accepted":
            continue

        custody_path = row["custody_fixture_path"]
        assert isinstance(custody_path, str)
        fixture = Path(custody_path)
        assert fixture.is_file()
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == row["fixture_sha256"]
        assert row["custody_state"] == "exact_fixture_custody"
        assert row["custody_bytes_equal"] is True
        assert row["refusal_code"] is None
        assert row["refusal_detail"] is None
        assert row["missing_required_members"] == []
        assert row["affected_channel_ids"] == []


def test_refused_rows_preserve_the_exact_producer_owned_gap() -> None:
    detail = "channels[] key mismatch: missing=['timing_uncertainty_s'], unknown=[]"

    for row in _rows():
        if row["structural_status"] != "refused":
            continue

        assert row["custody_state"] == "producer_fix_required"
        assert row["custody_fixture_path"] is None
        assert row["custody_bytes_equal"] is False
        assert row["refusal_code"] == "plan_structure_mismatch"
        assert row["refusal_detail"] == detail
        assert row["missing_required_members"] == ["timing_uncertainty_s"]
        affected = row["affected_channel_ids"]
        assert isinstance(affected, list)
        assert affected
        assert len(affected) == len(set(affected))


def test_no_portfolio_status_row_escalates_epistemic_or_control_authority() -> None:
    status = _load(STATUS_PATH)
    payload = status["payload"]
    assert isinstance(payload, dict)

    for boundary in (payload, *_rows()):
        assert boundary["physical_observation_claimed"] is False
        assert boundary["physical_phase_qualified"] is False
        assert boundary["authority"] == "review_only"
        assert boundary["actionable"] is False
        assert boundary["direct_actuation_authorized"] is False


def test_public_reference_explains_acceptance_refusal_and_fix_forward() -> None:
    text = " ".join(REFERENCE_PATH.read_text(encoding="utf-8").split())

    for marker in (
        "**20 producers** were examined",
        "**10 fixtures** are structurally accepted",
        "**10 fixtures** fail closed",
        "**0 fixtures** constitute a qualified physical observation",
        "omits the required `timing_uncertainty_s` member",
        "omission and a declared non-applicable timing bound are different "
        "source claims",
        "SPO must then replay the new bytes",
        "must not relax the schema or infer defaults",
        "reactor_diagnostic_plan_portfolio_status.v1.json",
        "reactor_diagnostic_plan_portfolio_status.schema.json",
    ):
        assert marker in text

    for project in EXPECTED_ACCEPTED | EXPECTED_REFUSED:
        assert project in text
