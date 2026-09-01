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
    "SCPN-BEAM-TARGET-CORE",
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
EXPECTED_REFUSED: set[str] = set()
EXPECTED_VERIFIED_PUBLIC = EXPECTED_ACCEPTED
EXPECTED_WORKFLOWS = (
    "CI",
    "CodeQL",
    "Docs",
    "Pre-commit",
    "SBOM",
    "Scorecard",
    "Security audit",
)


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
    assert status["schema_version"] == "1.2.0"
    assert status["payload"]["review_contract"].endswith("@1.2.0")
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
        "verified_public_producer_object": custody_counts[
            "verified_public_producer_object"
        ],
        "producer_fix_required": custody_counts["producer_fix_required"],
        "qualified_physical_observations": 0,
        "qualified_physical_phases": 0,
    }


def test_accepted_rows_bind_verified_public_objects_and_hosted_runs() -> None:
    for row in _rows():
        if row["structural_status"] != "accepted":
            continue
        assert row["refusal_code"] is None
        assert row["refusal_detail"] is None
        assert row["missing_required_members"] == []
        assert row["affected_channel_ids"] == []

        assert row["project"] in EXPECTED_VERIFIED_PUBLIC
        assert row["custody_state"] == "verified_public_producer_object"
        assert row["custody_fixture_path"] is None
        assert row["custody_bytes_equal"] is False
        assert row["remote_head_verified"] is True
        run_ids = row["hosted_ci_run_ids"]
        assert len(run_ids) == len(EXPECTED_WORKFLOWS)
        assert len(set(run_ids)) == len(run_ids)
        assert all(isinstance(run_id, int) and run_id > 0 for run_id in run_ids)


def test_remote_verification_is_exact_and_non_scientific() -> None:
    payload = _load(STATUS_PATH)["payload"]
    assert isinstance(payload, dict)
    verification = payload["remote_verification"]
    run_ids = [run_id for row in _rows() for run_id in row["hosted_ci_run_ids"]]
    assert verification["default_branch"] == "main"
    assert verification["remote_head_matches"] == 20
    assert verification["remote_head_mismatches"] == 0
    assert tuple(verification["workflow_names"]) == EXPECTED_WORKFLOWS
    assert verification["hosted_workflows_expected"] == 140
    assert verification["hosted_workflows_successful"] == 140
    assert verification["hosted_workflows_failed"] == 0
    assert verification["hosted_workflows_cancelled"] == 0
    assert verification["run_attempt"] == 1
    assert len(run_ids) == 140
    assert len(set(run_ids)) == len(run_ids)
    assert "not physical or operational evidence" in verification["evidence_boundary"]


def test_refused_rows_preserve_the_exact_producer_owned_gap() -> None:
    assert not [row for row in _rows() if row["structural_status"] == "refused"]


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
        "**20 fixtures** are structurally accepted",
        "**0 current fixtures** have byte-identical SPO custody",
        "**20 fixtures** are digest-pinned public producer objects",
        "**140/140 hosted workflows** completed successfully",
        "**0 fixtures** constitute a qualified physical observation",
        "20 accepted / 0 refused",
        "reactor_diagnostic_plan_portfolio_status.v1.json",
        "reactor_diagnostic_plan_portfolio_status.schema.json",
    ):
        assert marker in text

    for project in EXPECTED_ACCEPTED | EXPECTED_REFUSED:
        assert project in text
