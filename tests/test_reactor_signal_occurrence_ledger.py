# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reactor signal occurrence ledger guards

"""Verify the exact, review-only cross-project occurrence ledger snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

LEDGER_PATH = Path("docs/reference/data/reactor_signal_occurrence_ledger.v1.json")
SCHEMA_PATH = Path("docs/specs/reactor_signal_occurrence_ledger.schema.json")

EXPECTED_REVISIONS = {
    "SCPN-CONTROL": "a3b39652f8d97cbdd057afb4e9e5e2859369ab79",
    "SCPN-FUSION-CORE": "c30fb3932b47a812dc26d5846761030cdd0bc94c",
    "SCPN-MIF-CORE": "f3132574b0d4f45b29e2c27cfc2c830ee868c13e",
    "SCPN-PHASE-ORCHESTRATOR": "386c6537b22a3e36fd10402dbe68cffc8721a360",
}
EXPECTED_COUNTS = {
    "SCPN-CONTROL": 9,
    "SCPN-FUSION-CORE": 11,
    "SCPN-MIF-CORE": 11,
    "SCPN-PHASE-ORCHESTRATOR": 12,
}
EXPECTED_IDS = tuple(
    [f"SPO-{index:03d}" for index in range(1, 13)]
    + [f"FUS-{index:03d}" for index in range(1, 12)]
    + [f"MIF-{index:03d}" for index in range(1, 12)]
    + [f"CTRL-{index:03d}" for index in range(1, 10)]
)
SOURCE_DIGEST_SENTINELS = {
    (
        "SCPN-PHASE-ORCHESTRATOR",
        "src/scpn_phase_orchestrator/reactor_semantics/mast_magnetic_review.py",
    ): "a8323c3ebd1767d4498cdb054c3b9337708cee5a75bf01501b4fd7d3a87eb17b",
    (
        "SCPN-FUSION-CORE",
        "src/scpn_fusion/io/mast_magnetic_archive_codec.py",
    ): "721a0f3cbf88ddb0083de19faf83ef27b9592d6cd1aa260d547d1a59b0106734",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/interop/merge_compression_observation.py",
    ): "7ee845dd2566aebbd0324210b372900475a2f6122ef30147c16becfd9afee32e",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/interop/trigger_io.py",
    ): "42b3e41527311685e32ff1b7eb938adb2e67088a24aef39d8e47be2bf083d999",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/lifecycle/plasmoid_merger_petri_net.py",
    ): "90f8bd8457557d714a29a245b710b62aedccfef514e86eb5e38ee9379971bb11",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/diagnostics/normalisation.py",
    ): "9a24a1d2c8d4cbbebaa0722eca5db3e52fcd8eeb3b28192e298f086bdf7b63bb",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/diagnostics/stress_inject.py",
    ): "0d1ea66a0a518dce568a3e1009c8378b9ee01878fcc0b8257e637a2c6e26b2b8",
    (
        "SCPN-MIF-CORE",
        "src/scpn_mif_core/kinematic/trigger_probability.py",
    ): "928cb839f456b8de15ec969bf4951ad98c3f56c0ac0ee22a5767136bacd5afa3",
    (
        "SCPN-CONTROL",
        "src/scpn_control/reactor_semantic_admission/regime_assessment_admission.py",
    ): "fbc243b55d8186d9ddca61ac2b749c6ff2344dcbd68614b2e32d78331e0f2724",
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


def test_occurrence_ledger_matches_its_strict_schema_and_payload_seal() -> None:
    ledger = _load(LEDGER_PATH)
    schema = _load(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(ledger)
    assert (
        ledger["payload_sha256"]
        == hashlib.sha256(_canonical(ledger["payload"])).hexdigest()
    )


def test_occurrence_ledger_pins_exact_revisions_counts_and_order() -> None:
    payload = _load(LEDGER_PATH)["payload"]
    assert isinstance(payload, dict)
    projects = payload["projects"]
    occurrences = payload["occurrences"]
    assert isinstance(projects, list)
    assert isinstance(occurrences, list)

    assert {item["project"]: item["revision"] for item in projects} == (
        EXPECTED_REVISIONS
    )
    assert Counter(item["project"] for item in occurrences) == EXPECTED_COUNTS
    assert tuple(item["occurrence_id"] for item in occurrences) == EXPECTED_IDS
    assert payload["coverage"] == {**EXPECTED_COUNTS, "total_occurrence_groups": 43}


def test_occurrence_rows_preserve_epistemic_and_authority_boundaries() -> None:
    payload = _load(LEDGER_PATH)["payload"]
    assert isinstance(payload, dict)
    occurrences = payload["occurrences"]
    assert isinstance(occurrences, list)

    assert payload["authority"] == "review_only"
    assert payload["actionable"] is False
    assert payload["direct_actuation_authorized"] is False
    assert all(item["externally_actionable"] is False for item in occurrences)
    assert all(item["direct_actuation_authorized"] is False for item in occurrences)

    for item in occurrences:
        if item["physical_observation_admitted"]:
            assert item["physical_source_present"]
        if item["physical_phase_eligible"]:
            assert item["physical_observation_admitted"]
        if item["semantic_carrier"] in {
            "event_timestamp",
            "legacy_normalized_angle",
            "numerical_phase",
            "protocol_phase",
        }:
            assert item["physical_phase_eligible"] is False
        if item["evidence_maturity"] in {
            "physical_archive",
            "physical_source_review",
            "qualification_record",
        }:
            assert item["physical_source_present"] is True

    admitted = [
        item["occurrence_id"]
        for item in occurrences
        if item["physical_observation_admitted"]
    ]
    physical_phase = [
        item["occurrence_id"] for item in occurrences if item["physical_phase_eligible"]
    ]
    model_classifiers = {
        item["occurrence_id"]
        for item in occurrences
        if item["regime_classification_performed"]
    }
    assert admitted == []
    assert physical_phase == []
    assert model_classifiers == {
        "SPO-010",
        "FUS-005",
        "FUS-007",
        "CTRL-007",
        "CTRL-008",
    }


def test_occurrence_sources_are_unique_per_group_and_digest_pinned() -> None:
    payload = _load(LEDGER_PATH)["payload"]
    assert isinstance(payload, dict)
    occurrences = payload["occurrences"]
    assert isinstance(occurrences, list)
    source_index: dict[tuple[str, str], str] = {}

    for item in occurrences:
        paths = [source["path"] for source in item["sources"]]
        assert paths == sorted(paths)
        assert len(paths) == len(set(paths))
        for source in item["sources"]:
            key = (item["project"], source["path"])
            previous = source_index.setdefault(key, source["sha256"])
            assert previous == source["sha256"]

    for key, digest in SOURCE_DIGEST_SENTINELS.items():
        assert source_index[key] == digest


def test_occurrence_gap_references_resolve_and_prior_deltas_do_not_overclaim() -> None:
    payload = _load(LEDGER_PATH)["payload"]
    assert isinstance(payload, dict)
    gap_ids = {item["gap_id"] for item in payload["gap_definitions"]}
    occurrences = payload["occurrences"]
    assert isinstance(occurrences, list)

    assert all(set(item["gap_ids"]) <= gap_ids for item in occurrences)
    state_gap = next(
        item for item in payload["gap_definitions"] if item["gap_id"] == "STATE-01"
    )
    assert "evidence-disposition contract" in state_gap["meaning"]
    assert "map to U0 validity" in state_gap["meaning"]
    assert "require physical-regime abstention" in state_gap["meaning"]
    assert "regime vocabulary" not in state_gap["meaning"]
    statuses = {item["status"] for item in payload["prior_atlas_deltas"]}
    assert statuses == {
        "closed_architecture_gap_physical_evidence_still_open",
        "new_control_review_boundary",
        "new_mif_contract_boundaries_state_gap",
        "new_physical_source_boundary",
        "partially_closed",
    }
    delta_text = " ".join(item["change"] for item in payload["prior_atlas_deltas"])
    assert "do not close phase-observability prerequisites" in delta_text
