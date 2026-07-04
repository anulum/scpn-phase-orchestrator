# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — power-grid PRC audit bundle tests

"""Tests for the hash-sealed power-grid PRC assessor bundle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.power_grid_prc_bundle import (
    DVOC_DAMPING_ROLE,
    IBR_RIDE_THROUGH_ROLE,
    PMU_RINGDOWN_ROLE,
    POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA,
    POWER_GRID_PRC_CLAIM_BOUNDARY,
    PowerGridPRCInputArtifact,
    build_power_grid_prc_audit_bundle,
)

_CREATED_AT = "2026-07-04T15:10:00Z"
DVOC_OSCILLATION_AUDIT_SCHEMA = "scpn_dvoc_oscillation_damping_audit_v1"
PMU_RINGDOWN_AUDIT_SCHEMA = "scpn_pmu_ringdown_prc_audit_v1"
IBR_RIDE_THROUGH_AUDIT_SCHEMA = "scpn_ibr_ride_through_prc029_audit_v1"


def _evidence_record(
    *,
    schema: str,
    event_id: str,
    flagged_count: int,
    review_only: bool = True,
    claim_boundary: str = POWER_GRID_PRC_CLAIM_BOUNDARY,
) -> dict[str, object]:
    """Return a deterministic evidence record with a valid content hash."""
    payload: dict[str, object] = {
        "schema": schema,
        "event_id": event_id,
        "flagged_count": flagged_count,
        "claim_boundary": claim_boundary,
        "review_only": review_only,
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _artifact(
    role: str,
    schema: str,
    *,
    source_sha256: str | None = None,
) -> PowerGridPRCInputArtifact:
    """Return one input artifact for the bundle builder."""
    default_hashes = {
        DVOC_DAMPING_ROLE: "a" * 64,
        PMU_RINGDOWN_ROLE: "b" * 64,
        IBR_RIDE_THROUGH_ROLE: "c" * 64,
    }
    return PowerGridPRCInputArtifact(
        role=role,
        source_name=f"{role}.json",
        source_sha256=source_sha256 or default_hashes.get(role, "d" * 64),
        record=_evidence_record(
            schema=schema,
            event_id=f"{role}-event",
            flagged_count=1 if role != DVOC_DAMPING_ROLE else 0,
        ),
    )


def _complete_artifacts() -> tuple[PowerGridPRCInputArtifact, ...]:
    """Return the three required power-grid evidence artifacts."""
    return (
        _artifact(DVOC_DAMPING_ROLE, DVOC_OSCILLATION_AUDIT_SCHEMA),
        _artifact(PMU_RINGDOWN_ROLE, PMU_RINGDOWN_AUDIT_SCHEMA),
        _artifact(IBR_RIDE_THROUGH_ROLE, IBR_RIDE_THROUGH_AUDIT_SCHEMA),
    )


def test_build_power_grid_prc_audit_bundle_seals_all_required_artifacts() -> None:
    """The bundle binds dVOC, PMU, and PRC-029 evidence under one digest."""
    bundle = build_power_grid_prc_audit_bundle(
        bundle_id="PG-REVIEW-001",
        created_at=_CREATED_AT,
        operator_context="western interconnection post-event review",
        artifacts=_complete_artifacts(),
    )

    assert bundle.schema == POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA
    assert bundle.claim_boundary == POWER_GRID_PRC_CLAIM_BOUNDARY
    assert bundle.review_only is True
    assert bundle.bundle_id == "PG-REVIEW-001"
    assert bundle.operator_context == "western interconnection post-event review"
    assert tuple(artifact.role for artifact in bundle.artifacts) == (
        DVOC_DAMPING_ROLE,
        PMU_RINGDOWN_ROLE,
        IBR_RIDE_THROUGH_ROLE,
    )
    assert bundle.evidence_hashes == {
        artifact.role: artifact.evidence_hash for artifact in bundle.artifacts
    }

    record = bundle.to_audit_record()
    assert record["schema"] == POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA
    assert record["claim_boundary"] == POWER_GRID_PRC_CLAIM_BOUNDARY
    assert record["review_only"] is True
    assert record["evidence_hashes"] == dict(bundle.evidence_hashes)
    assert len(str(record["content_hash"])) == 64
    assert record == bundle.to_audit_record()


@pytest.mark.parametrize(
    ("artifacts", "match"),
    [
        (
            _complete_artifacts()[:2],
            "missing required power-grid evidence roles",
        ),
        (
            _complete_artifacts()
            + (_artifact(PMU_RINGDOWN_ROLE, PMU_RINGDOWN_AUDIT_SCHEMA),),
            "duplicate power-grid evidence role pmu_ringdown",
        ),
        (
            (
                _artifact(DVOC_DAMPING_ROLE, PMU_RINGDOWN_AUDIT_SCHEMA),
                _artifact(PMU_RINGDOWN_ROLE, PMU_RINGDOWN_AUDIT_SCHEMA),
                _artifact(IBR_RIDE_THROUGH_ROLE, IBR_RIDE_THROUGH_AUDIT_SCHEMA),
            ),
            "dvoc_oscillation_damping evidence schema",
        ),
    ],
)
def test_build_power_grid_prc_audit_bundle_rejects_incomplete_role_set(
    artifacts: tuple[PowerGridPRCInputArtifact, ...], match: str
) -> None:
    """The handoff package must contain exactly the required evidence roles."""
    with pytest.raises(ValueError, match=match):
        build_power_grid_prc_audit_bundle(
            bundle_id="PG-REVIEW-001",
            created_at=_CREATED_AT,
            operator_context="operator review",
            artifacts=artifacts,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda record: {**record, "content_hash": "0" * 64},
            "pmu_ringdown content_hash does not match",
        ),
        (
            lambda record: {**record, "review_only": False},
            "pmu_ringdown must be review-only",
        ),
        (
            lambda record: {
                **record,
                "claim_boundary": "live_actuation_claim_forbidden",
                "content_hash": canonical_record_hash(
                    {
                        key: value
                        for key, value in {
                            **record,
                            "claim_boundary": "live_actuation_claim_forbidden",
                        }.items()
                        if key != "content_hash"
                    }
                ),
            },
            "pmu_ringdown claim_boundary",
        ),
    ],
)
def test_build_power_grid_prc_audit_bundle_rejects_untrusted_records(
    mutator: object, match: str
) -> None:
    """Tampered or non-review evidence records fail before bundle hashing."""
    artifacts = list(_complete_artifacts())
    original = artifacts[1]
    mutate = mutator
    if not callable(mutate):
        raise AssertionError("mutator must be callable")
    artifacts[1] = PowerGridPRCInputArtifact(
        role=original.role,
        source_name=original.source_name,
        source_sha256=original.source_sha256,
        record=mutate(original.record),
    )

    with pytest.raises(ValueError, match=match):
        build_power_grid_prc_audit_bundle(
            bundle_id="PG-REVIEW-001",
            created_at=_CREATED_AT,
            operator_context="operator review",
            artifacts=tuple(artifacts),
        )


@pytest.mark.parametrize(
    ("artifact", "match"),
    [
        (
            PowerGridPRCInputArtifact(
                role="unknown",
                source_name="unknown.json",
                source_sha256="a" * 64,
                record=_evidence_record(
                    schema=PMU_RINGDOWN_AUDIT_SCHEMA,
                    event_id="unknown",
                    flagged_count=0,
                ),
            ),
            "unsupported power-grid evidence role",
        ),
        (
            PowerGridPRCInputArtifact(
                role=PMU_RINGDOWN_ROLE,
                source_name="",
                source_sha256="a" * 64,
                record=_evidence_record(
                    schema=PMU_RINGDOWN_AUDIT_SCHEMA,
                    event_id="pmu",
                    flagged_count=0,
                ),
            ),
            "source_name must be a non-empty string",
        ),
        (
            PowerGridPRCInputArtifact(
                role=PMU_RINGDOWN_ROLE,
                source_name="pmu.json",
                source_sha256="not-a-sha",
                record=_evidence_record(
                    schema=PMU_RINGDOWN_AUDIT_SCHEMA,
                    event_id="pmu",
                    flagged_count=0,
                ),
            ),
            "source_sha256 must be a 64-character",
        ),
    ],
)
def test_power_grid_prc_input_artifact_rejects_bad_metadata(
    artifact: PowerGridPRCInputArtifact, match: str
) -> None:
    """Invalid role and source metadata are rejected during bundle assembly."""
    artifacts = (
        _artifact(DVOC_DAMPING_ROLE, DVOC_OSCILLATION_AUDIT_SCHEMA),
        artifact,
        _artifact(IBR_RIDE_THROUGH_ROLE, IBR_RIDE_THROUGH_AUDIT_SCHEMA),
    )

    with pytest.raises(ValueError, match=match):
        build_power_grid_prc_audit_bundle(
            bundle_id="PG-REVIEW-001",
            created_at=_CREATED_AT,
            operator_context="operator review",
            artifacts=artifacts,
        )


def test_power_grid_prc_input_artifact_rejects_non_mapping_record() -> None:
    """Evidence records must be parsed JSON objects before bundle assembly."""
    artifacts = (
        _artifact(DVOC_DAMPING_ROLE, DVOC_OSCILLATION_AUDIT_SCHEMA),
        PowerGridPRCInputArtifact(
            role=PMU_RINGDOWN_ROLE,
            source_name="pmu.json",
            source_sha256="a" * 64,
            record=cast(Mapping[str, object], object()),
        ),
        _artifact(IBR_RIDE_THROUGH_ROLE, IBR_RIDE_THROUGH_AUDIT_SCHEMA),
    )

    with pytest.raises(ValueError, match="pmu_ringdown record must be a mapping"):
        build_power_grid_prc_audit_bundle(
            bundle_id="PG-REVIEW-001",
            created_at=_CREATED_AT,
            operator_context="operator review",
            artifacts=artifacts,
        )
