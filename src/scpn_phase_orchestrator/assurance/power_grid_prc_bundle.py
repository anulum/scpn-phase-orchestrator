# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — power-grid PRC assessor bundle

"""Hash-sealed power-grid PRC assessor bundles.

The power-grid review lane emits several independent evidence records: the dVOC
oscillation-damping audit, an operator PMU ringdown screen, and an IBR
ride-through screen. This module binds those records into one deterministic,
review-only handoff package. It verifies each child content hash before the
bundle is sealed, so an assessor can detect both source-file mutation and
evidence-record mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)

__all__ = [
    "DVOC_DAMPING_ROLE",
    "IBR_RIDE_THROUGH_ROLE",
    "PMU_RINGDOWN_ROLE",
    "POWER_GRID_PRC_AUDIT_BUNDLE_DISCLAIMER",
    "POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA",
    "POWER_GRID_PRC_CLAIM_BOUNDARY",
    "POWER_GRID_PRC_REQUIRED_ROLES",
    "PowerGridPRCArtifact",
    "PowerGridPRCAuditBundle",
    "PowerGridPRCInputArtifact",
    "build_power_grid_prc_audit_bundle",
]

DVOC_DAMPING_ROLE = "dvoc_oscillation_damping"
PMU_RINGDOWN_ROLE = "pmu_ringdown"
IBR_RIDE_THROUGH_ROLE = "ibr_ride_through"

POWER_GRID_PRC_REQUIRED_ROLES = (
    DVOC_DAMPING_ROLE,
    PMU_RINGDOWN_ROLE,
    IBR_RIDE_THROUGH_ROLE,
)

POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA = "scpn_power_grid_prc_audit_bundle_v1"
POWER_GRID_PRC_CLAIM_BOUNDARY = "review_only_offline_no_live_actuation"

POWER_GRID_PRC_AUDIT_BUNDLE_DISCLAIMER = (
    "This power-grid PRC audit bundle is a deterministic technical handoff for "
    "qualified assessor review. It binds review-only dVOC oscillation-damping, "
    "PMU ringdown, and PRC-029 ride-through evidence records; it does not "
    "constitute legal advice, certification, a conformity assessment, or "
    "permission for live actuation."
)

_ROLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        DVOC_DAMPING_ROLE: "scpn_dvoc_oscillation_damping_audit_v1",
        PMU_RINGDOWN_ROLE: "scpn_pmu_ringdown_prc_audit_v1",
        IBR_RIDE_THROUGH_ROLE: "scpn_ibr_ride_through_prc029_audit_v1",
    }
)


@dataclass(frozen=True, slots=True)
class PowerGridPRCInputArtifact:
    """A source evidence record prepared for bundle assembly.

    Attributes
    ----------
    role : str
        Required evidence role in the power-grid PRC bundle.
    source_name : str
        Operator-facing basename or label of the evidence JSON source.
    source_sha256 : str
        SHA-256 digest of the exact evidence JSON bytes consumed by the bundle
        builder.
    record : Mapping[str, object]
        Parsed evidence record. Its own ``content_hash`` is rechecked before the
        bundle is emitted.
    """

    role: str
    source_name: str
    source_sha256: str
    record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PowerGridPRCArtifact:
    """A validated evidence artifact inside the assessor bundle.

    Attributes
    ----------
    role : str
        Required evidence role in the bundle.
    source_name : str
        Source evidence JSON basename or label.
    source_sha256 : str
        SHA-256 digest of the exact evidence JSON bytes consumed.
    evidence_schema : str
        Schema identifier in the child evidence record.
    evidence_hash : str
        Validated child ``content_hash``.
    record : Mapping[str, object]
        Parsed child evidence record, preserved verbatim for assessor replay.
    """

    role: str
    source_name: str
    source_sha256: str
    evidence_schema: str
    evidence_hash: str
    record: Mapping[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the artifact.

        Returns
        -------
        dict[str, object]
            Stable source metadata plus the preserved child evidence record.
        """
        return {
            "role": self.role,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "evidence_schema": self.evidence_schema,
            "evidence_hash": self.evidence_hash,
            "record": dict(self.record),
        }


@dataclass(frozen=True, slots=True)
class PowerGridPRCAuditBundle:
    """A hash-sealed power-grid PRC assessor handoff bundle.

    Attributes
    ----------
    schema : str
        Bundle schema identifier.
    bundle_id : str
        Operator-assigned bundle identifier.
    created_at : str
        Timestamp supplied by the caller.
    operator_context : str
        Human-readable review context for the assessor handoff.
    artifacts : tuple[PowerGridPRCArtifact, ...]
        Validated child artifacts in required role order.
    evidence_hashes : Mapping[str, str]
        Read-only map from role to child evidence hash.
    claim_boundary : str
        Review-only claim boundary.
    review_only : bool
        Always ``True`` for this bundle.
    disclaimer : str
        Regulatory and live-actuation disclaimer.
    content_hash : str
        SHA-256 of the canonical bundle payload excluding this field.
    """

    schema: str
    bundle_id: str
    created_at: str
    operator_context: str
    artifacts: tuple[PowerGridPRCArtifact, ...]
    evidence_hashes: Mapping[str, str]
    claim_boundary: str = POWER_GRID_PRC_CLAIM_BOUNDARY
    review_only: bool = True
    disclaimer: str = POWER_GRID_PRC_AUDIT_BUNDLE_DISCLAIMER
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Freeze hash maps and compute the canonical bundle digest."""
        object.__setattr__(
            self,
            "evidence_hashes",
            MappingProxyType(dict(self.evidence_hashes)),
        )
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for hashing and JSON export."""
        return {
            "schema": self.schema,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "operator_context": self.operator_context,
            "artifacts": [artifact.to_audit_record() for artifact in self.artifacts],
            "evidence_hashes": dict(self.evidence_hashes),
            "claim_boundary": self.claim_boundary,
            "review_only": self.review_only,
            "disclaimer": self.disclaimer,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the assessor bundle.

        Returns
        -------
        dict[str, object]
            The canonical bundle payload plus ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def build_power_grid_prc_audit_bundle(
    *,
    bundle_id: str,
    created_at: str,
    operator_context: str,
    artifacts: Sequence[PowerGridPRCInputArtifact],
) -> PowerGridPRCAuditBundle:
    """Build a deterministic power-grid PRC assessor handoff bundle.

    Parameters
    ----------
    bundle_id : str
        Operator-assigned bundle identifier.
    created_at : str
        Timestamp supplied by the caller.
    operator_context : str
        Human-readable review context.
    artifacts : Sequence[PowerGridPRCInputArtifact]
        Candidate child evidence records with source-file digests.

    Returns
    -------
    PowerGridPRCAuditBundle
        Hash-sealed bundle containing exactly the required child roles.

    Raises
    ------
    ValueError
        If identifiers, source metadata, child schemas, child hashes, or the
        role set are invalid.
    """
    bundle = _non_empty_str(bundle_id, "bundle_id")
    timestamp = _non_empty_str(created_at, "created_at")
    context = _non_empty_str(operator_context, "operator_context")
    ordered_artifacts = _validate_artifacts(artifacts)
    evidence_hashes = {
        artifact.role: artifact.evidence_hash for artifact in ordered_artifacts
    }
    return PowerGridPRCAuditBundle(
        schema=POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA,
        bundle_id=bundle,
        created_at=timestamp,
        operator_context=context,
        artifacts=ordered_artifacts,
        evidence_hashes=evidence_hashes,
    )


def _validate_artifacts(
    artifacts: Sequence[PowerGridPRCInputArtifact],
) -> tuple[PowerGridPRCArtifact, ...]:
    """Return validated artifacts in required role order."""
    by_role: dict[str, PowerGridPRCArtifact] = {}
    for candidate in artifacts:
        artifact = _validate_artifact(candidate)
        if artifact.role in by_role:
            raise ValueError(f"duplicate power-grid evidence role {artifact.role}")
        by_role[artifact.role] = artifact
    missing = [role for role in POWER_GRID_PRC_REQUIRED_ROLES if role not in by_role]
    if missing:
        raise ValueError(
            "missing required power-grid evidence roles: " + ", ".join(missing)
        )
    return tuple(by_role[role] for role in POWER_GRID_PRC_REQUIRED_ROLES)


def _validate_artifact(candidate: PowerGridPRCInputArtifact) -> PowerGridPRCArtifact:
    """Return a validated child artifact."""
    role = _non_empty_str(candidate.role, "role")
    if role not in _ROLE_SCHEMAS:
        raise ValueError(f"unsupported power-grid evidence role {role}")
    source_name = _non_empty_str(candidate.source_name, "source_name")
    source_sha256 = require_sha256(candidate.source_sha256, "source_sha256")
    if not isinstance(candidate.record, Mapping):
        raise ValueError(f"{role} record must be a mapping")
    record = dict(candidate.record)
    expected_schema = _ROLE_SCHEMAS[role]
    evidence_schema = _non_empty_str(record.get("schema"), f"{role} evidence schema")
    if evidence_schema != expected_schema:
        raise ValueError(
            f"{role} evidence schema must be {expected_schema}, got {evidence_schema}"
        )
    if record.get("claim_boundary") != POWER_GRID_PRC_CLAIM_BOUNDARY:
        raise ValueError(
            f"{role} claim_boundary must be {POWER_GRID_PRC_CLAIM_BOUNDARY}"
        )
    if record.get("review_only") is not True:
        raise ValueError(f"{role} must be review-only")
    evidence_hash = require_sha256(record.get("content_hash"), f"{role} content_hash")
    payload_without_hash = {
        key: value for key, value in record.items() if key != "content_hash"
    }
    actual_hash = canonical_record_hash(payload_without_hash)
    if actual_hash != evidence_hash:
        raise ValueError(
            f"{role} content_hash does not match canonical evidence payload"
        )
    return PowerGridPRCArtifact(
        role=role,
        source_name=source_name,
        source_sha256=source_sha256,
        evidence_schema=evidence_schema,
        evidence_hash=evidence_hash,
        record=record,
    )


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` if it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
