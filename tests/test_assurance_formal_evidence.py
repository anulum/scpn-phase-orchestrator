# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — formal-verification-package assurance evidence tests

"""Tests for deriving assurance evidence from a formal-verification-package manifest.

``build_formal_verification_evidence`` maps a JSON-safe
``FormalVerificationPackage.to_audit_record()`` manifest into a single
``formal_verification`` evidence item, restating the manifest verbatim, and rejects
manifests that are missing a required field or carry a field of the wrong type.
"""

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.assurance.formal_evidence as _formal_evidence
from scpn_phase_orchestrator.assurance import (
    FORMAL_VERIFICATION,
    build_formal_verification_evidence,
)
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalSafetyProperty,
    FormalTextArtifact,
    build_formal_verification_package,
)

assert _formal_evidence is not None


def _package_manifest(
    *,
    artifacts: dict[str, FormalTextArtifact] | None = None,
    properties: list[FormalSafetyProperty] | None = None,
) -> dict[str, object]:
    """Return a realistic verification-package manifest for the tests."""
    if artifacts is None:
        artifacts = {"safety": FormalTextArtifact("smt2", "(assert (= x x))")}
    if properties is None:
        properties = [
            FormalSafetyProperty(
                name="bounded",
                artifact_name="safety",
                checker="smt",
                expression="(check-sat)",
            )
        ]
    package = build_formal_verification_package(artifacts, properties)
    return package.to_audit_record()


def test_builds_formal_evidence_from_real_package() -> None:
    manifest = _package_manifest()

    item = build_formal_verification_evidence(manifest)

    assert item.evidence_id == "formal-verification-package"
    assert item.category == FORMAL_VERIFICATION
    assert item.record == manifest
    assert "1 property over 1 artefact" in item.summary
    assert "spo-formal-verification" in item.summary


def test_summary_pluralises_multiple_properties_and_artefacts() -> None:
    manifest = _package_manifest(
        artifacts={
            "safe_a": FormalTextArtifact("smt2", "(assert true)"),
            "safe_b": FormalTextArtifact("smt2", "(assert false)"),
        },
        properties=[
            FormalSafetyProperty(
                name="prop_a",
                artifact_name="safe_a",
                checker="smt",
                expression="(check-sat)",
            ),
            FormalSafetyProperty(
                name="prop_b",
                artifact_name="safe_b",
                checker="smt",
                expression="(check-sat)",
            ),
        ],
    )

    item = build_formal_verification_evidence(manifest)

    assert "2 properties over 2 artefacts" in item.summary


def test_evidence_content_hash_is_deterministic() -> None:
    manifest = _package_manifest()

    first = build_formal_verification_evidence(manifest)
    second = build_formal_verification_evidence(manifest)

    assert first.content_hash == second.content_hash


def test_rejects_non_mapping_manifest() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        build_formal_verification_evidence([])  # type: ignore[arg-type]


def test_rejects_missing_package_name() -> None:
    manifest = _package_manifest()
    del manifest["package_name"]

    with pytest.raises(ValueError, match="package_name"):
        build_formal_verification_evidence(manifest)


def test_rejects_blank_package_hash() -> None:
    manifest = _package_manifest()
    manifest["package_hash"] = "   "

    with pytest.raises(ValueError, match="package_hash"):
        build_formal_verification_evidence(manifest)


def test_rejects_non_list_properties() -> None:
    manifest = _package_manifest()
    manifest["properties"] = "not-a-list"

    with pytest.raises(ValueError, match="'properties' must be a list"):
        build_formal_verification_evidence(manifest)


def test_rejects_non_mapping_artifact_hashes() -> None:
    manifest = _package_manifest()
    manifest["artifact_hashes"] = ["safety"]

    with pytest.raises(ValueError, match="'artifact_hashes' must be a mapping"):
        build_formal_verification_evidence(manifest)
