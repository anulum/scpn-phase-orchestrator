# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance evidence from a formal-verification package

"""Derive formal-verification assurance evidence from a verification-package manifest.

The supervisor formal exporters
(:mod:`scpn_phase_orchestrator.supervisor.formal_export`) assemble a deterministic
:class:`~scpn_phase_orchestrator.supervisor.formal_export.FormalVerificationPackage`
— exported PRISM/TLA/SMT artefact hashes, the model-checking property library, and
the exact (non-executing) checker commands — whose
``to_audit_record()`` is a JSON-safe manifest. This module maps that manifest into a
single :class:`~scpn_phase_orchestrator.assurance.evidence.EvidenceItem` in the
``formal_verification`` category, so a conformity package can attest the formal
argument the supervisor produced.

The helper consumes the serialised manifest (a ``Mapping``), not the
``FormalVerificationPackage`` object, mirroring
:func:`~scpn_phase_orchestrator.assurance.run_evidence.build_run_evidence`: the
assurance package stays free of the supervisor import chain and can attest a
manifest persisted to disk. It restates the manifest verbatim and never fabricates
properties or checker results — it records which properties were posed against which
artefacts, not that any checker accepted them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scpn_phase_orchestrator.assurance.evidence import (
    FORMAL_VERIFICATION,
    EvidenceItem,
    build_evidence_item,
)


def _require_non_empty_str(record: Mapping[str, object], key: str) -> str:
    """Return ``record[key]`` as a non-empty string, else raise ``ValueError``."""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"formal verification package manifest field {key!r} "
            "must be a non-empty string"
        )
    return value


def build_formal_verification_evidence(
    package_record: Mapping[str, object],
) -> EvidenceItem:
    """Build a formal-verification evidence item from a verification-package manifest.

    Parameters
    ----------
    package_record:
        A JSON-safe ``FormalVerificationPackage.to_audit_record()`` mapping. It
        must carry a non-empty ``package_name`` and ``package_hash``, a list of
        ``properties``, and an ``artifact_hashes`` mapping.

    Returns
    -------
    EvidenceItem
        A ``formal_verification`` evidence item whose record is the manifest
        verbatim, summarising how many properties were posed against how many
        exported artefacts.

    Raises
    ------
    ValueError
        If the manifest is missing a required field or a field has the wrong type.
    """
    if not isinstance(package_record, Mapping):
        raise ValueError("formal verification package manifest must be a mapping")

    package_name = _require_non_empty_str(package_record, "package_name")
    _require_non_empty_str(package_record, "package_hash")

    properties = package_record.get("properties")
    if not isinstance(properties, Sequence) or isinstance(properties, str | bytes):
        raise ValueError(
            "formal verification package manifest field 'properties' must be a list"
        )
    artifact_hashes = package_record.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError(
            "formal verification package manifest field 'artifact_hashes' "
            "must be a mapping"
        )

    summary = (
        f"Formal verification package {package_name!r}: "
        f"{len(properties)} propert{'y' if len(properties) == 1 else 'ies'} "
        f"over {len(artifact_hashes)} "
        f"artefact{'' if len(artifact_hashes) == 1 else 's'}"
    )
    return build_evidence_item(
        evidence_id="formal-verification-package",
        category=FORMAL_VERIFICATION,
        summary=summary,
        record=dict(package_record),
    )


__all__ = ["build_formal_verification_evidence"]
