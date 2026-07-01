# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SLSA v1 build-provenance statement (in-toto)

"""Build a deterministic SLSA v1 provenance statement for released artefacts.

The audit-chain seal (:mod:`scpn_phase_orchestrator.runtime.audit_pqc`) and the
certification envelope (:mod:`scpn_phase_orchestrator.assurance.envelope`) both
attest to a *run*: what the software did once it executed. This module attests to
the *build*: which artefacts were produced, from which resolved inputs, by which
builder. That is the supply-chain provenance a downstream consumer needs to answer
"is this wheel the one that came out of the declared build, and nothing else?".

The output is an `in-toto Statement v1
<https://in-toto.io/Statement/v1>`_ carrying a `SLSA provenance v1
<https://slsa.dev/provenance/v1>`_ predicate:

* ``subject`` — the produced artefacts, each pinned by SHA-256 digest;
* ``predicate.buildDefinition`` — the build type, the external parameters that drove
  it, optional internal parameters, and the resolved dependencies (source commit,
  toolchain), each itself digest-pinned;
* ``predicate.runDetails`` — the builder identity and the invocation metadata.

Everything is derived from caller-supplied values only — there are no wall-clock
reads, environment probes, or network calls — so the same build inputs always
serialise to the same statement and the same :func:`provenance_statement_hash`.
Signing lives in :mod:`scpn_phase_orchestrator.assurance.dsse`; this module produces
the payload that gets signed. The statement makes no live-actuation or conformity
claim: it is a factual record of a build, meeting the *content* obligation of SLSA
Build Level 2 (signed provenance describing the build), with the signing and
provenance-generation obligations met by the DSSE layer and the hosting build
service respectively.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)

IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SLSA_PROVENANCE_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"


def _require_non_empty_str(value: object, field_name: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_json_mapping(value: object, field_name: str) -> dict[str, object]:
    """Return a shallow copy of ``value`` if it is a string-keyed mapping, else raise.

    The parameter blocks of a SLSA statement are free-form JSON objects, but their
    keys must be strings for the canonical serialisation to be deterministic.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class ArtifactSubject:
    """A produced artefact pinned by its SHA-256 digest.

    Attributes
    ----------
    name:
        The artefact name (e.g. the wheel filename), non-empty.
    sha256:
        The artefact's SHA-256 digest, lowercase hex (64 characters).
    """

    name: str
    sha256: str

    def __post_init__(self) -> None:
        """Validate the subject name and digest."""
        _require_non_empty_str(self.name, "subject name")
        require_sha256(self.sha256, "subject sha256")

    def to_dict(self) -> dict[str, object]:
        """Return the in-toto subject mapping (``name`` + ``digest.sha256``)."""
        return {"name": self.name, "digest": {"sha256": self.sha256}}


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    """A resolved build input pinned by digest (source commit, toolchain, dependency).

    Attributes
    ----------
    uri:
        The resource locator (e.g. ``git+https://…@<commit>`` or a package URL),
        non-empty.
    sha256:
        The resource's SHA-256 digest, lowercase hex (64 characters).
    name:
        An optional human-readable name; the empty string omits it from the output.
    """

    uri: str
    sha256: str
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the resource locator and digest."""
        _require_non_empty_str(self.uri, "resolved dependency uri")
        require_sha256(self.sha256, "resolved dependency sha256")

    def to_dict(self) -> dict[str, object]:
        """Return the in-toto resource-descriptor mapping."""
        descriptor: dict[str, object] = {
            "uri": self.uri,
            "digest": {"sha256": self.sha256},
        }
        if self.name:
            descriptor["name"] = self.name
        return descriptor


@dataclass(frozen=True, slots=True)
class BuildDefinition:
    """The reproducible definition of how the artefacts were built.

    Attributes
    ----------
    build_type:
        A URI naming the build type / recipe convention, non-empty.
    external_parameters:
        The externally supplied parameters that fully drove the build (JSON object).
    internal_parameters:
        Builder-internal parameters, defaulting to an empty object.
    resolved_dependencies:
        The digest-pinned inputs the build resolved (source, toolchain, deps).
    """

    build_type: str
    external_parameters: Mapping[str, object]
    internal_parameters: Mapping[str, object] = field(default_factory=dict)
    resolved_dependencies: tuple[ResourceDescriptor, ...] = ()

    def __post_init__(self) -> None:
        """Validate the build type and parameter blocks."""
        _require_non_empty_str(self.build_type, "build_type")
        _require_json_mapping(self.external_parameters, "external_parameters")
        _require_json_mapping(self.internal_parameters, "internal_parameters")

    def to_dict(self) -> dict[str, object]:
        """Return the SLSA ``buildDefinition`` mapping.

        ``resolvedDependencies`` is sorted by ``(uri, name)`` so the same set of
        inputs always serialises identically. ``internalParameters`` is omitted when
        empty to keep the statement minimal.
        """
        definition: dict[str, object] = {
            "buildType": self.build_type,
            "externalParameters": _require_json_mapping(
                self.external_parameters, "external_parameters"
            ),
        }
        internal = _require_json_mapping(
            self.internal_parameters, "internal_parameters"
        )
        if internal:
            definition["internalParameters"] = internal
        definition["resolvedDependencies"] = [
            descriptor.to_dict()
            for descriptor in sorted(
                self.resolved_dependencies, key=lambda item: (item.uri, item.name)
            )
        ]
        return definition


@dataclass(frozen=True, slots=True)
class RunDetails:
    """The identity of the builder and the invocation that produced the artefacts.

    Attributes
    ----------
    builder_id:
        A URI identifying the build platform / builder, non-empty.
    invocation_id:
        A stable identifier for this build invocation, non-empty.
    started_on:
        Optional RFC 3339 build-start timestamp; the empty string omits it. Supplied
        by the caller (never read from the wall clock) to preserve determinism.
    finished_on:
        Optional RFC 3339 build-finish timestamp; the empty string omits it.
    """

    builder_id: str
    invocation_id: str
    started_on: str = ""
    finished_on: str = ""

    def __post_init__(self) -> None:
        """Validate the builder identity and invocation id."""
        _require_non_empty_str(self.builder_id, "builder_id")
        _require_non_empty_str(self.invocation_id, "invocation_id")

    def to_dict(self) -> dict[str, object]:
        """Return the SLSA ``runDetails`` mapping.

        The ``metadata`` block always carries ``invocationId`` and adds
        ``startedOn`` / ``finishedOn`` only when the caller supplied them.
        """
        metadata: dict[str, object] = {"invocationId": self.invocation_id}
        if self.started_on:
            metadata["startedOn"] = self.started_on
        if self.finished_on:
            metadata["finishedOn"] = self.finished_on
        return {"builder": {"id": self.builder_id}, "metadata": metadata}


@dataclass(frozen=True, slots=True)
class SlsaProvenanceStatement:
    """A complete in-toto Statement v1 carrying a SLSA provenance v1 predicate.

    Attributes
    ----------
    subjects:
        The produced artefacts, each digest-pinned; must be non-empty.
    build_definition:
        How the artefacts were built.
    run_details:
        Who built them and under which invocation.
    """

    subjects: tuple[ArtifactSubject, ...]
    build_definition: BuildDefinition
    run_details: RunDetails

    def __post_init__(self) -> None:
        """Reject an empty subject list; the sub-objects self-validate."""
        if not self.subjects:
            raise ValueError("provenance statement requires at least one subject")

    def to_statement(self) -> dict[str, object]:
        """Return the JSON-safe in-toto Statement.

        ``subject`` is sorted by artefact name so a given set of artefacts always
        serialises identically.

        Returns
        -------
        dict[str, object]
            The ``_type`` / ``subject`` / ``predicateType`` / ``predicate`` mapping,
            ready to canonicalise, hash, and wrap in a DSSE envelope.
        """
        return {
            "_type": IN_TOTO_STATEMENT_TYPE,
            "subject": [
                subject.to_dict()
                for subject in sorted(self.subjects, key=lambda item: item.name)
            ],
            "predicateType": SLSA_PROVENANCE_PREDICATE_TYPE,
            "predicate": {
                "buildDefinition": self.build_definition.to_dict(),
                "runDetails": self.run_details.to_dict(),
            },
        }

    def statement_hash(self) -> str:
        """Return the SHA-256 of the canonical statement serialisation."""
        return canonical_record_hash(self.to_statement())


def build_slsa_provenance_statement(
    subjects: tuple[ArtifactSubject, ...],
    build_definition: BuildDefinition,
    run_details: RunDetails,
) -> SlsaProvenanceStatement:
    """Assemble a validated SLSA provenance statement.

    Parameters
    ----------
    subjects:
        The produced artefacts (non-empty), each digest-pinned.
    build_definition:
        The reproducible build definition.
    run_details:
        The builder identity and invocation metadata.

    Returns
    -------
    SlsaProvenanceStatement
        The validated statement.

    Raises
    ------
    ValueError
        If the subject list is empty or any field is malformed.
    """
    return SlsaProvenanceStatement(
        subjects=subjects,
        build_definition=build_definition,
        run_details=run_details,
    )


def provenance_statement_hash(statement: SlsaProvenanceStatement) -> str:
    """Return the canonical SHA-256 digest of a provenance statement.

    Parameters
    ----------
    statement:
        The statement to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 over the canonical statement JSON.
    """
    return statement.statement_hash()


__all__ = [
    "IN_TOTO_STATEMENT_TYPE",
    "SLSA_PROVENANCE_PREDICATE_TYPE",
    "ArtifactSubject",
    "BuildDefinition",
    "ResourceDescriptor",
    "RunDetails",
    "SlsaProvenanceStatement",
    "build_slsa_provenance_statement",
    "provenance_statement_hash",
]
