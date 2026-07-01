# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SLSA provenance statement tests

"""Coverage for the deterministic SLSA v1 provenance statement builder.

The statement carries no crypto, so every path here runs without the ``pqc`` extra:
field validation on each dataclass, the omit-when-empty behaviour of optional blocks,
the deterministic sorting of subjects and resolved dependencies, and the stability of
the canonical statement hash under input reordering.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.assurance import provenance
from scpn_phase_orchestrator.assurance.provenance import (
    IN_TOTO_STATEMENT_TYPE,
    SLSA_PROVENANCE_PREDICATE_TYPE,
    ArtifactSubject,
    BuildDefinition,
    ResourceDescriptor,
    RunDetails,
    SlsaProvenanceStatement,
    build_slsa_provenance_statement,
    provenance_statement_hash,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _build_definition(**overrides: object) -> BuildDefinition:
    params: dict[str, object] = {
        "build_type": "https://slsa.dev/build/pypi@v1",
        "external_parameters": {"ref": "refs/tags/v1.0"},
    }
    params.update(overrides)
    return BuildDefinition(**params)  # type: ignore[arg-type]


def _run_details(**overrides: object) -> RunDetails:
    params: dict[str, object] = {
        "builder_id": "https://github.com/anulum/spo/ci",
        "invocation_id": "run-42",
    }
    params.update(overrides)
    return RunDetails(**params)  # type: ignore[arg-type]


# --- private validators ---------------------------------------------------


class TestValidators:
    def test_require_non_empty_str_rejects_empty_and_non_string(self) -> None:
        with pytest.raises(ValueError, match="widget must be a non-empty string"):
            provenance._require_non_empty_str("", "widget")
        with pytest.raises(ValueError, match="widget must be a non-empty string"):
            provenance._require_non_empty_str(3, "widget")

    def test_require_json_mapping_rejects_non_mapping_and_non_string_keys(self) -> None:
        with pytest.raises(ValueError, match="block must be a mapping"):
            provenance._require_json_mapping(["x"], "block")
        with pytest.raises(ValueError, match="block keys must be strings"):
            provenance._require_json_mapping({1: "x"}, "block")

    def test_require_json_mapping_copies_mapping(self) -> None:
        source = {"a": 1}
        result = provenance._require_json_mapping(source, "block")
        assert result == source
        assert result is not source


# --- ArtifactSubject ------------------------------------------------------


class TestArtifactSubject:
    def test_valid_subject_to_dict(self) -> None:
        subject = ArtifactSubject(name="wheel", sha256=_A)
        assert subject.to_dict() == {"name": "wheel", "digest": {"sha256": _A}}

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="subject name"):
            ArtifactSubject(name="", sha256=_A)

    def test_rejects_bad_digest(self) -> None:
        with pytest.raises(ValueError, match="subject sha256"):
            ArtifactSubject(name="wheel", sha256="tooshort")


# --- ResourceDescriptor ---------------------------------------------------


class TestResourceDescriptor:
    def test_to_dict_includes_name_when_present(self) -> None:
        descriptor = ResourceDescriptor(uri="pkg:pypi/x@1", sha256=_A, name="x")
        assert descriptor.to_dict() == {
            "uri": "pkg:pypi/x@1",
            "digest": {"sha256": _A},
            "name": "x",
        }

    def test_to_dict_omits_empty_name(self) -> None:
        descriptor = ResourceDescriptor(uri="pkg:pypi/x@1", sha256=_A)
        assert "name" not in descriptor.to_dict()

    def test_rejects_empty_uri(self) -> None:
        with pytest.raises(ValueError, match="resolved dependency uri"):
            ResourceDescriptor(uri="", sha256=_A)

    def test_rejects_bad_digest(self) -> None:
        with pytest.raises(ValueError, match="resolved dependency sha256"):
            ResourceDescriptor(uri="pkg:pypi/x@1", sha256="nope")


# --- BuildDefinition ------------------------------------------------------


class TestBuildDefinition:
    def test_rejects_empty_build_type(self) -> None:
        with pytest.raises(ValueError, match="build_type"):
            _build_definition(build_type="")

    def test_rejects_non_mapping_external_parameters(self) -> None:
        with pytest.raises(ValueError, match="external_parameters must be a mapping"):
            _build_definition(external_parameters=["not", "a", "map"])

    def test_to_dict_omits_empty_internal_parameters(self) -> None:
        definition = _build_definition()
        assert "internalParameters" not in definition.to_dict()

    def test_to_dict_includes_internal_parameters_when_present(self) -> None:
        definition = _build_definition(internal_parameters={"runner": "ubuntu"})
        assert definition.to_dict()["internalParameters"] == {"runner": "ubuntu"}

    def test_resolved_dependencies_sorted_by_uri_then_name(self) -> None:
        definition = _build_definition(
            resolved_dependencies=(
                ResourceDescriptor(uri="pkg:b", sha256=_B),
                ResourceDescriptor(uri="pkg:a", sha256=_A),
            )
        )
        uris = [dep["uri"] for dep in definition.to_dict()["resolvedDependencies"]]
        assert uris == ["pkg:a", "pkg:b"]


# --- RunDetails -----------------------------------------------------------


class TestRunDetails:
    def test_rejects_empty_builder_id(self) -> None:
        with pytest.raises(ValueError, match="builder_id"):
            _run_details(builder_id="")

    def test_rejects_empty_invocation_id(self) -> None:
        with pytest.raises(ValueError, match="invocation_id"):
            _run_details(invocation_id="")

    def test_metadata_omits_timestamps_by_default(self) -> None:
        metadata = _run_details().to_dict()["metadata"]
        assert metadata == {"invocationId": "run-42"}

    def test_metadata_includes_supplied_timestamps(self) -> None:
        details = _run_details(
            started_on="2026-07-01T00:00:00Z", finished_on="2026-07-01T00:05:00Z"
        )
        metadata = details.to_dict()["metadata"]
        assert metadata["startedOn"] == "2026-07-01T00:00:00Z"
        assert metadata["finishedOn"] == "2026-07-01T00:05:00Z"

    def test_builder_block_carries_id(self) -> None:
        assert _run_details().to_dict()["builder"] == {
            "id": "https://github.com/anulum/spo/ci"
        }


# --- SlsaProvenanceStatement ----------------------------------------------


class TestStatement:
    def test_rejects_empty_subjects(self) -> None:
        with pytest.raises(ValueError, match="at least one subject"):
            SlsaProvenanceStatement(
                subjects=(),
                build_definition=_build_definition(),
                run_details=_run_details(),
            )

    def test_statement_shape(self) -> None:
        statement = build_slsa_provenance_statement(
            (ArtifactSubject(name="wheel", sha256=_A),),
            _build_definition(),
            _run_details(),
        )
        record = statement.to_statement()
        assert record["_type"] == IN_TOTO_STATEMENT_TYPE
        assert record["predicateType"] == SLSA_PROVENANCE_PREDICATE_TYPE
        predicate = record["predicate"]
        assert "buildDefinition" in predicate
        assert "runDetails" in predicate

    def test_subjects_sorted_by_name(self) -> None:
        statement = build_slsa_provenance_statement(
            (
                ArtifactSubject(name="zeta", sha256=_B),
                ArtifactSubject(name="alpha", sha256=_A),
            ),
            _build_definition(),
            _run_details(),
        )
        names = [subject["name"] for subject in statement.to_statement()["subject"]]
        assert names == ["alpha", "zeta"]

    def test_hash_is_stable_under_subject_reordering(self) -> None:
        first = build_slsa_provenance_statement(
            (
                ArtifactSubject(name="alpha", sha256=_A),
                ArtifactSubject(name="zeta", sha256=_B),
            ),
            _build_definition(),
            _run_details(),
        )
        second = build_slsa_provenance_statement(
            (
                ArtifactSubject(name="zeta", sha256=_B),
                ArtifactSubject(name="alpha", sha256=_A),
            ),
            _build_definition(),
            _run_details(),
        )
        assert provenance_statement_hash(first) == provenance_statement_hash(second)
        assert first.statement_hash() == provenance_statement_hash(first)

    def test_hash_changes_when_a_digest_changes(self) -> None:
        base = build_slsa_provenance_statement(
            (ArtifactSubject(name="wheel", sha256=_A),),
            _build_definition(),
            _run_details(),
        )
        changed = build_slsa_provenance_statement(
            (ArtifactSubject(name="wheel", sha256=_C),),
            _build_definition(),
            _run_details(),
        )
        assert base.statement_hash() != changed.statement_hash()
