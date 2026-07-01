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
    pypi_resolved_dependency,
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

    def test_require_str_mapping_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError, match="ver must be a mapping"):
            provenance._require_str_mapping(["x"], "ver")

    def test_require_str_mapping_rejects_non_string_keys_and_values(self) -> None:
        with pytest.raises(ValueError, match="ver keys must be strings"):
            provenance._require_str_mapping({1: "x"}, "ver")
        with pytest.raises(ValueError, match="ver values must be strings"):
            provenance._require_str_mapping({"k": 2}, "ver")

    def test_require_str_mapping_copies_mapping(self) -> None:
        source = {"python": "3.12"}
        result = provenance._require_str_mapping(source, "ver")
        assert result == source
        assert result is not source

    def test_descriptor_dicts_sorted_by_uri_then_name(self) -> None:
        descriptors = (
            ResourceDescriptor(uri="pkg:b", sha256=_B, name="z"),
            ResourceDescriptor(uri="pkg:a", sha256=_A, name="y"),
            ResourceDescriptor(uri="pkg:a", sha256=_C, name="x"),
        )
        result = provenance._descriptor_dicts(descriptors)
        assert [(item["uri"], item["name"]) for item in result] == [
            ("pkg:a", "x"),
            ("pkg:a", "y"),
            ("pkg:b", "z"),
        ]


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

    def test_default_run_details_omit_new_optional_blocks(self) -> None:
        record = _run_details().to_dict()
        assert record["builder"] == {"id": "https://github.com/anulum/spo/ci"}
        assert "byproducts" not in record

    def test_rejects_non_string_builder_version_value(self) -> None:
        with pytest.raises(ValueError, match="builder_version values must be strings"):
            _run_details(builder_version={"python": 3})

    def test_builder_version_included_when_present(self) -> None:
        builder = _run_details(builder_version={"python": "3.12"}).to_dict()["builder"]
        assert builder["version"] == {"python": "3.12"}

    def test_builder_dependencies_sorted_and_included(self) -> None:
        details = _run_details(
            builder_dependencies=(
                ResourceDescriptor(uri="pkg:b", sha256=_B),
                ResourceDescriptor(uri="pkg:a", sha256=_A),
            )
        )
        builder = details.to_dict()["builder"]
        uris = [dep["uri"] for dep in builder["builderDependencies"]]
        assert uris == ["pkg:a", "pkg:b"]

    def test_byproducts_sorted_and_included(self) -> None:
        details = _run_details(
            byproducts=(
                ResourceDescriptor(uri="pkg:sbom", sha256=_B, name="sbom.json"),
                ResourceDescriptor(uri="pkg:log", sha256=_A, name="build.log"),
            )
        )
        byproducts = details.to_dict()["byproducts"]
        assert [item["uri"] for item in byproducts] == ["pkg:log", "pkg:sbom"]


# --- pypi_resolved_dependency ---------------------------------------------


class TestPypiResolvedDependency:
    def test_normalises_name_for_purl_and_preserves_original(self) -> None:
        descriptor = pypi_resolved_dependency("Foo_Bar.Baz", "1.2.3", _A)
        assert descriptor.uri == "pkg:pypi/foo-bar-baz@1.2.3"
        assert descriptor.name == "Foo_Bar.Baz"
        assert descriptor.sha256 == _A

    def test_collapses_repeated_separators(self) -> None:
        descriptor = pypi_resolved_dependency("a--b__c", "0.1", _A)
        assert descriptor.uri == "pkg:pypi/a-b-c@0.1"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="dependency name"):
            pypi_resolved_dependency("", "1.0", _A)

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(ValueError, match="dependency version"):
            pypi_resolved_dependency("click", "", _A)

    def test_rejects_bad_digest(self) -> None:
        with pytest.raises(ValueError, match="resolved dependency sha256"):
            pypi_resolved_dependency("click", "8.1.7", "nope")

    def test_feeds_a_fuller_resolved_dependency_tree(self) -> None:
        definition = _build_definition(
            resolved_dependencies=(
                pypi_resolved_dependency("click", "8.1.7", _A),
                pypi_resolved_dependency("cryptography", "49.0.0", _B),
            )
        )
        deps = definition.to_dict()["resolvedDependencies"]
        assert [dep["uri"] for dep in deps] == [
            "pkg:pypi/click@8.1.7",
            "pkg:pypi/cryptography@49.0.0",
        ]


# --- backward compatibility -----------------------------------------------


class TestBackwardCompatibility:
    def test_statement_hash_unchanged_when_new_blocks_empty(self) -> None:
        # A statement built without any of the new optional run-detail blocks must
        # serialise byte-identically to the pre-extension format, so previously
        # issued attestations keep the same hash.
        statement = build_slsa_provenance_statement(
            (ArtifactSubject(name="wheel", sha256=_A),),
            _build_definition(),
            _run_details(),
        )
        record = statement.to_statement()
        assert record["predicate"]["runDetails"] == {
            "builder": {"id": "https://github.com/anulum/spo/ci"},
            "metadata": {"invocationId": "run-42"},
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
