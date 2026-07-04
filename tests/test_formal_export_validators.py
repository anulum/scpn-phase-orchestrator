# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal exporter dataclass and builder validators

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    export_policy_rules_prism,
)
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalCheckerAvailability,
    FormalCheckerCommand,
    FormalCheckerResult,
    FormalRuntimeCertificate,
    FormalSafetyProperty,
    FormalTextArtifact,
    FormalVerificationPackage,
    audit_formal_checker_availability,
    build_formal_verification_package,
    build_runtime_control_certificate,
)

_DIGEST = "a" * 64


def _smt_package(package_name: str = "spo_pkg") -> FormalVerificationPackage:
    artifact = FormalTextArtifact(artifact_type="smt2", text="(assert true)")
    prop = FormalSafetyProperty(
        name="prop_a",
        artifact_name="art_a",
        checker="smt",
        expression="unsat",
    )
    return build_formal_verification_package(
        {"art_a": artifact},
        (prop,),
        package_name=package_name,
    )


def _availability(**changes: object) -> FormalCheckerAvailability:
    base: dict[str, object] = {
        "property_name": "prop_a",
        "checker": "smt",
        "artifact_name": "art_a",
        "executable": "z3",
        "command": ("z3", "art_a.smt2"),
        "available": True,
        "resolved_path": "/usr/bin/z3",
        "status": "ready_not_executed",
    }
    base.update(changes)
    return FormalCheckerAvailability(**base)  # type: ignore[arg-type]


def _result(package_hash: str, **changes: object) -> FormalCheckerResult:
    base: dict[str, object] = {
        "property_name": "prop_a",
        "checker": "smt",
        "artifact_name": "art_a",
        "package_hash": package_hash,
        "result_hash": _DIGEST,
        "status": "passed",
        "passed": True,
    }
    base.update(changes)
    return FormalCheckerResult(**base)  # type: ignore[arg-type]


class _DuplicateRuntimeBounds(Mapping[str, object]):
    """Mapping fixture that exposes duplicate keys through ``items()``."""

    def __getitem__(self, key: str) -> object:
        if key == "R_min":
            return 0.5
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("R_min", "R_min"))

    def __len__(self) -> int:
        return 2


# --- FormalTextArtifact ----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"artifact_type": "xml", "text": "ok"}, "type must be 'promela' or 'smt2'"),
        ({"artifact_type": "smt2", "text": "   "}, "must contain non-empty text"),
        ({"artifact_type": "smt2", "text": "ok\x00"}, "unsafe control characters"),
    ],
)
def test_formal_text_artifact_rejects_corruptions(kwargs, match) -> None:
    with pytest.raises(PolicyError, match=match):
        FormalTextArtifact(**kwargs)


def test_formal_text_artifact_accepts_whitespace_control_characters() -> None:
    artifact = FormalTextArtifact(artifact_type="promela", text="ltl { [] p }\n\t")
    assert artifact.artifact_type == "promela"


# --- FormalSafetyProperty --------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"checker": "z3", "expression": "unsat"},
            "checker must be 'prism', 'tlc', 'spin', or 'smt'",
        ),
        ({"checker": "smt", "expression": "  "}, "expression must be a non-empty"),
        (
            {"checker": "smt", "expression": "unsat", "description": "bad\x01"},
            "description must not contain control characters",
        ),
    ],
)
def test_formal_safety_property_rejects_corruptions(kwargs, match) -> None:
    base = {"name": "prop_a", "artifact_name": "art_a"}
    base.update(kwargs)
    with pytest.raises(PolicyError, match=match):
        FormalSafetyProperty(**base)  # type: ignore[arg-type]


# --- FormalCheckerCommand --------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"checker": "z3"}, "checker must be 'prism', 'tlc', 'spin', or 'smt'"),
        ({"command": ()}, "checker command must not be empty"),
        ({"command": ("z3", "  ")}, "command parts must be non-empty strings"),
        ({"command": ("z3", "art\x00")}, "must not contain control characters"),
    ],
)
def test_formal_checker_command_rejects_corruptions(kwargs, match) -> None:
    base: dict[str, object] = {
        "property_name": "prop_a",
        "checker": "smt",
        "artifact_name": "art_a",
        "command": ("z3", "art_a.smt2"),
    }
    base.update(kwargs)
    with pytest.raises(PolicyError, match=match):
        FormalCheckerCommand(**base)  # type: ignore[arg-type]


# --- FormalCheckerAvailability ---------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"checker": "z3"}, "availability checker must be"),
        ({"executable": "  "}, "executable must be a non-empty string"),
        ({"executable": "z3\x00"}, "executable must not contain control characters"),
        (
            {"resolved_path": "  "},
            "resolved_path must be None or a non-empty safe string",
        ),
        (
            {"available": True, "status": "queued"},
            "availability status is unsupported",
        ),
        ({"command": ()}, "availability command must not be empty"),
        ({"command": ("z3", "  ")}, "command parts must be non-empty strings"),
        ({"command": ("z3", "x\x00")}, "command parts must not contain control"),
    ],
)
def test_formal_checker_availability_rejects_corruptions(kwargs, match) -> None:
    with pytest.raises(PolicyError, match=match):
        _availability(**kwargs)


# --- FormalCheckerResult ---------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"checker": "z3"}, "checker result checker must be"),
        (
            {"status": "errored", "passed": False},
            "checker result status is unsupported",
        ),
        ({"status": "passed", "passed": "yes"}, "passed flag must be a boolean"),
        (
            {"status": "passed", "passed": True, "detail": "bad\x00"},
            "detail must not contain control characters",
        ),
    ],
)
def test_formal_checker_result_rejects_corruptions(kwargs, match) -> None:
    with pytest.raises(PolicyError, match=match):
        _result(_DIGEST, **kwargs)


# --- Identifier / bounds helpers -------------------------------------------


def test_runtime_certificate_rejects_non_tuple_identifier_collections() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="must be a tuple"):
        FormalRuntimeCertificate(
            certificate_name="cert_a",
            package_name=package.package_name,
            package_hash=package.package_hash,
            runtime_bounds={"R_min": 0.5},
            checker_availability=(),
            checker_results=(),
            required_property_count=0,
            passed_required_count=0,
            missing_required_properties=["prop_a"],  # type: ignore[arg-type]
            failed_required_properties=(),
            unavailable_checker_properties=(),
            status="blocked",
            certificate_hash=_DIGEST,
        )


def test_runtime_certificate_rejects_duplicate_identifier_entries() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="must not contain duplicates"):
        FormalRuntimeCertificate(
            certificate_name="cert_a",
            package_name=package.package_name,
            package_hash=package.package_hash,
            runtime_bounds={"R_min": 0.5},
            checker_availability=(),
            checker_results=(),
            required_property_count=0,
            passed_required_count=0,
            missing_required_properties=("prop_a", "prop_a"),
            failed_required_properties=(),
            unavailable_checker_properties=(),
            status="blocked",
            certificate_hash=_DIGEST,
        )


@pytest.mark.parametrize(
    ("bounds", "match"),
    [
        ({}, "runtime bounds must be a non-empty mapping"),
        ({"R_min": True}, "runtime bounds must be finite real numbers"),
        ({"R_min": float("inf")}, "runtime bounds must be finite real numbers"),
    ],
)
def test_runtime_certificate_rejects_malformed_bounds(bounds, match) -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match=match):
        build_runtime_control_certificate(package, (), (), bounds)


def test_runtime_certificate_rejects_duplicate_runtime_bound_names() -> None:
    package = _smt_package()
    with pytest.raises(
        PolicyError,
        match="runtime bounds must not contain duplicate names",
    ):
        build_runtime_control_certificate(
            package,
            (),
            (),
            _DuplicateRuntimeBounds(),
        )


# --- FormalRuntimeCertificate __post_init__ branches -----------------------


def _certificate(**changes: object) -> FormalRuntimeCertificate:
    package = _smt_package()
    base: dict[str, object] = {
        "certificate_name": "cert_a",
        "package_name": package.package_name,
        "package_hash": package.package_hash,
        "runtime_bounds": {"R_min": 0.5},
        "checker_availability": (),
        "checker_results": (),
        "required_property_count": 0,
        "passed_required_count": 0,
        "missing_required_properties": (),
        "failed_required_properties": (),
        "unavailable_checker_properties": (),
        "status": "blocked",
        "certificate_hash": _DIGEST,
    }
    base.update(changes)
    return FormalRuntimeCertificate(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"checker_availability": ("nope",)}, "must contain availability records"),
        ({"checker_results": ("nope",)}, "must contain result records"),
        (
            {"required_property_count": -1},
            "required_property_count must be non-negative",
        ),
        ({"passed_required_count": True}, "passed_required_count must be non-negative"),
        (
            {"required_property_count": 1, "passed_required_count": 2},
            "must not exceed required count",
        ),
        ({"status": "approved"}, "certificate status is unsupported"),
        (
            {
                "status": "verified_non_actuating",
                "missing_required_properties": ("prop_a",),
            },
            "must have complete passed evidence",
        ),
        ({"actuation_permitted": True}, "must remain non-actuating"),
    ],
)
def test_runtime_certificate_post_init_rejects_corruptions(changes, match) -> None:
    with pytest.raises(PolicyError, match=match):
        _certificate(**changes)


# --- audit_formal_checker_availability -------------------------------------


def test_audit_requires_a_formal_package() -> None:
    with pytest.raises(PolicyError, match="audit requires a formal package"):
        audit_formal_checker_availability("not-a-package")  # type: ignore[arg-type]


def test_audit_resolves_host_path_without_injected_executables() -> None:
    package = _smt_package()
    records = audit_formal_checker_availability(package)
    assert len(records) == 1
    record = records[0]
    assert record.property_name == "prop_a"
    assert record.execution_permitted is False
    # Resolution status mirrors whether the host actually exposes ``z3``.
    assert record.available is (record.resolved_path is not None)


# --- build_runtime_control_certificate evidence contracts ------------------


def test_certificate_builder_requires_a_formal_package() -> None:
    with pytest.raises(PolicyError, match="certificate requires a formal package"):
        build_runtime_control_certificate(
            "not-a-package",  # type: ignore[arg-type]
            (),
            (),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_non_record_availability() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="availability records are required"):
        build_runtime_control_certificate(
            package,
            ("nope",),
            (),
            {"R_min": 0.5},  # type: ignore[arg-type]
        )


def test_certificate_builder_rejects_unknown_availability_property() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="availability references unknown property"):
        build_runtime_control_certificate(
            package,
            (_availability(property_name="ghost"),),
            (),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_duplicate_availability_property() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="duplicate checker availability property"):
        build_runtime_control_certificate(
            package,
            (_availability(), _availability()),
            (),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_mismatched_availability() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="availability does not match package"):
        build_runtime_control_certificate(
            package,
            (_availability(checker="spin", artifact_name="art_a"),),
            (),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_non_record_results() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="result records are required"):
        build_runtime_control_certificate(
            package,
            (_availability(),),
            ("nope",),  # type: ignore[arg-type]
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_unknown_result_property() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="result references unknown property"):
        build_runtime_control_certificate(
            package,
            (_availability(),),
            (_result(package.package_hash, property_name="ghost"),),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_duplicate_result_property() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="duplicate checker result property"):
        build_runtime_control_certificate(
            package,
            (_availability(),),
            (_result(package.package_hash), _result(package.package_hash)),
            {"R_min": 0.5},
        )


def test_certificate_builder_rejects_mismatched_result() -> None:
    package = _smt_package()
    with pytest.raises(PolicyError, match="result does not match package"):
        build_runtime_control_certificate(
            package,
            (_availability(),),
            (_result(package.package_hash, checker="spin"),),
            {"R_min": 0.5},
        )


def test_certificate_builder_verifies_complete_passed_evidence() -> None:
    package = _smt_package()
    certificate = build_runtime_control_certificate(
        package,
        (_availability(),),
        (_result(package.package_hash),),
        {"R_min": 0.5},
    )
    assert certificate.status == "verified_non_actuating"
    assert certificate.actuation_permitted is False
    assert certificate.passed_required_count == 1


# --- build_formal_verification_package artifact contracts ------------------


def test_package_builder_rejects_non_export_artifact() -> None:
    prop = FormalSafetyProperty(
        name="prop_a",
        artifact_name="art_a",
        checker="smt",
        expression="unsat",
    )
    with pytest.raises(PolicyError, match="must be PrismExport, TLAExport"):
        build_formal_verification_package(
            {"art_a": "raw text"},  # type: ignore[dict-item]
            (prop,),
        )


def test_package_builder_rejects_empty_artifact_text() -> None:
    empty = FormalTextArtifact.__new__(FormalTextArtifact)
    object.__setattr__(empty, "artifact_type", "smt2")
    object.__setattr__(empty, "text", "   ")
    prop = FormalSafetyProperty(
        name="prop_a",
        artifact_name="art_a",
        checker="smt",
        expression="unsat",
    )
    with pytest.raises(PolicyError, match="is empty"):
        build_formal_verification_package({"art_a": empty}, (prop,))


# --- compound policy logic guard -------------------------------------------


def test_export_rejects_non_string_compound_logic() -> None:
    rule = PolicyRule(
        name="bad_logic_type",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            [PolicyCondition("R", 0, ">", 0.5)],
            logic=123,  # type: ignore[arg-type]
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        max_fires=2,
    )
    with pytest.raises(PolicyError, match="logic must be AND or OR"):
        export_policy_rules_prism([rule])
