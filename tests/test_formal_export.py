# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal exporter tests

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator import supervisor as supervisor_api
from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.supervisor import (
    CompoundCondition,
    FormalCheckerAvailability,
    FormalCheckerCommand,
    FormalCheckerResult,
    FormalRuntimeCertificate,
    FormalSafetyProperty,
    FormalTextArtifact,
    FormalVerificationPackage,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    PolicySTLSpec,
    audit_formal_checker_availability,
    build_formal_verification_package,
    build_runtime_control_certificate,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)
from scpn_phase_orchestrator.supervisor.formal_export import PrismExport, TLAExport
from scpn_phase_orchestrator.supervisor.formal_export.smt_export import (
    export_policy_rules_smt,
)
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)


def _net() -> PetriNet:
    return PetriNet(
        [
            Place("warmup"),
            Place("nominal"),
            Place("cool-down"),
            Place("done"),
        ],
        [
            Transition(
                name="start",
                inputs=[Arc("warmup")],
                outputs=[Arc("nominal")],
                guard=Guard("stability.proxy", ">", 0.6),
            ),
            Transition(
                name="wind down",
                inputs=[Arc("nominal", weight=2)],
                outputs=[Arc("cool-down")],
                guard=Guard("R_bad.0", "<=", 0.3),
            ),
            Transition(
                name="finish",
                inputs=[Arc("cool-down")],
                outputs=[Arc("done")],
            ),
        ],
    )


def test_petri_net_prism_export_serialises_guards_and_arcs() -> None:
    export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )

    assert isinstance(export, PrismExport)
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "stability.proxy": "stability_proxy",
    }
    assert "mdp\n" in export.model
    assert "module supervisor_net" in export.model
    assert "const double R_bad_0;" in export.model
    assert "const double stability_proxy;" in export.model
    assert "warmup : [0..2] init 1;" in export.model
    assert "nominal : [0..2] init 2;" in export.model
    assert "[start] stability_proxy > 0.59999999999999998 & warmup >= 1" in (
        export.model
    )
    assert "[wind_down] R_bad_0 <= 0.29999999999999999 & nominal >= 2" in (export.model)
    assert "(nominal'=nominal-2)" in export.model
    assert "(cool_down'=cool_down+1)" in export.model
    assert 'label "active_done" = done > 0;' in export.model


def test_formal_verification_package_records_hashes_and_checker_commands() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    properties = (
        FormalSafetyProperty(
            name="petri_tokens_bounded",
            artifact_name="petri_supervisor",
            checker="prism",
            expression='P>=1 [ G !"active_done" | F "active_done" ]',
            description="Petri supervisor reaches a reviewed terminal place.",
        ),
        FormalSafetyProperty(
            name="tla_type_ok",
            artifact_name="petri_supervisor_tla",
            checker="tlc",
            expression="Safety",
            description="TLA state variables remain inside exported bounds.",
        ),
    )

    package = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        properties,
        package_name="spo-formal-review",
    )
    repeated = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        properties,
        package_name="spo-formal-review",
    )
    record = package.to_audit_record()

    assert isinstance(package, FormalVerificationPackage)
    assert package.package_hash == repeated.package_hash
    assert record["artifact_types"] == {
        "petri_supervisor": "prism",
        "petri_supervisor_tla": "tla",
    }
    assert sorted(record["artifact_hashes"]) == [
        "petri_supervisor",
        "petri_supervisor_tla",
    ]
    assert all(len(item) == 64 for item in record["artifact_hashes"].values())
    assert record["properties"] == [
        {
            "name": "petri_tokens_bounded",
            "artifact_name": "petri_supervisor",
            "checker": "prism",
            "expression": 'P>=1 [ G !"active_done" | F "active_done" ]',
            "description": "Petri supervisor reaches a reviewed terminal place.",
            "required": True,
        },
        {
            "name": "tla_type_ok",
            "artifact_name": "petri_supervisor_tla",
            "checker": "tlc",
            "expression": "Safety",
            "description": "TLA state variables remain inside exported bounds.",
            "required": True,
        },
    ]
    assert record["checker_commands"] == [
        {
            "property_name": "petri_tokens_bounded",
            "checker": "prism",
            "artifact_name": "petri_supervisor",
            "command": [
                "prism",
                "petri_supervisor.prism",
                "-pf",
                'P>=1 [ G !"active_done" | F "active_done" ]',
            ],
            "execution_permitted": False,
        },
        {
            "property_name": "tla_type_ok",
            "checker": "tlc",
            "artifact_name": "petri_supervisor_tla",
            "command": [
                "tlc2.TLC",
                "petri_supervisor_tla.tla",
                "-config",
                "petri_supervisor_tla.cfg",
            ],
            "execution_permitted": False,
        },
    ]


def test_formal_checker_availability_audit_is_non_executing_and_deterministic() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    package = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        (
            FormalSafetyProperty(
                name="petri_tokens_bounded",
                artifact_name="petri_supervisor",
                checker="prism",
                expression='P>=1 [ G !"active_done" | F "active_done" ]',
            ),
            FormalSafetyProperty(
                name="tla_type_ok",
                artifact_name="petri_supervisor_tla",
                checker="tlc",
                expression="Safety",
            ),
        ),
    )

    audit = audit_formal_checker_availability(
        package,
        executable_paths={
            "prism": "/opt/prism/bin/prism",
            "tlc2.TLC": None,
        },
    )
    repeated = audit_formal_checker_availability(
        package,
        executable_paths={
            "prism": "/opt/prism/bin/prism",
            "tlc2.TLC": None,
        },
    )

    assert audit == repeated
    assert all(isinstance(item, FormalCheckerAvailability) for item in audit)
    assert [item.to_audit_record() for item in audit] == [
        {
            "property_name": "petri_tokens_bounded",
            "checker": "prism",
            "artifact_name": "petri_supervisor",
            "executable": "prism",
            "command": [
                "prism",
                "petri_supervisor.prism",
                "-pf",
                'P>=1 [ G !"active_done" | F "active_done" ]',
            ],
            "available": True,
            "resolved_path": "/opt/prism/bin/prism",
            "status": "ready_not_executed",
            "execution_permitted": False,
        },
        {
            "property_name": "tla_type_ok",
            "checker": "tlc",
            "artifact_name": "petri_supervisor_tla",
            "executable": "tlc2.TLC",
            "command": [
                "tlc2.TLC",
                "petri_supervisor_tla.tla",
                "-config",
                "petri_supervisor_tla.cfg",
            ],
            "available": False,
            "resolved_path": None,
            "status": "missing_executable",
            "execution_permitted": False,
        },
    ]


def test_formal_package_supports_spin_and_smt_workflow_manifests() -> None:
    promela = FormalTextArtifact(
        artifact_type="promela",
        text="never { /* terminal reachability monitor */ accept_init: skip }\n",
    )
    smt = FormalTextArtifact(
        artifact_type="smt2",
        text="(set-logic QF_LRA)\n(assert true)\n(check-sat)\n",
    )

    package = build_formal_verification_package(
        {
            "protocol_spin": promela,
            "barrier_smt": smt,
        },
        (
            FormalSafetyProperty(
                name="spin_no_bad_cycle",
                artifact_name="protocol_spin",
                checker="spin",
                expression="never_claim_accepts_no_bad_cycle",
                description="Promela monitor has no accepted bad cycle.",
            ),
            FormalSafetyProperty(
                name="smt_barrier_feasible",
                artifact_name="barrier_smt",
                checker="smt",
                expression="check-sat",
                description="SMT barrier constraints are satisfiable.",
            ),
        ),
        package_name="spo-frontier-formal-review",
    )
    readiness = audit_formal_checker_availability(
        package,
        executable_paths={"spin": "/ci/bin/spin", "z3": None},
    )

    record = package.to_audit_record()
    assert record["artifact_types"] == {
        "barrier_smt": "smt2",
        "protocol_spin": "promela",
    }
    assert record["checker_commands"] == [
        {
            "property_name": "spin_no_bad_cycle",
            "checker": "spin",
            "artifact_name": "protocol_spin",
            "command": ["spin", "-run", "protocol_spin.pml"],
            "execution_permitted": False,
        },
        {
            "property_name": "smt_barrier_feasible",
            "checker": "smt",
            "artifact_name": "barrier_smt",
            "command": ["z3", "barrier_smt.smt2"],
            "execution_permitted": False,
        },
    ]
    assert [item.to_audit_record() for item in readiness] == [
        {
            "property_name": "spin_no_bad_cycle",
            "checker": "spin",
            "artifact_name": "protocol_spin",
            "executable": "spin",
            "command": ["spin", "-run", "protocol_spin.pml"],
            "available": True,
            "resolved_path": "/ci/bin/spin",
            "status": "ready_not_executed",
            "execution_permitted": False,
        },
        {
            "property_name": "smt_barrier_feasible",
            "checker": "smt",
            "artifact_name": "barrier_smt",
            "executable": "z3",
            "command": ["z3", "barrier_smt.smt2"],
            "available": False,
            "resolved_path": None,
            "status": "missing_executable",
            "execution_permitted": False,
        },
    ]


def test_runtime_control_certificate_binds_passed_formal_evidence() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    package = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        (
            FormalSafetyProperty(
                name="petri_tokens_bounded",
                artifact_name="petri_supervisor",
                checker="prism",
                expression='P>=1 [ G !"active_done" | F "active_done" ]',
            ),
            FormalSafetyProperty(
                name="tla_type_ok",
                artifact_name="petri_supervisor_tla",
                checker="tlc",
                expression="Safety",
            ),
        ),
        package_name="spo-formal-review",
    )
    availability = audit_formal_checker_availability(
        package,
        executable_paths={
            "prism": "/opt/prism/bin/prism",
            "tlc2.TLC": "/opt/tlc/tlc2.TLC",
        },
    )
    results = (
        FormalCheckerResult(
            property_name="tla_type_ok",
            checker="tlc",
            artifact_name="petri_supervisor_tla",
            package_hash=package.package_hash,
            result_hash="b" * 64,
            status="passed",
            passed=True,
            detail="TLC Safety accepted in reviewed CI run.",
        ),
        FormalCheckerResult(
            property_name="petri_tokens_bounded",
            checker="prism",
            artifact_name="petri_supervisor",
            package_hash=package.package_hash,
            result_hash="a" * 64,
            status="passed",
            passed=True,
            detail="PRISM property accepted in reviewed CI run.",
        ),
    )

    certificate = build_runtime_control_certificate(
        package,
        tuple(reversed(availability)),
        results,
        {"R_min": 0.7, "max_latency_s": 0.05},
    )
    repeated = build_runtime_control_certificate(
        package,
        availability,
        tuple(reversed(results)),
        {"max_latency_s": 0.05, "R_min": 0.7},
    )
    record = certificate.to_audit_record()

    assert isinstance(certificate, FormalRuntimeCertificate)
    assert certificate.certificate_hash == repeated.certificate_hash
    assert certificate.status == "verified_non_actuating"
    assert certificate.required_property_count == 2
    assert certificate.passed_required_count == 2
    assert certificate.actuation_permitted is False
    assert record["runtime_bounds"] == {"R_min": 0.7, "max_latency_s": 0.05}
    assert record["missing_required_properties"] == []
    assert record["failed_required_properties"] == []
    assert record["unavailable_checker_properties"] == []
    assert len(record["certificate_hash"]) == 64
    assert all(
        item["execution_permitted"] is False for item in record["checker_results"]
    )


def test_runtime_control_certificate_blocks_incomplete_or_stale_evidence() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )
    package = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        (
            FormalSafetyProperty(
                name="petri_tokens_bounded",
                artifact_name="petri_supervisor",
                checker="prism",
                expression='P>=1 [ G !"active_done" | F "active_done" ]',
            ),
            FormalSafetyProperty(
                name="tla_type_ok",
                artifact_name="petri_supervisor_tla",
                checker="tlc",
                expression="Safety",
            ),
        ),
        package_name="spo-formal-review",
    )
    blocked = build_runtime_control_certificate(
        package,
        audit_formal_checker_availability(
            package,
            executable_paths={"prism": "/opt/prism/bin/prism", "tlc2.TLC": None},
        ),
        (
            FormalCheckerResult(
                property_name="petri_tokens_bounded",
                checker="prism",
                artifact_name="petri_supervisor",
                package_hash=package.package_hash,
                result_hash="a" * 64,
                status="passed",
                passed=True,
            ),
        ),
        {"R_min": 0.7},
    )

    assert blocked.status == "blocked"
    assert blocked.passed_required_count == 1
    assert blocked.missing_required_properties == ("tla_type_ok",)
    assert blocked.unavailable_checker_properties == ("tla_type_ok",)
    assert blocked.actuation_permitted is False

    with pytest.raises(PolicyError, match="package_hash does not match"):
        build_runtime_control_certificate(
            package,
            (),
            (
                FormalCheckerResult(
                    property_name="petri_tokens_bounded",
                    checker="prism",
                    artifact_name="petri_supervisor",
                    package_hash="c" * 64,
                    result_hash="a" * 64,
                    status="passed",
                    passed=True,
                ),
            ),
            {"R_min": 0.7},
        )
    with pytest.raises(PolicyError, match="runtime bounds"):
        build_runtime_control_certificate(package, (), (), {"R_min": float("nan")})


def test_formal_checker_result_rejects_non_auditable_results() -> None:
    with pytest.raises(PolicyError, match="status must match"):
        FormalCheckerResult(
            property_name="petri_tokens_bounded",
            checker="prism",
            artifact_name="petri_supervisor",
            package_hash="a" * 64,
            result_hash="b" * 64,
            status="failed",
            passed=True,
        )
    with pytest.raises(PolicyError, match="execution must stay disabled"):
        FormalCheckerResult(
            property_name="petri_tokens_bounded",
            checker="prism",
            artifact_name="petri_supervisor",
            package_hash="a" * 64,
            result_hash="b" * 64,
            status="passed",
            passed=True,
            execution_permitted=True,
        )
    with pytest.raises(PolicyError, match="SHA-256"):
        FormalCheckerResult(
            property_name="petri_tokens_bounded",
            checker="prism",
            artifact_name="petri_supervisor",
            package_hash="A" * 64,
            result_hash="b" * 64,
            status="passed",
            passed=True,
        )


def test_formal_checker_availability_and_commands_fail_closed() -> None:
    with pytest.raises(PolicyError, match="execution must stay disabled"):
        FormalCheckerCommand(
            property_name="unsafe_exec",
            checker="prism",
            artifact_name="petri_supervisor",
            command=("prism", "petri_supervisor.prism"),
            execution_permitted=True,
        )
    with pytest.raises(PolicyError, match="must not permit execution"):
        FormalCheckerAvailability(
            property_name="unsafe_availability",
            checker="prism",
            artifact_name="petri_supervisor",
            executable="prism",
            command=("prism", "petri_supervisor.prism"),
            available=True,
            resolved_path="/opt/prism/bin/prism",
            status="ready_not_executed",
            execution_permitted=True,
        )
        with pytest.raises(PolicyError, match="status must match"):
            FormalCheckerAvailability(
                property_name="mismatch",
                checker="tlc",
                artifact_name="petri_supervisor_tla",
                executable="tlc2.TLC",
                command=("tlc2.TLC", "petri_supervisor_tla.tla"),
                available=True,
                status="missing_executable",
            )


def test_formal_checker_readiness_respects_explicit_executable_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    package = build_formal_verification_package(
        {
            "petri_supervisor": prism_export,
            "petri_supervisor_tla": tla_export,
        },
        (
            FormalSafetyProperty(
                name="petri_tokens_bounded",
                artifact_name="petri_supervisor",
                checker="prism",
                expression='P>=1 [ G !"active_done" | F "active_done" ]',
            ),
            FormalSafetyProperty(
                name="tla_type_ok",
                artifact_name="petri_supervisor_tla",
                checker="tlc",
                expression="Safety",
            ),
        ),
        package_name="spo-formal-readiness",
    )

    monkeypatch.setattr(
        "scpn_phase_orchestrator.supervisor.formal_export.shutil.which",
        lambda command: f"/host/bin/{command}",
    )

    explicit_only = audit_formal_checker_availability(
        package,
        executable_paths={"prism": "/ci/bin/prism"},
    )
    assert [record.to_audit_record() for record in explicit_only] == [
        {
            "property_name": "petri_tokens_bounded",
            "checker": "prism",
            "artifact_name": "petri_supervisor",
            "executable": "prism",
            "command": [
                "prism",
                "petri_supervisor.prism",
                "-pf",
                'P>=1 [ G !"active_done" | F "active_done" ]',
            ],
            "available": True,
            "resolved_path": "/ci/bin/prism",
            "status": "ready_not_executed",
            "execution_permitted": False,
        },
        {
            "property_name": "tla_type_ok",
            "checker": "tlc",
            "artifact_name": "petri_supervisor_tla",
            "executable": "tlc2.TLC",
            "command": [
                "tlc2.TLC",
                "petri_supervisor_tla.tla",
                "-config",
                "petri_supervisor_tla.cfg",
            ],
            "available": False,
            "resolved_path": None,
            "status": "missing_executable",
            "execution_permitted": False,
        },
    ]

    wildcard = audit_formal_checker_availability(
        package,
        executable_paths={"prism": "/ci/bin/prism", "tlc2.TLC": "/ci/bin/tlc"},
    )
    assert wildcard[1].resolved_path == "/ci/bin/tlc"
    assert all(item.execution_permitted is False for item in wildcard)


def test_formal_verification_package_validates_manifest_contracts() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    properties = (
        FormalSafetyProperty(
            name="petri_tokens_bounded",
            artifact_name="petri_supervisor",
            checker="prism",
            expression='P>=1 [ G !"active_done" | F "active_done" ]',
        ),
        FormalSafetyProperty(
            name="tla_type_ok",
            artifact_name="petri_supervisor_tla",
            checker="tlc",
            expression="Safety",
        ),
    )

    with pytest.raises(PolicyError, match="package_name"):
        build_formal_verification_package(
            {"petri_supervisor": prism_export, "petri_supervisor_tla": tla_export},
            properties,
            package_name="1-invalid",
        )

    with pytest.raises(PolicyError, match="duplicate formal property"):
        build_formal_verification_package(
            {"petri_supervisor": prism_export, "petri_supervisor_tla": tla_export},
            (properties[0], properties[0]),
        )


def test_formal_verification_package_hash_is_stable_across_artifact_ordering() -> None:
    prism_export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    tla_export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1}),
        module_name="supervisor net",
    )
    properties = (
        FormalSafetyProperty(
            name="petri_tokens_bounded",
            artifact_name="petri_supervisor",
            checker="prism",
            expression='P>=1 [ G !"active_done" | F "active_done" ]',
        ),
        FormalSafetyProperty(
            name="tla_type_ok",
            artifact_name="petri_supervisor_tla",
            checker="tlc",
            expression="Safety",
        ),
    )

    package_a = build_formal_verification_package(
        {"petri_supervisor": prism_export, "petri_supervisor_tla": tla_export},
        properties,
        package_name="spo-formal-review",
    )
    package_b = build_formal_verification_package(
        {"petri_supervisor_tla": tla_export, "petri_supervisor": prism_export},
        properties,
        package_name="spo-formal-review",
    )

    assert package_a.package_hash == package_b.package_hash
    assert package_a.to_audit_record() == package_b.to_audit_record()


@pytest.mark.parametrize(
    ("artifacts", "properties", "message"),
    [
        ({}, [FormalSafetyProperty("p", "a", "prism", "P>=1 [ F true ]")], "artifacts"),
        (
            {"a": export_petri_net_prism(_net(), Marking(tokens={"warmup": 1}))},
            [],
            "properties",
        ),
        (
            {"a": export_petri_net_prism(_net(), Marking(tokens={"warmup": 1}))},
            [FormalSafetyProperty("p", "missing", "prism", "P>=1 [ F true ]")],
            "unknown artifact",
        ),
        (
            {"a": export_petri_net_prism(_net(), Marking(tokens={"warmup": 1}))},
            [FormalSafetyProperty("p", "a", "tlc", "Safety")],
            "checker does not match",
        ),
        (
            {"a": FormalTextArtifact("promela", "never { accept_init: skip }\n")},
            [FormalSafetyProperty("p", "a", "smt", "check-sat")],
            "checker does not match",
        ),
    ],
)
def test_formal_verification_package_rejects_invalid_manifests(
    artifacts,
    properties,
    message: str,
) -> None:
    with pytest.raises(PolicyError, match=message):
        build_formal_verification_package(artifacts, properties)


def test_petri_net_prism_export_is_deterministic() -> None:
    first = export_petri_net_prism(_net(), Marking(tokens={"warmup": 1})).model
    second = export_petri_net_prism(_net(), Marking(tokens={"warmup": 1})).model

    assert first == second


def test_petri_net_prism_export_rejects_bad_token_bound() -> None:
    with pytest.raises(PolicyError, match="max_tokens"):
        export_petri_net_prism(_net(), Marking(tokens={"warmup": 1}), max_tokens=0)


def test_petri_net_prism_export_rejects_initial_tokens_above_bound() -> None:
    with pytest.raises(PolicyError, match="exceeds max_tokens"):
        export_petri_net_prism(_net(), Marking(tokens={"warmup": 3}), max_tokens=2)


def test_petri_net_prism_export_normalises_collisions_and_source_arcs() -> None:
    net = PetriNet(
        [
            Place("!!!"),
            Place("123-start"),
            Place("phase-1"),
            Place("phase 1"),
        ],
        [
            Transition(
                name="emit source",
                inputs=[],
                outputs=[Arc("123-start")],
            ),
            Transition(
                name="advance",
                inputs=[Arc("123-start")],
                outputs=[Arc("phase-1"), Arc("phase 1")],
            ),
        ],
    )

    export = export_petri_net_prism(
        net,
        Marking(tokens={"!!!": 1}),
        module_name="123 module",
    )

    assert export.place_names == {
        "!!!": "p",
        "123-start": "p_123_start",
        "phase 1": "phase_1",
        "phase-1": "phase_1_2",
    }
    assert "module module_123_module" in export.model
    assert "[emit_source] true & true ->" in export.model
    assert "(p_123_start'=p_123_start+1)" in export.model
    assert "//   phase 1 -> phase_1" in export.model
    assert "//   phase-1 -> phase_1_2" in export.model


def test_petri_net_prism_export_rejects_empty_net() -> None:
    with pytest.raises(PolicyError, match="without places"):
        export_petri_net_prism(PetriNet([], []), Marking())


def test_petri_net_prism_export_preserves_collision_invariants() -> None:
    net = PetriNet(
        [Place("in"), Place("out-put"), Place("out put")],
        [
            Transition(
                name="emit phase",
                inputs=[],
                outputs=[Arc("out-put")],
            ),
            Transition(
                name="emit-phase",
                inputs=[Arc("in")],
                outputs=[Arc("out put")],
            ),
            Transition(
                name="emit phase_2",
                inputs=[Arc("out-put")],
                outputs=[Arc("out put")],
            ),
        ],
    )

    export = export_petri_net_prism(
        net,
        Marking(tokens={"in": 3, "out-put": 0, "out put": 0}),
        module_name="9",
    )

    identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    assert export.place_names == {
        "in": "in",
        "out put": "out_put",
        "out-put": "out_put_2",
    }
    assert export.transition_names == {
        "emit phase": "emit_phase",
        "emit-phase": "emit_phase_2",
        "emit phase_2": "emit_phase_2_2",
    }
    assert all(
        identifier_re.fullmatch(name) is not None
        for name in (
            *export.place_names.values(),
            *export.transition_names.values(),
            *export.metric_names.values(),
        )
    )
    assert len(set(export.transition_names.values())) == len(export.transition_names)


def test_petri_net_prism_export_source_transition_no_inputs_guard() -> None:
    source_net = PetriNet(
        [Place("seed"), Place("active")],
        [Transition(name="seed burst", inputs=[], outputs=[Arc("active")])],
    )

    export = export_petri_net_prism(
        source_net,
        Marking(tokens={"seed": 1, "active": 0}),
    )

    assert "[seed_burst] true & true -> (active'=active+1)" in export.model


def test_petri_net_tla_export_serialises_actions_and_invariants() -> None:
    export = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )

    assert isinstance(export, TLAExport)
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "stability.proxy": "stability_proxy",
    }
    assert "---- MODULE supervisor_net ----" in export.module
    assert "EXTENDS Naturals, TLC" in export.module
    assert "CONSTANTS R_bad_0, stability_proxy" in export.module
    assert "VARIABLES cool_down, done, nominal, warmup" in export.module
    assert "Init ==" in export.module
    assert "  /\\ warmup = 1" in export.module
    assert "TypeOK ==" in export.module
    assert "  /\\ nominal \\in 0..2" in export.module
    assert "start ==" in export.module
    assert "  /\\ stability_proxy > 0.59999999999999998" in export.module
    assert "  /\\ warmup >= 1" in export.module
    assert "  /\\ warmup' = warmup - 1" in export.module
    assert "  /\\ nominal' = nominal + 1" in export.module
    assert "wind_down ==" in export.module
    assert "  /\\ R_bad_0 <= 0.29999999999999999" in export.module
    assert "Spec == Init /\\ [][Next]_<<cool_down, done, nominal, warmup>>" in (
        export.module
    )
    assert "Safety == TypeOK" in export.module
    assert "Active_done == done > 0" in export.module


def test_petri_net_tla_export_serialises_source_and_stuttering_nets() -> None:
    source_net = PetriNet(
        [Place("123-start"), Place("phase-1")],
        [
            Transition(
                name="emit source",
                inputs=[],
                outputs=[Arc("phase-1")],
            )
        ],
    )

    source_export = export_petri_net_tla(
        source_net,
        Marking(tokens={"123-start": 1}),
        module_name="123 module",
    )
    assert "---- MODULE SpoModule_123_module ----" in source_export.module
    assert "emit_source ==" in source_export.module
    assert "  /\\ TRUE" not in source_export.module
    assert "  /\\ phase_1' = phase_1 + 1" in source_export.module

    stuttering_export = export_petri_net_tla(
        PetriNet([Place("idle")], []),
        Marking(tokens={"idle": 1}),
    )
    assert "Next ==\n  /\\ UNCHANGED <<idle>>" in stuttering_export.module


def test_petri_net_tla_export_rejects_empty_net_and_bad_bounds() -> None:
    with pytest.raises(PolicyError, match="without places"):
        export_petri_net_tla(PetriNet([], []), Marking())

    with pytest.raises(PolicyError, match="max_tokens"):
        export_petri_net_tla(_net(), Marking(tokens={"warmup": 1}), max_tokens=0)

    with pytest.raises(PolicyError, match="exceeds max_tokens"):
        export_petri_net_tla(_net(), Marking(tokens={"warmup": 3}), max_tokens=2)


def test_petri_net_tla_export_is_deterministic_and_normalised() -> None:
    first = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
    ).module
    second = export_petri_net_tla(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
    ).module
    assert first == second

    assert (
        "---- MODULE SpoModule_123_module ----"
        in export_petri_net_tla(
            _net(),
            Marking(tokens={"warmup": 1, "nominal": 2}),
            module_name="123 module",
        ).module
    )
    assert (
        "---- MODULE start ----"
        in export_petri_net_tla(
            _net(),
            Marking(tokens={"warmup": 1, "nominal": 2}),
            module_name=".start",
        ).module
    )


def _rules() -> list[PolicyRule]:
    return [
        PolicyRule(
            name="boost K",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition(
                metric="R_good",
                layer=0,
                op="<",
                threshold=0.6,
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
            max_fires=2,
        ),
        PolicyRule(
            name="damp_bad",
            regimes=["CRITICAL"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition(
                        metric="R_bad",
                        layer=0,
                        op=">",
                        threshold=0.4,
                    ),
                    PolicyCondition(
                        metric="stability_proxy",
                        layer=None,
                        op="<=",
                        threshold=0.5,
                    ),
                ],
                logic="AND",
            ),
            actions=[
                PolicyAction(knob="alpha", scope="layer_0", value=-0.05, ttl_s=3.0)
            ],
        ),
    ]


def test_policy_rules_prism_export_serialises_rules_and_actions() -> None:
    export = export_policy_rules_prism(_rules(), module_name="policy model")

    assert export.rule_names == {"boost K": "boost_K", "damp_bad": "damp_bad"}
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "R_good.0": "R_good_0",
        "stability_proxy": "stability_proxy",
    }
    assert "module policy_model" in export.model
    assert "//   CRITICAL -> 0" in export.model
    assert "//   DEGRADED -> 1" in export.model
    assert "boost_K_fires : [0..2] init 0;" in export.model
    assert "damp_bad_fires : [0..1] init 0;" in export.model
    assert "[boost_K] (regime = 1 | regime = 0) & R_good_0 < 0.59999999999999998" in (
        export.model
    )
    assert (
        "[damp_bad] (regime = 0) & "
        "(R_bad_0 > 0.40000000000000002 & stability_proxy <= 0.5)"
    ) in export.model
    assert 'label "fires_boost_K" = boost_K_fires > 0;' in export.model
    assert 'label "emits_boost_K_K_global_0" = boost_K_fires > 0;' in export.model


def test_policy_rules_prism_export_preserves_or_guard_and_unique_actions() -> None:
    rules = [
        PolicyRule(
            name="shed load",
            regimes=["DEGRADED"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition("boundary_violation_count", None, ">=", 1.0),
                    PolicyCondition("imprint_mean", None, "<", 0.2),
                ],
                logic="OR",
            ),
            actions=[
                PolicyAction("K", "global", -0.1, 2.0),
                PolicyAction("K", "global", -0.2, 4.0),
            ],
        )
    ]

    export = export_policy_rules_prism(rules, module_name="safety policy")

    assert export.action_names == {
        "shed load.K.global.0": "shed_load_K_global_0",
        "shed load.K.global.1": "shed_load_K_global_1",
    }
    assert (
        "[shed_load] (regime = 0) & "
        "(boundary_violation_count >= 1 | imprint_mean < 0.20000000000000001)"
    ) in export.model
    assert 'label "emits_shed_load_K_global_0" = shed_load_fires > 0;' in (export.model)
    assert 'label "emits_shed_load_K_global_1" = shed_load_fires > 0;' in (export.model)
    assert "//   shed_load_K_global_1: knob='K', scope='global', value=-0.2" in (
        export.model
    )


def test_policy_rules_prism_export_deduplicates_rules_and_actions() -> None:
    rules = [
        PolicyRule(
            name="rule 1",
            regimes=["DEGRADED"],
            condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.1),
            actions=[PolicyAction(knob="gain", scope="global", value=0.2, ttl_s=1.0)],
        ),
        PolicyRule(
            name="rule-1",
            regimes=["DEGRADED"],
            condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.2),
            actions=[PolicyAction(knob="gain", scope="global", value=0.1, ttl_s=1.0)],
        ),
    ]

    export = export_policy_rules_prism(rules, module_name="policy-1")
    assert export.rule_names == {"rule 1": "rule_1", "rule-1": "rule_1_2"}
    assert export.action_names == {
        "rule 1.gain.global.0": "rule_1_gain_global_0",
        "rule-1.gain.global.0": "rule_1_gain_global_0_2",
    }


def test_policy_rules_prism_export_rejects_bad_rules() -> None:
    with pytest.raises(PolicyError, match="without rules"):
        export_policy_rules_prism([])

    bad = PolicyRule(
        name="bad",
        regimes=["DEGRADED"],
        condition=PolicyCondition(metric="R", layer=0, op="!=", threshold=0.1),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    with pytest.raises(PolicyError, match="unsupported operator"):
        export_policy_rules_prism([bad])


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (
            PolicyRule(
                name="",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "names must not be empty",
        ),
        (
            PolicyRule(
                name="no regimes",
                regimes=[],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "has no regimes",
        ),
        (
            PolicyRule(
                name="no actions",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[],
            ),
            "has no actions",
        ),
        (
            PolicyRule(
                name="empty metric",
                regimes=["DEGRADED"],
                condition=PolicyCondition("", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "empty metric",
        ),
        (
            PolicyRule(
                name="non finite",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", float("inf")),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "non-finite threshold",
        ),
    ],
)
def test_policy_rules_prism_export_rejects_invalid_rule_shapes(
    rule: PolicyRule,
    message: str,
) -> None:
    with pytest.raises(PolicyError, match=message):
        export_policy_rules_prism([rule])


def test_policy_rules_prism_export_rejects_empty_compound_conditions() -> None:
    rule = PolicyRule(
        name="empty compound",
        regimes=["DEGRADED"],
        condition=CompoundCondition(conditions=[]),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )

    with pytest.raises(PolicyError, match="compound policy condition"):
        export_policy_rules_prism([rule])


def test_policy_rules_tla_export_serialises_guards_and_emission_predicates() -> None:
    export = export_policy_rules_tla(_rules(), module_name="policy model")

    assert isinstance(export, TLAExport)
    assert export.rule_names == {"boost K": "boost_K", "damp_bad": "damp_bad"}
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "R_good.0": "R_good_0",
        "stability_proxy": "stability_proxy",
    }
    assert "---- MODULE policy_model ----" in export.module
    assert "CONSTANTS regime, R_bad_0, R_good_0, stability_proxy" in export.module
    assert "VARIABLES boost_K_fires, damp_bad_fires" in export.module
    assert "  /\\ boost_K_fires \\in 0..2" in export.module
    assert "boost_K ==" in export.module
    assert "  /\\ (regime = 1 \\/ regime = 0)" in export.module
    assert "  /\\ R_good_0 < 0.59999999999999998" in export.module
    assert "  /\\ boost_K_fires' = boost_K_fires + 1" in export.module
    assert "  /\\ damp_bad_fires' = damp_bad_fires" in export.module
    assert (
        "  /\\ (R_bad_0 > 0.40000000000000002 /\\ stability_proxy <= 0.5)"
    ) in export.module
    assert (
        "Spec == Init /\\ [][Next]_<<boost_K_fires, damp_bad_fires>>"
    ) in export.module
    assert "Fires_boost_K == boost_K_fires > 0" in export.module
    assert "Emits_boost_K_K_global_0 == boost_K_fires > 0" in export.module


def test_policy_rules_tla_export_preserves_or_safety_guard_and_unique_actions() -> None:
    rules = [
        PolicyRule(
            name="shed load",
            regimes=["DEGRADED"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition("boundary_violation_count", None, ">=", 1.0),
                    PolicyCondition("imprint_mean", None, "<", 0.2),
                ],
                logic="OR",
            ),
            actions=[
                PolicyAction("K", "global", -0.1, 2.0),
                PolicyAction("K", "global", -0.2, 4.0),
            ],
        )
    ]

    export = export_policy_rules_tla(rules, module_name="safety policy")

    assert export.action_names == {
        "shed load.K.global.0": "shed_load_K_global_0",
        "shed load.K.global.1": "shed_load_K_global_1",
    }
    assert "---- MODULE safety_policy ----" in export.module
    assert (
        "  /\\ (boundary_violation_count >= 1 \\/ imprint_mean < 0.20000000000000001)"
    ) in export.module
    assert "Emits_shed_load_K_global_0 == shed_load_fires > 0" in export.module
    assert "Emits_shed_load_K_global_1 == shed_load_fires > 0" in export.module
    assert "\\*   shed_load_K_global_1: knob='K', scope='global', value=-0.2" in (
        export.module
    )


def test_policy_rules_tla_export_rejects_empty_rules() -> None:
    with pytest.raises(PolicyError, match="without rules"):
        export_policy_rules_tla([])


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (
            PolicyRule(
                name="",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "names must not be empty",
        ),
        (
            PolicyRule(
                name="no regimes",
                regimes=[],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "has no regimes",
        ),
        (
            PolicyRule(
                name="no actions",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", 0.1),
                actions=[],
            ),
            "has no actions",
        ),
        (
            PolicyRule(
                name="empty metric",
                regimes=["DEGRADED"],
                condition=PolicyCondition("", None, ">", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "empty metric",
        ),
        (
            PolicyRule(
                name="non finite",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, ">", float("inf")),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "non-finite threshold",
        ),
        (
            PolicyRule(
                name="bad op",
                regimes=["DEGRADED"],
                condition=PolicyCondition("R", None, "!=", 0.1),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            ),
            "unsupported operator",
        ),
    ],
)
def test_policy_rules_tla_export_rejects_invalid_rule_shapes(
    rule: PolicyRule,
    message: str,
) -> None:
    with pytest.raises(PolicyError, match=message):
        export_policy_rules_tla([rule])


def test_policy_rules_tla_export_rejects_empty_compound_conditions() -> None:
    rule = PolicyRule(
        name="empty compound",
        regimes=["DEGRADED"],
        condition=CompoundCondition(conditions=[]),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )

    with pytest.raises(PolicyError, match="compound policy condition"):
        export_policy_rules_tla([rule])


def test_policy_rules_smt_export_serialises_feasible_rule_guards() -> None:
    export = export_policy_rules_smt(_rules(), module_name="policy model")

    assert supervisor_api.export_policy_rules_smt is export_policy_rules_smt
    assert isinstance(export, FormalTextArtifact)
    assert export.artifact_type == "smt2"
    assert "(set-logic QF_LRA)" in export.text
    assert "; Module: policy_model" in export.text
    assert "(declare-const regime Real)" in export.text
    assert "(declare-const R_bad_0 Real)" in export.text
    assert "(declare-const boost_K_fire_count Real)" in export.text
    assert "(assert (or (= regime 0) (= regime 1)))" in export.text
    assert (
        "(define-fun fires_boost_K () Bool "
        "(and (or (= regime 1) (= regime 0)) "
        "(< R_good_0 0.59999999999999998) (< boost_K_fire_count 2)))"
    ) in export.text
    assert (
        "(define-fun fires_damp_bad () Bool "
        "(and (= regime 0) "
        "(and (> R_bad_0 0.40000000000000002) (<= stability_proxy 0.5)) "
        "(< damp_bad_fire_count 1)))"
    ) in export.text
    assert "(define-fun emits_boost_K_K_global_0 () Bool fires_boost_K)" in export.text
    assert "(assert (or fires_boost_K fires_damp_bad))" in export.text
    assert export.text.endswith("(check-sat)\n")


def test_policy_rules_smt_export_reuses_rule_validation() -> None:
    with pytest.raises(PolicyError, match="without rules"):
        export_policy_rules_smt([])

    bad = PolicyRule(
        name="bad",
        regimes=["DEGRADED"],
        condition=PolicyCondition(metric="R", layer=0, op="!=", threshold=0.1),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    with pytest.raises(PolicyError, match="unsupported operator"):
        export_policy_rules_smt([bad])


def test_policy_rules_smt_export_rejects_empty_compound_conditions() -> None:
    rule = PolicyRule(
        name="empty compound",
        regimes=["DEGRADED"],
        condition=CompoundCondition(conditions=[]),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )

    with pytest.raises(PolicyError, match="compound policy condition"):
        export_policy_rules_smt([rule])


def test_policy_rules_smt_export_formats_negative_exponent_literals() -> None:
    export = export_policy_rules_smt(
        [
            PolicyRule(
                name="tiny boundary",
                regimes=["DEGRADED"],
                condition=PolicyCondition(
                    metric="delta",
                    layer=None,
                    op=">=",
                    threshold=-1e-20,
                ),
                actions=[PolicyAction("K", "global", 0.1, 1.0)],
            )
        ],
        module_name="tiny",
    )

    assert "(>= delta (- 0." in export.text
    assert "1e-20" not in export.text.lower()
    assert "(assert fires_tiny_boundary)" in export.text


def test_stl_specs_prism_export_serialises_satisfaction_labels() -> None:
    export = export_stl_specs_prism(
        [
            PolicySTLSpec(
                name="keep sync",
                spec="always (R >= 0.3 and amplitude_spread < 0.2)",
                severity="hard",
            ),
            PolicySTLSpec(
                name="recover",
                spec="eventually (R >= 0.8)",
            ),
        ],
        module_name="stl model",
    )

    assert export.stl_names == {"keep sync": "keep_sync", "recover": "recover"}
    assert export.metric_names == {
        "R": "R",
        "amplitude_spread": "amplitude_spread",
    }
    assert "module stl_model" in export.model
    assert "const double R;" in export.model
    assert "const double amplitude_spread;" in export.model
    assert (
        'label "stl_keep_sync_satisfied" = '
        "R >= 0.29999999999999999 & amplitude_spread < 0.20000000000000001;"
    ) in export.model
    assert 'label "stl_recover_violated" = !(R >= 0.80000000000000004);' in (
        export.model
    )


def test_stl_specs_prism_export_rejects_unsupported_syntax() -> None:
    with pytest.raises(PolicyError, match="unsupported export syntax"):
        export_stl_specs_prism([PolicySTLSpec("until", "x until y")])


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([], "without specs"),
        ([PolicySTLSpec("", "always (R >= 0.3)")], "names must not be empty"),
        (
            [PolicySTLSpec("bad severity", "always (R >= 0.3)", "critical")],
            "unsupported severity",
        ),
        (
            [PolicySTLSpec("bad predicate", "always (R != 0.3)")],
            "unsupported predicate syntax",
        ),
    ],
)
def test_stl_specs_prism_export_rejects_invalid_monitor_shapes(
    specs: list[PolicySTLSpec],
    message: str,
) -> None:
    with pytest.raises(PolicyError, match=message):
        export_stl_specs_prism(specs)


def test_stl_specs_prism_export_dedups_and_parses_conjunctions() -> None:
    export = export_stl_specs_prism(
        [
            PolicySTLSpec(
                name="keep sync",
                spec="always (R >= 0.3 and amplitude_spread < 0.5)",
                severity="hard",
            ),
            PolicySTLSpec(
                name="keep-sync",
                spec="eventually (R >= 0.2 && amplitude_spread <= 0.4)",
                severity="hard",
            ),
            PolicySTLSpec(
                name="keep&sync",
                spec="always (R == 0.1)",
                severity="soft",
            ),
        ],
        module_name="stl 1",
    )

    assert export.stl_names == {
        "keep sync": "keep_sync",
        "keep-sync": "keep_sync_2",
        "keep&sync": "keep_sync_3",
    }
    assert export.metric_names == {"R": "R", "amplitude_spread": "amplitude_spread"}
    assert (
        'label "stl_keep_sync_satisfied" = '
        "R >= 0.29999999999999999 & amplitude_spread < 0.5;"
    ) in export.model
    assert (
        'label "stl_keep_sync_2_satisfied" = '
        "R >= 0.20000000000000001 & amplitude_spread <= 0.40000000000000002;"
    ) in export.model
    assert (
        'label "stl_keep_sync_3_violated" = !(R == 0.10000000000000001);'
    ) in export.model


def test_formal_export_cli_writes_prism_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
        "protocol_net": {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                },
            ],
        },
    }
    spec_path = tmp_path / "binding_spec.yaml"
    out_path = tmp_path / "protocol.prism"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--output",
            str(out_path),
            "--module-name",
            "formal_test",
        ],
    )

    assert result.exit_code == 0
    assert "PRISM model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "module formal_test" in model
    assert "[start] stability_proxy > 0 & warmup >= 1" in model


def test_formal_export_cli_writes_protocol_tla_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-tla-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
        "protocol_net": {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                },
            ],
        },
    }
    spec_path = tmp_path / "binding_spec.yaml"
    out_path = tmp_path / "protocol.tla"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "protocol-tla",
            "--output",
            str(out_path),
            "--module-name",
            "FormalTLA",
        ],
    )

    assert result.exit_code == 0
    assert "TLA+ model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "---- MODULE FormalTLA ----" in model
    assert "start ==" in model
    assert "  /\\ stability_proxy > 0" in model
    assert "Spec == Init /\\ [][Next]_<<nominal, warmup>>" in model


def test_formal_export_cli_writes_policy_prism_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-policy-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    policy = {
        "rules": [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R_good",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            }
        ]
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "policy.prism"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy",
            "--output",
            str(out_path),
            "--module-name",
            "policy_test",
        ],
    )

    assert result.exit_code == 0
    assert "PRISM model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "module policy_test" in model
    assert "[boost] (regime = 0) & R_good_0 < 0.69999999999999996" in model


def test_formal_export_cli_writes_policy_tla_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-policy-tla-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    policy = {
        "rules": [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R_good",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            }
        ]
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "policy.tla"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy-tla",
            "--output",
            str(out_path),
            "--module-name",
            "PolicyTLA",
        ],
    )

    assert result.exit_code == 0
    assert "TLA+ model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "---- MODULE PolicyTLA ----" in model
    assert "boost ==" in model
    assert "  /\\ R_good_0 < 0.69999999999999996" in model
    assert "Fires_boost == boost_fires > 0" in model


def test_formal_export_cli_writes_policy_smt_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-policy-smt-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    policy = {
        "rules": [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R_good",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            }
        ]
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "policy.smt2"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy-smt",
            "--output",
            str(out_path),
            "--module-name",
            "policy smt",
        ],
    )

    assert result.exit_code == 0
    assert "SMT-LIB model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "; Module: policy_smt" in model
    assert "(declare-const R_good_0 Real)" in model
    assert "(define-fun fires_boost () Bool" in model
    assert "(check-sat)" in model


def test_formal_export_cli_writes_stl_prism_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-stl-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    policy = {
        "rules": [],
        "stl_monitors": [
            {
                "name": "keep_sync",
                "spec": "always (R >= 0.3)",
                "severity": "hard",
            }
        ],
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "stl.prism"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "stl",
            "--output",
            str(out_path),
            "--module-name",
            "stl_test",
        ],
    )

    assert result.exit_code == 0
    assert "PRISM model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "module stl_test" in model
    assert 'label "stl_keep_sync_satisfied" = R >= 0.29999999999999999;' in model


def test_formal_export_cli_package_includes_generated_policy_smt(
    tmp_path: Path,
) -> None:
    spec = {
        "name": "formal-package-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
        "protocol_net": {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                },
            ],
        },
    }
    policy = {
        "rules": [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {"metric": "R_good", "op": "<", "threshold": 0.7},
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            }
        ]
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "package.json"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--include-checker-readiness",
            "--checker-path",
            "z3=/ci/bin/z3",
            "--checker-path",
            "prism=/ci/bin/prism",
            "--checker-path",
            "tlc2.TLC=/ci/bin/tlc2.TLC",
            "--output",
            str(out_path),
            "--module-name",
            "formal_package",
        ],
    )

    assert result.exit_code == 0
    assert "Formal verification package written:" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_types"]["policy_smt"] == "smt2"
    assert payload["artifact_hashes"]["policy_smt"]
    assert {
        "name": "policy_smt_feasible",
        "artifact_name": "policy_smt",
        "checker": "smt",
        "expression": "check-sat",
        "description": "Policy rule activation envelope is SMT-feasible.",
        "required": True,
    } in payload["properties"]
    assert {
        "property_name": "policy_smt_feasible",
        "checker": "smt",
        "artifact_name": "policy_smt",
        "command": ["z3", "policy_smt.smt2"],
        "execution_permitted": False,
    } in payload["checker_commands"]
    readiness = {
        item["property_name"]: item for item in payload["checker_availability"]
    }
    assert readiness["policy_smt_feasible"]["status"] == "ready_not_executed"
    assert readiness["policy_smt_feasible"]["resolved_path"] == "/ci/bin/z3"


def test_formal_export_cli_requires_protocol_net(tmp_path: Path) -> None:
    spec = {
        "name": "formal-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(main, ["formal-export", str(spec_path)])

    assert result.exit_code == 1
    assert "ERROR: binding spec has no protocol_net" in result.output
