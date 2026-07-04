# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Publish workflow hygiene tests

"""Regression tests for release workflow and container deployment hygiene."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _publish_workflow() -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text()),
    )


def _ci_workflow() -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text()),
    )


def test_linux_maturin_wheels_use_executable_python312_interpreter() -> None:
    workflow = _publish_workflow()
    matrix = workflow["jobs"]["build-wheels"]["strategy"]["matrix"]["include"]

    linux_rows = [row for row in matrix if str(row["os"]).startswith("ubuntu")]
    assert linux_rows
    assert all(
        row["python_interpreter"] == "/opt/python/cp312-cp312/bin/python"
        for row in linux_rows
    )
    assert any(
        row["os"] == "ubuntu-24.04-arm" and row["target"] == "aarch64"
        for row in linux_rows
    )


def test_maturin_step_uses_matrix_interpreter() -> None:
    workflow = _publish_workflow()
    steps = workflow["jobs"]["build-wheels"]["steps"]
    maturin_step = next(
        step for step in steps if step.get("name") == "Build wheel via maturin"
    )

    args = maturin_step["with"]["args"]
    assert "-i ${{ matrix.python_interpreter }}" in args


def test_pre_publish_gate_runs_meta_distribution_evidence_guard() -> None:
    workflow = _publish_workflow()
    steps = workflow["jobs"]["preflight"]["steps"]
    commands = [str(step.get("run", "")) for step in steps]

    assert "python tools/check_meta_distribution.py" in commands


def test_pre_publish_gate_deselects_host_sensitive_tests() -> None:
    workflow = _publish_workflow()
    test_command = next(
        step["run"]
        for step in workflow["jobs"]["preflight"]["steps"]
        if step.get("name") == "Run release preflight tests"
    )

    assert '-m "not slow and not performance"' in test_command
    assert '-k "not performance"' in test_command


def test_dockerfile_base_digests_match_known_resolving_manifests() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (
        "python:3.13-slim@sha256:"
        "e544a7fcbdf8555eceda66bf86cafb006c736339f76141918bcb812f3174c00a"
    ) in dockerfile
    assert dockerfile.count("python:3.13-slim@sha256:") == 3

    assert (
        "sha256:89e45d1b4de96d61e457618ae4da44690eae5578fe2f11d26d1ed02ce5c8e412"
        not in dockerfile
    )
    assert (
        "sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3"
        not in dockerfile
    )
    assert (
        "sha256:58525e1a8dada8e72d6f8a11a0ddff8d981fd888549108db52455d577f927f77"
        not in dockerfile
    )
    assert "rust:1.83-slim" not in dockerfile
    assert "rust:1.95-slim" not in dockerfile
    assert (
        "sha256:2be8daddbd3438e0e0c82ddd4a37e0e7ff3c1e0a0e7e0e4ed4e3be0ba26d3e21"
        not in dockerfile
    )


def test_container_scans_gate_only_fixable_high_findings() -> None:
    workflow = _publish_workflow()
    steps = workflow["jobs"]["build-container"]["steps"]
    trivy_step = next(
        step for step in steps if step.get("name") == "Scan image with Trivy"
    )
    grype_step = next(
        step for step in steps if step.get("name") == "Scan image with Grype"
    )

    assert trivy_step["with"]["severity"] == "CRITICAL,HIGH"
    assert trivy_step["with"]["exit-code"] == 1
    assert trivy_step["with"]["ignore-unfixed"] is True

    grype_command = grype_step["run"]
    assert "--fail-on high" in grype_command
    assert "--only-fixed" in grype_command


def test_ci_slow_tests_run_once_outside_python_matrix() -> None:
    workflow = _ci_workflow()
    jobs = workflow["jobs"]

    test_command = next(
        step["run"]
        for step in jobs["test"]["steps"]
        if step.get("name") == "Run tests (with coverage on 3.12)"
    )
    assert '-m "not slow and not performance"' in test_command
    assert '-k "not performance"' in test_command

    slow_job = jobs["slow-tests"]
    assert slow_job["timeout-minutes"] == 20
    assert slow_job["steps"][1]["with"]["python-version"] == "3.12"
    slow_command = slow_job["steps"][-1]["run"]
    assert "-m slow" in slow_command

    gate_needs = jobs["ci-gate"]["needs"]
    assert "slow-tests" in gate_needs


def test_ci_lint_job_ratchets_source_docstring_coverage() -> None:
    workflow = _ci_workflow()
    lint_steps = workflow["jobs"]["lint"]["steps"]
    commands = [str(step.get("run", "")) for step in lint_steps]
    interrogate_command = next(
        command for command in commands if command.startswith("interrogate ")
    )

    assert "interrogate src/scpn_phase_orchestrator" in interrogate_command
    assert "--fail-under 100" in interrogate_command
    assert "--ignore-init-method" in interrogate_command
    assert "--ignore-magic" in interrogate_command
    assert "--exclude src/scpn_phase_orchestrator/runtime/grpc_gen" in (
        interrogate_command
    )
    assert "--quiet" in interrogate_command


def test_dev_locks_hash_pin_interrogate_docstring_gate() -> None:
    for lockfile in (
        "requirements/dev-lock.txt",
        "requirements/dev-lock-py311.txt",
        "requirements/dev-lock-py313.txt",
    ):
        text = (ROOT / lockfile).read_text(encoding="utf-8")

        assert "interrogate==1.7.0 \\" in text
        assert (
            "--hash=sha256:a320d6ec644dfd887cc58247a345054fc4d9f981100c45184470068f4b3719b0"
            in text
        )
        assert (
            "--hash=sha256:b13ff4dd8403369670e2efe684066de9fcb868ad9d7f2b4095d8112142dc9d12"
            in text
        )
        assert "py==1.11.0 \\" in text
        assert "tabulate==0.10.0 \\" in text


def test_ffi_matrix_excludes_slow_tests() -> None:
    workflow = _ci_workflow()
    ffi_steps = workflow["jobs"]["ffi-test"]["steps"]
    pytest_commands = [
        str(step["run"])
        for step in ffi_steps
        if "pytest tests/" in str(step.get("run"))
    ]

    assert pytest_commands
    assert all(
        '-m "not slow and not performance"' in command for command in pytest_commands
    )
    assert all('-k "not performance"' in command for command in pytest_commands)
