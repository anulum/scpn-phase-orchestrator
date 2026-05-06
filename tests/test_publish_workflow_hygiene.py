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


def test_dockerfile_base_digests_match_known_resolving_manifests() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (
        "python:3.13-slim@sha256:"
        "a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d"
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
