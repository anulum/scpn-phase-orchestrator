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

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _publish_workflow() -> dict[str, object]:
    return yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())


def test_linux_maturin_wheels_use_manylinux_python312_interpreter() -> None:
    workflow = _publish_workflow()
    build_wheels = workflow["jobs"]["build-wheels"]  # type: ignore[index]
    matrix = build_wheels["strategy"]["matrix"]["include"]  # type: ignore[index]

    linux_rows = [row for row in matrix if row["os"] == "ubuntu-latest"]
    assert linux_rows
    assert all(
        row["python_interpreter"] == "/opt/python/cp312-cp312/bin/python"
        for row in linux_rows
    )


def test_maturin_step_uses_matrix_interpreter() -> None:
    workflow = _publish_workflow()
    steps = workflow["jobs"]["build-wheels"]["steps"]  # type: ignore[index]
    maturin_step = next(
        step for step in steps if step.get("name") == "Build wheel via maturin"
    )

    args = maturin_step["with"]["args"]
    assert "-i ${{ matrix.python_interpreter }}" in args


def test_dockerfile_base_digests_match_known_resolving_manifests() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (
        "python:3.12-slim@sha256:"
        "46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3"
    ) in dockerfile
    assert dockerfile.count("python:3.12-slim@sha256:") == 3

    assert (
        "sha256:89e45d1b4de96d61e457618ae4da44690eae5578fe2f11d26d1ed02ce5c8e412"
        not in dockerfile
    )
    assert "rust:1.83-slim" not in dockerfile
    assert "rust:1.95-slim" not in dockerfile
    assert (
        "sha256:2be8daddbd3438e0e0c82ddd4a37e0e7ff3c1e0a0e7e0e4ed4e3be0ba26d3e21"
        not in dockerfile
    )
