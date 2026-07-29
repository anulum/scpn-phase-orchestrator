# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — polyglot documentation regressions

"""Keep native Rust, Go, Julia, and Mojo documentation generation wired."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOC_PAGE = ROOT / "docs" / "reference" / "polyglot_api.md"


def _workflow_jobs() -> dict[str, Any]:
    workflow = cast("dict[str, Any]", yaml.safe_load(WORKFLOW.read_text()))
    return cast("dict[str, Any]", workflow["jobs"])


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_ci_generates_and_uploads_each_native_documentation_format() -> None:
    """Every maintained non-Python backend must retain a CI artifact."""
    jobs = _workflow_jobs()
    expected = {
        "rust-check": (
            "Generate Rust API documentation",
            "Upload Rust API documentation",
        ),
        "go-backend": ("Generate Go API documentation", "Upload Go API documentation"),
        "julia-backend": (
            "Generate Julia API documentation",
            "Upload Julia API documentation",
        ),
        "mojo-backend": (
            "Generate Mojo API documentation",
            "Upload Mojo API documentation",
        ),
    }
    for job_name, (generate_name, upload_name) in expected.items():
        _step(jobs[job_name], generate_name)
        upload = _step(jobs[job_name], upload_name)
        assert "documentation" in generate_name.lower()
        assert upload["uses"].startswith("actions/upload-artifact@")
        assert upload["with"]["name"].startswith("polyglot-docs-")

    rust = _step(jobs["rust-check"], "Generate Rust API documentation")
    assert rust["env"]["RUSTDOCFLAGS"] == "-D warnings"
    assert "cargo clean --doc" in rust["run"]
    assert "cargo doc --locked --no-deps --workspace" in rust["run"]


def test_public_docs_explain_all_native_formats_and_local_command() -> None:
    """The artifact contract must be discoverable without reading CI YAML."""
    text = DOC_PAGE.read_text(encoding="utf-8")
    for token in (
        "cargo doc",
        "go doc",
        "Julia",
        "mojo doc",
        "generate_polyglot_docs.py",
    ):
        assert token in text
    assert "polyglot_api.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain is unavailable")
def test_go_generator_preserves_independent_translation_units(tmp_path: Path) -> None:
    """The coordinator must not merge standalone c-shared Go programs."""
    repo = tmp_path / "repo"
    source_root = repo / "go"
    source_root.mkdir(parents=True)
    (source_root / "go.mod").write_text("module example.test/docs\n\ngo 1.23\n")
    for name in ("alpha", "beta"):
        (source_root / f"{name}.go").write_text(
            f"// Package main documents {name}.\npackage main\n\n"
            f"// {name.title()} returns its unit name.\n"
            f'func {name.title()}() string {{ return "{name}" }}\n\nfunc main() {{}}\n',
            encoding="utf-8",
        )

    output = tmp_path / "artifacts"
    subprocess.run(
        (
            sys.executable,
            str(ROOT / "tools" / "generate_polyglot_docs.py"),
            "go",
            "--repo-root",
            str(repo),
            "--output",
            str(output),
        ),
        check=True,
    )

    generated = sorted((output / "go").glob("*.txt"))
    assert [path.name for path in generated] == ["alpha.txt", "beta.txt"]
    assert "func Alpha() string" in generated[0].read_text(encoding="utf-8")
    assert "func Beta() string" in generated[1].read_text(encoding="utf-8")
