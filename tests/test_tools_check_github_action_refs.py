# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Tests for check_github_action_refs.py

"""Unit tests for the GitHub Actions reference guard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_github_action_refs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_github_action_refs_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_extract_action_refs_ignores_local_and_docker_refs(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        """
        steps:
          - uses: actions/checkout@0123456789012345678901234567890123456789
          - uses: ./local-action
          - uses: docker://alpine:3.20
        """,
        encoding="utf-8",
    )

    refs = mod.extract_action_refs([workflow])

    assert len(refs) == 1
    assert refs[0].repo == "actions/checkout"
    assert refs[0].ref == "0123456789012345678901234567890123456789"


def test_validate_refs_reports_missing_ref(monkeypatch) -> None:
    action_ref = mod.ActionRef(
        path=Path(".github/workflows/publish.yml"),
        line=10,
        repo="PyO3/maturin-action",
        ref="a" * 40,
    )

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["gh", "api", path], 1, "", "missing")

    monkeypatch.setattr(mod, "_gh_api", fake_gh_api)

    missing, non_sha = mod.validate_refs([action_ref])

    assert missing == [action_ref]
    assert non_sha == []


def test_validate_refs_reports_non_sha_ref(monkeypatch) -> None:
    action_ref = mod.ActionRef(
        path=Path(".github/workflows/publish.yml"),
        line=10,
        repo="actions/checkout",
        ref="v6",
    )

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["gh", "api", path], 0, "{}", "")

    monkeypatch.setattr(mod, "_gh_api", fake_gh_api)

    missing, non_sha = mod.validate_refs([action_ref])

    assert missing == []
    assert non_sha == [action_ref]


def test_main_returns_zero_for_resolving_sha(tmp_path: Path, monkeypatch) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "- uses: actions/checkout@0123456789012345678901234567890123456789\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        mod,
        "_gh_api",
        lambda path: subprocess.CompletedProcess(["gh", "api", path], 0, "{}", ""),
    )

    assert mod.main([str(workflow)]) == 0


def test_main_returns_one_when_gh_missing(tmp_path: Path, monkeypatch) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "- uses: actions/checkout@0123456789012345678901234567890123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)

    assert mod.main([str(workflow)]) == 1
