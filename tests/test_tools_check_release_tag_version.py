# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for check_release_tag_version.py

"""Unit tests for the release tag/package version guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_release_tag_version.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_release_tag_version_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_release_tag_version_accepts_semver_tag() -> None:
    assert mod._release_tag_version("v0.5.5") == "0.5.5"


def test_release_tag_version_accepts_prerelease_tag() -> None:
    assert mod._release_tag_version("v0.6.0rc1") == "0.6.0rc1"


def test_release_tag_version_rejects_branch_name() -> None:
    assert mod._release_tag_version("main") is None


def test_project_version_extracts_pyproject_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "scpn-phase-orchestrator"\nversion = "0.5.5"\n',
        encoding="utf-8",
    )
    assert mod._project_version(pyproject) == "0.5.5"


def test_main_accepts_matching_tag(tmp_path: Path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.5.5"\n', encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.5.5")
    assert mod.main() == 0


def test_main_rejects_mismatched_tag(tmp_path: Path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.5.0"\n', encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.5.4")
    assert mod.main() == 1


def test_main_rejects_missing_ref(tmp_path: Path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.5.5"\n', encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    assert mod.main() == 1
