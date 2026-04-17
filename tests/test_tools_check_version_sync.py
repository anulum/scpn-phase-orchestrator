# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for check_version_sync.py

"""Unit tests for ``tools/check_version_sync.py``.

Gate that pyproject.toml, CITATION.cff, and spo-kernel/Cargo.toml
agree on the release version. Covers the three regex extractors, the
missing-file / missing-key paths, and the mismatch vs match exit
codes by running the script as a subprocess against a scratch repo
layout.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_version_sync.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_version_sync_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# ---------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------


class TestExtract:
    def test_pyproject_version_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\nname = "foo"\nversion = "0.6.0"\n', encoding="utf-8")
        assert mod._extract(f, r'^version\s*=\s*"([^"]+)"') == "0.6.0"

    def test_citation_version_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "CITATION.cff"
        f.write_text("cff-version: 1.2.0\nversion: 0.6.0\n", encoding="utf-8")
        assert mod._extract(f, r"^version:\s*(\S+)") == "0.6.0"

    def test_cargo_nested_version_ignored(self, tmp_path: Path) -> None:
        """The regex is anchored at line-start; nested ``version = "x"``
        lines inside ``[dependencies]`` would not match by accident."""
        f = tmp_path / "Cargo.toml"
        f.write_text(
            '[package]\nname = "x"\nversion = "0.6.0"\n'
            '[dependencies]\nserde = { version = "1.0" }\n',
            encoding="utf-8",
        )
        assert mod._extract(f, r'^version\s*=\s*"([^"]+)"') == "0.6.0"

    def test_missing_pattern_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.toml"
        f.write_text("name = 'foo'\n", encoding="utf-8")
        assert mod._extract(f, r'^version\s*=\s*"([^"]+)"') is None


# ---------------------------------------------------------------------
# End-to-end via subprocess
# ---------------------------------------------------------------------


def _make_repo(
    tmp_path: Path, py_ver: str | None, cite_ver: str | None, cargo_ver: str | None
) -> Path:
    """Lay out a scratch repo with the three version-bearing files."""
    if py_ver is not None:
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "foo"\nversion = "{py_ver}"\n', encoding="utf-8"
        )
    if cite_ver is not None:
        (tmp_path / "CITATION.cff").write_text(
            f"cff-version: 1.2.0\nversion: {cite_ver}\n", encoding="utf-8"
        )
    if cargo_ver is not None:
        kernel = tmp_path / "spo-kernel"
        kernel.mkdir()
        (kernel / "Cargo.toml").write_text(
            f'[package]\nname = "spo-kernel"\nversion = "{cargo_ver}"\n',
            encoding="utf-8",
        )
    return tmp_path


def _run_with_fake_root(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run the script with ROOT redirected to ``repo``.

    The script resolves ROOT from its own file location, so we exec a
    small wrapper that monkeypatches ``mod.ROOT`` before calling main.
    """
    wrapper = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}');"
        "m = importlib.util.module_from_spec(spec);"
        "sys.modules['m'] = m;"
        "spec.loader.exec_module(m);"
        f"m.ROOT = __import__('pathlib').Path(r'{repo}');"
        "sys.exit(m.main())"
    )
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_in_sync_returns_zero(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "0.6.0", "0.6.0", "0.6.0")
    proc = _run_with_fake_root(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK: all versions = 0.6.0" in proc.stdout


def test_pyproject_mismatch_returns_one(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "0.6.1", "0.6.0", "0.6.0")
    proc = _run_with_fake_root(repo)
    assert proc.returncode == 1
    assert "version mismatch" in proc.stdout
    assert "pyproject.toml: 0.6.1" in proc.stdout


def test_cargo_mismatch_returns_one(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "0.6.0", "0.6.0", "0.5.9")
    proc = _run_with_fake_root(repo)
    assert proc.returncode == 1
    assert "Cargo.toml: 0.5.9" in proc.stdout


def test_missing_file_returns_one(tmp_path: Path) -> None:
    """Deleting CITATION.cff should surface as a FileNotFoundError via
    the regex reader; script exits non-zero."""
    repo = _make_repo(tmp_path, "0.6.0", None, "0.6.0")
    proc = _run_with_fake_root(repo)
    assert proc.returncode != 0


def test_pyproject_version_missing_returns_one(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "foo"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text("version: 0.6.0\n", encoding="utf-8")
    kernel = tmp_path / "spo-kernel"
    kernel.mkdir()
    (kernel / "Cargo.toml").write_text(
        '[package]\nversion = "0.6.0"\n', encoding="utf-8"
    )
    proc = _run_with_fake_root(tmp_path)
    assert proc.returncode == 1
    assert "could not extract version" in proc.stdout
    assert "pyproject.toml" in proc.stdout
