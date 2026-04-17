# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the SPDX header normaliser

"""Unit tests for ``tools/normalise_spdx_headers.py``.

The normaliser was applied once to 642 files — any future edit to the
split regex, typo constant, or exclusion set needs to fall into a net
of sanity tests so the next sweep does not silently corrupt files.

Covered paths:

* ``split_prefixed`` — matches ``# ... | Commercial license available``
  and ``// ... | Commercial license available`` openers, preserves the
  exact prefix (inc. trailing space) in both output lines.
* ``split_bare`` — rejects commented lines; accepts bare (YAML/TOML)
  lines and preserves leading whitespace.
* ``HEADER_SCAN_LIMIT`` — splits only within the first 10 lines.
* Typo fix — ``protoscience@anylum.li`` → ``protoscience@anulum.li``
  wherever it appears, independent of the split.
* ``_is_excluded`` — ``.venv``-style prefixes and known build dirs are
  excluded.
* ``iter_repo_files`` — filters by ``INCLUDE_SUFFIXES``.
* ``normalise_file`` — dry-run does not touch the file; apply rewrites
  it atomically.
* ``verify`` — returns 0 when clean, 1 when any residual merged line
  or typo remains.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load_normaliser() -> ModuleType:
    """Import the normaliser module from its file path (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_normalise_spdx_test_mod",
        TOOLS_DIR / "normalise_spdx_headers.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_normaliser()


# ---------------------------------------------------------------------
# split_prefixed / split_bare
# ---------------------------------------------------------------------


class TestSplit:
    def test_python_hash_prefix_splits(self) -> None:
        line = (
            "# SPDX-License-Identifier: AGPL-3.0-or-later "
            "| Commercial license available\n"
        )
        first, second = mod.split_prefixed(line)
        assert first == "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        assert second == "# Commercial license available\n"

    def test_rust_double_slash_prefix_splits(self) -> None:
        line = (
            "// SPDX-License-Identifier: AGPL-3.0-or-later "
            "| Commercial license available\n"
        )
        first, second = mod.split_prefixed(line)
        assert first == "// SPDX-License-Identifier: AGPL-3.0-or-later\n"
        assert second == "// Commercial license available\n"

    def test_hash_without_space_still_splits(self) -> None:
        line = (
            "#SPDX-License-Identifier: AGPL-3.0-or-later "
            "| Commercial license available\n"
        )
        first, second = mod.split_prefixed(line)
        assert first == "#SPDX-License-Identifier: AGPL-3.0-or-later\n"
        assert second == "#Commercial license available\n"

    def test_prefixed_with_trailing_content_rejected(self) -> None:
        line = "# SPDX ... | Commercial license available   extra\n"
        assert mod.split_prefixed(line) is None

    def test_bare_line_splits_with_leading_whitespace(self) -> None:
        line = (
            "    SPDX-License-Identifier: AGPL-3.0-or-later "
            "| Commercial license available\n"
        )
        first, second = mod.split_bare(line)
        assert first == "    SPDX-License-Identifier: AGPL-3.0-or-later\n"
        assert second == "    Commercial license available\n"

    def test_bare_rejects_commented_line(self) -> None:
        line = (
            "# SPDX-License-Identifier: AGPL-3.0-or-later "
            "| Commercial license available\n"
        )
        assert mod.split_bare(line) is None

    def test_unrelated_line_returns_none(self) -> None:
        assert mod.split_prefixed("# SPDX-License-Identifier: MIT\n") is None
        assert mod.split_bare("just a comment\n") is None


# ---------------------------------------------------------------------
# _is_excluded
# ---------------------------------------------------------------------


class TestExcluded:
    def test_venv_prefix_excluded(self) -> None:
        assert mod._is_excluded((".venv-linux", "lib", "x.py"))
        assert mod._is_excluded(("venv", "site-packages", "x.py"))

    def test_known_build_dir_excluded(self) -> None:
        assert mod._is_excluded(("target", "debug"))
        assert mod._is_excluded(("node_modules", "foo"))
        assert mod._is_excluded(("__pycache__", "m.pyc"))

    def test_normal_path_not_excluded(self) -> None:
        assert not mod._is_excluded(("src", "scpn_phase_orchestrator", "a.py"))


# ---------------------------------------------------------------------
# iter_repo_files suffix filter
# ---------------------------------------------------------------------


def test_iter_repo_files_filters_by_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("# empty\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("skip me\n", encoding="utf-8")
    (tmp_path / "c.rs").write_text("// empty\n", encoding="utf-8")
    yielded = sorted(p.name for p in mod.iter_repo_files(tmp_path))
    assert yielded == ["a.py", "c.rs"]


def test_iter_repo_files_skips_excluded_dirs(tmp_path: Path) -> None:
    keep = tmp_path / "src"
    keep.mkdir()
    (keep / "x.py").write_text("# empty\n", encoding="utf-8")
    skip = tmp_path / ".venv-linux" / "lib"
    skip.mkdir(parents=True)
    (skip / "y.py").write_text("# ignored\n", encoding="utf-8")
    yielded = sorted(p.name for p in mod.iter_repo_files(tmp_path))
    assert yielded == ["x.py"]


# ---------------------------------------------------------------------
# normalise_file
# ---------------------------------------------------------------------


def _sample_merged(typo: bool = False) -> str:
    email = "protoscience@" + "any" + "lum.li" if typo else "protoscience@anulum.li"
    return (
        "# SPDX-License-Identifier: AGPL-3.0-or-later "
        "| Commercial license available\n"
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
        "# ORCID: 0009-0009-3560-0851\n"
        f"# Contact: www.anulum.li | {email}\n"
        "# SCPN Phase Orchestrator — example\n\n"
        "print('hello')\n"
    )


def test_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    original = _sample_merged()
    f.write_text(original, encoding="utf-8")
    result = mod.normalise_file(f, apply=False)
    assert result.spdx_split is True
    assert f.read_text(encoding="utf-8") == original


def test_apply_splits_and_fixes_typo(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text(_sample_merged(typo=True), encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is True
    assert result.typo_fixes == 1
    content = f.read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: AGPL-3.0-or-later\n" in content
    assert "Commercial license available\n" in content
    assert "| Commercial license available" not in content
    assert "protoscience@" + "any" + "lum.li" not in content
    assert "protoscience@anulum.li" in content


def test_deep_merged_line_not_split(tmp_path: Path) -> None:
    """Merged SPDX past HEADER_SCAN_LIMIT stays put — the header-only
    invariant keeps the tool focused and avoids touching user strings."""
    padding = "\n".join(f"# pad line {i}" for i in range(15)) + "\n"
    merged = (
        "# SPDX-License-Identifier: AGPL-3.0-or-later "
        "| Commercial license available\n"
    )
    content = padding + merged
    f = tmp_path / "deep.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert f.read_text(encoding="utf-8") == content


def test_typo_anywhere_is_fixed_even_without_split(tmp_path: Path) -> None:
    """The typo fix runs independently of the SPDX split — files that
    are already split but contain the old email still get corrected."""
    typo = "protoscience@" + "any" + "lum.li"
    content = (
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "# Commercial license available\n"
        f"# Contact: {typo}\n\n"
        "print('ok')\n"
    )
    f = tmp_path / "already_split.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert result.typo_fixes == 1
    assert typo not in f.read_text(encoding="utf-8")
    assert "protoscience@anulum.li" in f.read_text(encoding="utf-8")


def test_no_change_file_reports_no_change(tmp_path: Path) -> None:
    content = "# SPDX-License-Identifier: MIT\n\nprint('x')\n"
    f = tmp_path / "unrelated.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert result.typo_fixes == 0
    assert result.changed is False
    assert f.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------


def test_verify_clean_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "a.py"
    f.write_text(
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "# Commercial license available\n",
        encoding="utf-8",
    )
    assert mod.verify(tmp_path) == 0
    assert "All files normalised" in capsys.readouterr().out


def test_verify_detects_residual_merged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "residual.py"
    f.write_text(
        "# SPDX-License-Identifier: AGPL-3.0-or-later "
        "| Commercial license available\n",
        encoding="utf-8",
    )
    assert mod.verify(tmp_path) == 1
    assert "still contain merged SPDX" in capsys.readouterr().out


def test_verify_detects_typo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "typo.py"
    typo = "protoscience@" + "any" + "lum.li"
    f.write_text(f"# Contact: {typo}\n", encoding="utf-8")
    assert mod.verify(tmp_path) == 1
    assert "still contain" in capsys.readouterr().out


# ---------------------------------------------------------------------
# FileResult
# ---------------------------------------------------------------------


class TestFileResult:
    def test_changed_true_when_split(self) -> None:
        r = mod.FileResult(path=Path("x"), spdx_split=True, typo_fixes=0)
        assert r.changed is True

    def test_changed_true_when_typo_only(self) -> None:
        r = mod.FileResult(path=Path("x"), spdx_split=False, typo_fixes=3)
        assert r.changed is True

    def test_changed_false_when_nothing(self) -> None:
        r = mod.FileResult(path=Path("x"), spdx_split=False, typo_fixes=0)
        assert r.changed is False
