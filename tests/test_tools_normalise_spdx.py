# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the SPDX header normaliser

"""Unit tests for ``tools/normalise_spdx_headers.py`` (pure-function paths).

The normaliser was applied once to 642 files — any future edit to the
split regex, typo constant, or exclusion set needs to fall into a net
of sanity tests so the next sweep does not silently corrupt files.

This file covers the pattern-matching and discovery surface:

* ``split_prefixed`` — matches ``# ... | Commercial license available``
  and ``// ... | Commercial license available`` openers, preserves the
  exact prefix (inc. trailing space) in both output lines.
* ``split_bare`` — rejects commented lines; accepts bare (YAML/TOML)
  lines and preserves leading whitespace.
* ``_is_excluded`` — ``.venv``-style prefixes and known build dirs are
  excluded.
* ``iter_repo_files`` — filters by ``INCLUDE_SUFFIXES``.
* ``FileResult`` — ``changed`` reflects either split or typo.

The I/O-touching paths (``normalise_file`` apply / dry-run,
``HEADER_SCAN_LIMIT`` boundary, ``verify`` exit codes) live in
``test_tools_normalise_spdx_io.py`` to keep both suites under the
no-monoliths soft limit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
# FileResult
#
# ``normalise_file`` and ``verify`` I/O paths moved to the companion
# file ``test_tools_normalise_spdx_io.py`` to keep each suite under the
# no-monoliths soft limit.
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
