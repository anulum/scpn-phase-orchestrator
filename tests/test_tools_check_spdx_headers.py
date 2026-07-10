# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for check_spdx_headers.py

"""Unit tests for ``tools/check_spdx_headers.py``.

Cover header detection (line 1, shebang-then-line 2, missing, undecodable), the
generated-source exemption, the missing-file collector, the git-backed tracked
enumeration and its no-git guard, and the ``main`` entry point's pass/fail
branches — including that the real repository tree is fully SPDX-compliant.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_spdx_headers.py"

SPDX_LINE = "# SPDX-License-Identifier: AGPL-3.0-or-later\n"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_spdx_headers_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_header_on_first_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.py", SPDX_LINE + "import os\n")
    assert mod.has_spdx_header(path) is True


def test_header_after_shebang(tmp_path: Path) -> None:
    path = _write(tmp_path / "b.py", "#!/usr/bin/env python3\n" + SPDX_LINE)
    assert mod.has_spdx_header(path) is True


def test_shebang_without_following_header(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.py", "#!/usr/bin/env python3\nimport os\n")
    assert mod.has_spdx_header(path) is False


def test_no_header_at_all(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.py", "import os\n")
    assert mod.has_spdx_header(path) is False


def test_undecodable_file_is_not_a_header(tmp_path: Path) -> None:
    path = tmp_path / "e.py"
    path.write_bytes(b"\xff\xfe\x00bad bytes")
    assert mod.has_spdx_header(path) is False


def test_is_exempt_matches_generated_grpc_only() -> None:
    assert mod._is_exempt("src/scpn_phase_orchestrator/runtime/grpc_gen/spo_pb2.py")
    assert not mod._is_exempt("src/scpn_phase_orchestrator/studio/panel_data.py")


def test_find_missing_skips_exempt_and_flags_bare(tmp_path: Path) -> None:
    good = _write(tmp_path / "src/pkg/good.py", SPDX_LINE + "x = 1\n")
    bare = _write(tmp_path / "src/pkg/bare.py", "x = 1\n")
    exempt = _write(
        tmp_path / "src/scpn_phase_orchestrator/runtime/grpc_gen/spo_pb2.py",
        "x = 1\n",  # generated, no project header, but exempt
    )
    missing = mod.find_missing([good, bare, exempt], root=tmp_path)
    assert missing == [bare]


def test_tracked_python_files_returns_repo_sources() -> None:
    files = mod.tracked_python_files()
    assert files, "the repository has tracked Python files"
    assert all(p.suffix == ".py" for p in files)
    # Assert on an already-tracked tool (this gate's own file may still be
    # unstaged when the test first runs).
    assert any(p.name == "capability_manifest.py" for p in files)


def test_tracked_python_files_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="git executable is required"):
        mod.tracked_python_files()


def test_main_passes_on_compliant_explicit_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write(tmp_path / "a.py", SPDX_LINE + "x = 1\n")
    b = _write(tmp_path / "b.py", "#!/usr/bin/env python3\n" + SPDX_LINE)
    assert mod.main([str(a), str(b)]) == 0
    assert "present on all 2" in capsys.readouterr().out


def test_main_fails_and_reports_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bare = _write(tmp_path / "bare.py", "x = 1\n")
    assert mod.main([str(bare)]) == 1
    err = capsys.readouterr().err
    assert "missing from 1" in err
    assert "bare.py" in err


def test_display_path_repo_relative_and_absolute(tmp_path: Path) -> None:
    # A path inside the repo renders repo-relative; one outside renders absolute.
    assert mod._display_path(SCRIPT) == "tools/check_spdx_headers.py"
    outside = tmp_path / "x.py"
    assert mod._display_path(outside) == str(outside.resolve())


def test_main_default_scans_tracked_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real repository tree must be fully SPDX-compliant (the gate's purpose).
    assert mod.main([]) == 0
