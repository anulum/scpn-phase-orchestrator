# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — tracked ignored file guard tests

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_tracked_ignored_test_mod", TOOLS_DIR / "check_tracked_ignored.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_clean_index_passes(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(mod.subprocess, "run", return_value=_completed(0)):
        rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: no ignored files are tracked" in out


def test_tracked_ignored_paths_fail(capsys: pytest.CaptureFixture[str]) -> None:
    stdout = ".coordination/sessions/private.md\ndocs/internal/note.md\n"
    with patch.object(mod.subprocess, "run", return_value=_completed(0, stdout=stdout)):
        rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL: ignored files are tracked by Git" in out
    assert ".coordination/sessions/private.md" in out
    assert "git rm --cached" in out


def test_git_inspection_failure_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(mod.subprocess, "run", return_value=_completed(128, stderr="boom")),
        pytest.raises(SystemExit) as exc,
    ):
        mod._tracked_ignored_paths()
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "could not inspect tracked ignored files" in out
