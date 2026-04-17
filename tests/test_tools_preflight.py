# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for tools/preflight.py

"""Unit tests for ``tools/preflight.py``.

Local CI orchestrator that chains ruff, ruff-format, version-sync,
mypy, module-linkage, pytest, bandit, and the cargo gates. Tests run
the two tight behavioural surfaces that don't need a full repo
checkout: ``run_gate`` output formatting + return value, and
``main`` flag parsing + gate assembly (with subprocess patched).
"""

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
        "_preflight_test_mod", TOOLS_DIR / "preflight.py"
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
        args=["dummy"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------
# run_gate
# ---------------------------------------------------------------------


class TestRunGate:
    def test_pass_prints_pass_and_returns_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(mod.subprocess, "run", return_value=_completed(0)):
            ok = mod.run_gate("ruff", ["dummy"], tmp_path)
        out = capsys.readouterr().out
        assert ok is True
        assert "PASS" in out
        assert "ruff" in out

    def test_fail_prints_fail_and_returns_false(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        proc = _completed(1, stdout="compile error", stderr="traceback here")
        with patch.object(mod.subprocess, "run", return_value=proc):
            ok = mod.run_gate("mypy", ["dummy"], tmp_path)
        out = capsys.readouterr().out
        assert ok is False
        assert "FAIL" in out
        assert "mypy" in out
        # Last 10 stdout + stderr lines are echoed on failure.
        assert "compile error" in out
        assert "traceback here" in out

    def test_fail_truncates_output_to_last_ten_lines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        long_stdout = "\n".join(f"line {i}" for i in range(40))
        proc = _completed(1, stdout=long_stdout)
        with patch.object(mod.subprocess, "run", return_value=proc):
            mod.run_gate("pytest", ["dummy"], tmp_path)
        out = capsys.readouterr().out
        # First 30 lines must not leak through.
        assert "line 0" not in out
        assert "line 29" not in out
        # Last 10 lines (30..39) should appear.
        assert "line 39" in out
        assert "line 30" in out

    def test_empty_stdout_stderr_no_body_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        proc = _completed(7, stdout="", stderr="")
        with patch.object(mod.subprocess, "run", return_value=proc):
            ok = mod.run_gate("silent", ["dummy"], tmp_path)
        out = capsys.readouterr().out
        assert ok is False
        assert "FAIL" in out
        # Header line is the only output when both streams are empty.
        assert len(out.strip().splitlines()) == 1

    def test_subprocess_run_receives_expected_args(self, tmp_path: Path) -> None:
        with patch.object(
            mod.subprocess, "run", return_value=_completed(0)
        ) as run:
            mod.run_gate("cargo-fmt", ["cargo", "fmt", "--check"], tmp_path)
        kwargs = run.call_args.kwargs
        assert kwargs["cwd"] == tmp_path
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


class TestMain:
    def test_all_pass_returns_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every gate passes → exit 0 + ALL CLEAR banner."""
        with (
            patch.object(mod.shutil, "which", return_value=None),  # no cargo
            patch.object(mod.subprocess, "run", return_value=_completed(0)),
            patch.object(sys, "argv", ["preflight", "--no-tests"]),
        ):
            rc = mod.main()
        out = capsys.readouterr().out
        assert rc == 0
        assert "ALL CLEAR" in out

    def test_first_failure_fails_fast(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failing gate short-circuits the rest (fail-fast)."""
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            # First call fails; subsequent would pass but shouldn't run.
            return _completed(1 if len(calls) == 1 else 0, stderr="boom")

        with (
            patch.object(mod.shutil, "which", return_value=None),
            patch.object(mod.subprocess, "run", side_effect=fake_run),
            patch.object(sys, "argv", ["preflight", "--no-tests"]),
        ):
            rc = mod.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "BLOCKED" in out
        # Only the first gate ran; the rest were skipped.
        assert len(calls) == 1

    def test_no_tests_flag_removes_pytest_gate(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return _completed(0)

        with (
            patch.object(mod.shutil, "which", return_value=None),
            patch.object(mod.subprocess, "run", side_effect=fake_run),
            patch.object(sys, "argv", ["preflight", "--no-tests"]),
        ):
            mod.main()
        # No gate should have invoked pytest.
        assert not any("pytest" in " ".join(c) for c in calls)

    def test_without_cargo_skips_rust_gates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return _completed(0)

        with (
            patch.object(mod.shutil, "which", return_value=None),
            patch.object(mod.subprocess, "run", side_effect=fake_run),
            patch.object(sys, "argv", ["preflight", "--no-tests"]),
        ):
            mod.main()
        out = capsys.readouterr().out
        assert "no cargo" in out
        assert not any(c and c[0] == "cargo" for c in calls)

    def test_with_cargo_includes_rust_gates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return _completed(0)

        with (
            patch.object(mod.shutil, "which", return_value="/usr/bin/cargo"),
            patch.object(mod.subprocess, "run", side_effect=fake_run),
            patch.object(sys, "argv", ["preflight", "--no-tests"]),
        ):
            mod.main()
        cargo_calls = [c for c in calls if c and c[0] == "cargo"]
        # fmt + clippy + test = 3 cargo gates.
        assert len(cargo_calls) == 3
        names = {c[1] for c in cargo_calls}
        assert names == {"fmt", "clippy", "test"}

    def test_coverage_flag_runs_extra_gates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--coverage adds both the pytest --cov run and the guard."""
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            return _completed(0)

        with (
            patch.object(mod.shutil, "which", return_value=None),
            patch.object(mod.subprocess, "run", side_effect=fake_run),
            patch.object(sys, "argv", ["preflight", "--no-tests", "--coverage"]),
        ):
            rc = mod.main()
        assert rc == 0
        flattened = [" ".join(c) for c in calls]
        assert any("--cov-report=xml" in s for s in flattened)
        assert any("tools/coverage_guard.py" in s for s in flattened)
