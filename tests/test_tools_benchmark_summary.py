# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for tools/benchmark_summary.py

"""Unit tests for ``tools/benchmark_summary.py``.

The script fans out to the Python-side benchmark scripts under
``benchmarks/``. Tests verify the two pure-logic surfaces with
``subprocess.run`` patched out:

* ``run_bench`` — happy path returns (stdout, seconds); the two
  error surfaces (``CalledProcessError``, ``FileNotFoundError``)
  map to an ``"Error ..."`` string rather than propagating the
  exception.
* ``main`` — constructs PYTHONPATH (prepending ``src/`` either as
  the whole value or concatenated with the inherited one), iterates
  the benchmark list, propagates exit code 1 whenever any run
  returns an error string.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_benchmark_summary_test_mod", TOOLS_DIR / "benchmark_summary.py"
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
# run_bench
# ---------------------------------------------------------------------


class TestRunBench:
    def test_happy_path_returns_stripped_stdout(self) -> None:
        with patch.object(
            mod.subprocess, "run", return_value=_completed(0, stdout="  10 Mflop/s\n")
        ):
            out, elapsed = mod.run_bench(["python", "x.py"], env={})
        assert out == "10 Mflop/s"
        assert elapsed >= 0.0

    def test_called_process_error_returns_error_string(self) -> None:
        err = subprocess.CalledProcessError(
            returncode=2, cmd=["python", "x.py"], stderr="  boom\n"
        )
        with patch.object(mod.subprocess, "run", side_effect=err):
            out, elapsed = mod.run_bench(["python", "x.py"], env={})
        assert out.startswith("Error (exit 2):")
        assert "boom" in out
        assert elapsed >= 0.0

    def test_file_not_found_returns_error_string(self) -> None:
        with patch.object(
            mod.subprocess,
            "run",
            side_effect=FileNotFoundError("no such file: xyz"),
        ):
            out, _ = mod.run_bench(["python", "xyz.py"], env={})
        assert out.startswith("Error: benchmark script not found")
        assert "xyz" in out

    def test_args_and_env_forwarded_to_subprocess(self, tmp_path: Path) -> None:
        env = {"PYTHONPATH": str(tmp_path / "src")}
        with patch.object(
            mod.subprocess, "run", return_value=_completed(0)
        ) as run:
            mod.run_bench(["python", "-c", "pass"], env=env)
        call = run.call_args
        assert call.args[0] == ["python", "-c", "pass"]
        assert call.kwargs["env"] == env
        assert call.kwargs["capture_output"] is True
        assert call.kwargs["check"] is True


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


class TestMain:
    def test_all_benchmarks_pass_returns_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(
            mod.subprocess, "run", return_value=_completed(0, stdout="ok")
        ):
            rc = mod.main()
        out = capsys.readouterr().out
        assert rc == 0
        # All 9 hardcoded benchmarks run.
        assert out.count("Running ") == 9

    def test_any_failure_returns_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A single failing sub-benchmark → exit 1; subsequent runs
        still execute (no fail-fast at this layer)."""

        def fake_run(
            args: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            if "sparse_benchmark.py" in args[1]:
                raise subprocess.CalledProcessError(
                    returncode=1, cmd=args, stderr="nope"
                )
            return _completed(0, stdout="ok")

        with patch.object(mod.subprocess, "run", side_effect=fake_run):
            rc = mod.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "Error (exit 1)" in out
        # Subsequent benchmarks still ran — we saw all 9 headers.
        assert out.count("Running ") == 9

    def test_pythonpath_prepends_to_existing(self) -> None:
        """If PYTHONPATH is already set, ``src/`` is prepended."""
        captured_envs: list[dict[str, str]] = []

        def fake_run(
            _: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            captured_envs.append(env)
            return _completed(0)

        old_env = os.environ.copy()
        try:
            os.environ["PYTHONPATH"] = "/existing/path"
            with patch.object(mod.subprocess, "run", side_effect=fake_run):
                mod.main()
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert captured_envs, "subprocess.run was not invoked"
        pp = captured_envs[0]["PYTHONPATH"]
        assert pp.endswith(f"{os.pathsep}/existing/path")
        assert pp.split(os.pathsep)[0].endswith("src")

    def test_pythonpath_without_existing_uses_src_only(self) -> None:
        captured_envs: list[dict[str, str]] = []

        def fake_run(
            _: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            captured_envs.append(env)
            return _completed(0)

        old_env = os.environ.copy()
        try:
            os.environ.pop("PYTHONPATH", None)
            with patch.object(mod.subprocess, "run", side_effect=fake_run):
                mod.main()
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert captured_envs
        assert captured_envs[0]["PYTHONPATH"].endswith("src")
        # No separator — just a single src path.
        assert os.pathsep not in captured_envs[0]["PYTHONPATH"]

    def test_all_nine_benchmarks_are_enumerated(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression guard: the benchmark list in main() must cover
        the nine expected modules."""
        with patch.object(mod.subprocess, "run", return_value=_completed(0)):
            mod.main()
        out = capsys.readouterr().out
        for marker in (
            "UPDE (Dense",
            "Sparse UPDE",
            "Stuart-Landau",
            "Simplicial",
            "Hypergraph",
            "Inertial",
            "Delayed",
            "Swarmalator",
            "Recurrence",
        ):
            assert marker in out, f"Missing benchmark group {marker!r}"
