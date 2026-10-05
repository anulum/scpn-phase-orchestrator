# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — subprocess coverage integration

"""Verify real child-process execution survives collection and CI data merging.

The driver exercises production token-bucket requests in an isolated interpreter.
Neither the measured parent nor this test imports the limiter. These are tests
of the coverage transport contract, not substitutes for limiter behaviour tests.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from coverage import CoverageData

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "pyproject.toml"
_DRIVER = _ROOT / "tests/fixtures/coverage_process_driver.py"
# The child imports the installed package, which is the checkout source in an
# editable install and site-packages in a wheel install. Locating the package
# does not import it.
_PACKAGE = importlib.util.find_spec("scpn_phase_orchestrator")
assert _PACKAGE is not None and _PACKAGE.origin is not None
_RUNTIME = Path(_PACKAGE.origin).resolve().parent / "runtime"
_LIMITER = _RUNTIME / "network_security.py"


def _run(command: list[str], directory: Path) -> None:
    """Run an independent measurement without inheriting an outer collector."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "COV_CORE_")) and key != "PYTEST_ADDOPTS"
    }
    environment["COVERAGE_FILE"] = str(directory / ".coverage")
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("collector", ["coverage", "pytest-cov"])
@pytest.mark.parametrize("branch", [False, True])
def test_child_execution_survives_ci_coverage_merge(
    tmp_path: Path, collector: str, branch: bool
) -> None:
    """Both collectors preserve isolated-child lines and optional branch arcs."""
    python = [sys.executable, "-m"]
    if collector == "coverage":
        command = [
            *python,
            "coverage",
            "run",
            f"--rcfile={_CONFIG}",
            f"--source={_RUNTIME}",
            *(["--branch"] if branch else []),
            str(_DRIVER),
        ]
        _run(command, tmp_path)
        parts = list(tmp_path.glob(".coverage.*"))
        assert len(parts) >= 2, "parent and child need separate data files"
        _run([*python, "coverage", "combine"], tmp_path)
    else:
        _run(
            [
                *python,
                "pytest",
                "-q",
                "-c",
                str(_CONFIG),
                str(_DRIVER),
                f"--cov={_RUNTIME}",
                f"--cov-config={_CONFIG}",
                "--cov-report=",
                # CI collects partial lanes at zero; its later merged guard
                # enforces the unchanged repository/module thresholds.
                "--cov-fail-under=0",
                *(["--cov-branch"] if branch else []),
            ],
            tmp_path,
        )
        assert not list(tmp_path.glob(".coverage.*"))

    # CI uploads the pytest-cov combined file under a lane-specific name.
    artifact = tmp_path / ".coverage.test"
    (tmp_path / ".coverage").rename(artifact)
    _run([*python, "coverage", "combine", str(artifact)], tmp_path)
    data = CoverageData(basename=str(tmp_path / ".coverage"))
    data.read()
    measured = next(
        name for name in data.measured_files() if Path(name).resolve() == _LIMITER
    )
    helper = next(
        node
        for node in ast.parse(_LIMITER.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_elapsed_time"
    )
    error_line = next(
        node.lineno for node in ast.walk(helper) if isinstance(node, ast.Raise)
    )
    return_line = next(
        node.lineno for node in ast.walk(helper) if isinstance(node, ast.Return)
    )
    assert {error_line, return_line} <= set(data.lines(measured) or [])
    assert data.has_arcs() is branch
    if branch:
        arcs = data.arcs(measured) or []
        assert any(end == error_line for _, end in arcs)
        assert any(end == return_line for _, end in arcs)
