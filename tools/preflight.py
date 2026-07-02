#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Local CI preflight

"""Local mirror of the CI gate chain.

Runs the lint/type/test/security gates in CI order with fail-fast
semantics. ``--coverage`` appends the line-coverage lane (pytest with
``--cov`` plus the coverage guard); ``--branch-coverage`` appends the
perf-isolated branch lane (performance tests deselected, ``--cov-branch``,
gated by ``tools/coverage_guard_branch_thresholds.json``); ``--no-tests``
skips the plain pytest gate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_SRC = "src/scpn_phase_orchestrator/"
_KERNEL = "spo-kernel"
PYTEST_HEAVY_IGNORES = [
    "--ignore=tests/test_kuramoto_layer.py",
    "--ignore=tests/test_stuart_landau_nn.py",
    "--ignore=tests/test_bold.py",
    "--ignore=tests/test_reservoir.py",
    "--ignore=tests/test_ude_kuramoto.py",
    "--ignore=tests/test_inverse.py",
    "--ignore=tests/test_oim.py",
    "--ignore=tests/test_quantum_bridge_live.py",
    "--ignore=tests/test_geometry_walk.py",
    "--ignore-glob=tests/test_nn_physics_validation*.py",
]

PYTHON_GATES: list[tuple[str, list[str], Path]] = [
    (
        "pre-commit",
        ["pre-commit", "run", "--all-files"],
        ROOT,
    ),
    (
        "tracked-ignored",
        [_PY, "tools/check_tracked_ignored.py"],
        ROOT,
    ),
    (
        "ndarray-hygiene",
        [_PY, "tools/check_ndarray_type_hygiene.py"],
        ROOT,
    ),
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"], ROOT),
    (
        "ruff format",
        [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"],
        ROOT,
    ),
    ("version-sync", [_PY, "tools/check_version_sync.py"], ROOT),
    (
        "mypy",
        [_PY, "-m", "mypy", _SRC],
        ROOT,
    ),
    (
        "product-boundaries",
        [_PY, "tools/check_product_boundaries.py"],
        ROOT,
    ),
    (
        "module-linkage",
        [_PY, "tools/check_test_module_linkage.py"],
        ROOT,
    ),
    (
        "pytest",
        [
            _PY,
            "-m",
            "pytest",
            "tests/",
            "-x",
            "--tb=short",
            "-q",
            *PYTEST_HEAVY_IGNORES,
        ],
        ROOT,
    ),
    (
        "bandit",
        [_PY, "-m", "bandit", "-r", "src/", "-c", "pyproject.toml", "--quiet"],
        ROOT,
    ),
]

RUST_GATES: list[tuple[str, list[str], Path]] = [
    ("cargo fmt", ["cargo", "fmt", "--check"], ROOT / _KERNEL),
    (
        "cargo clippy",
        [
            "cargo",
            "clippy",
            "--workspace",
            "--exclude",
            "spo-ffi",
            "--",
            "-D",
            "warnings",
        ],
        ROOT / _KERNEL,
    ),
    (
        "cargo test",
        ["cargo", "test", "--workspace", "--exclude", "spo-ffi"],
        ROOT / _KERNEL,
    ),
]

COVERAGE_GATE: tuple[str, list[str], Path] = (
    "coverage (pytest)",
    [
        _PY,
        "-m",
        "pytest",
        "tests/",
        "-x",
        "--tb=short",
        "-q",
        "--cov=scpn_phase_orchestrator",
        "--cov-report=xml:coverage-python.xml",
        *PYTEST_HEAVY_IGNORES,
    ],
    ROOT,
)
COVERAGE_CHECK: tuple[str, list[str], Path] = (
    "coverage (guard)",
    [_PY, "tools/coverage_guard.py", "--coverage-xml", "coverage-python.xml"],
    ROOT,
)

# Branch instrumentation flips host-sensitive wall-clock performance tests,
# so the branch lane deselects them instead of enabling branch coverage in
# the default lanes (mirrors the CI branch-coverage job).
BRANCH_COVERAGE_GATE: tuple[str, list[str], Path] = (
    "branch coverage (pytest)",
    [
        _PY,
        "-m",
        "pytest",
        "tests/",
        "-x",
        "--tb=short",
        "-q",
        "-m",
        "not slow and not performance",
        "-k",
        "not performance",
        "--cov=scpn_phase_orchestrator",
        "--cov-branch",
        "--cov-report=xml:coverage-branch.xml",
        *PYTEST_HEAVY_IGNORES,
    ],
    ROOT,
)
BRANCH_COVERAGE_CHECK: tuple[str, list[str], Path] = (
    "branch coverage (guard)",
    [
        _PY,
        "tools/coverage_guard.py",
        "--coverage-xml",
        "coverage-branch.xml",
        "--thresholds",
        "tools/coverage_guard_branch_thresholds.json",
    ],
    ROOT,
)


def run_gate(name: str, cmd: list[str], cwd: Path) -> bool:
    """Run one gate command, print PASS/FAIL with timing, return success."""
    env = os.environ.copy()
    if "pytest" in cmd:
        src_path = str((ROOT / "src").resolve())
        existing = env.get("PYTHONPATH")
        if existing:
            env["PYTHONPATH"] = f"{src_path}:{existing}"
        else:
            env["PYTHONPATH"] = src_path

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        {line}")
    return False


def main() -> int:
    """Run the selected preflight gates; return the process exit code."""
    coverage = "--coverage" in sys.argv
    branch_coverage = "--branch-coverage" in sys.argv
    skip_tests = "--no-tests" in sys.argv
    has_cargo = shutil.which("cargo") is not None

    gates: list[tuple[str, list[str], Path]] = list(PYTHON_GATES)
    if skip_tests:
        gates = [(n, c, d) for n, c, d in gates if n != "pytest"]
    if has_cargo:
        gates.extend(RUST_GATES)

    parts = [f"{len(gates)} gates"]
    if not has_cargo:
        parts.append("no cargo")
    if coverage:
        parts.append("coverage")
    if branch_coverage:
        parts.append("branch coverage")
    print(f"preflight: {', '.join(parts)}")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd, cwd in gates:
        if not run_gate(name, cmd, cwd):
            failed.append(name)
            break  # fail fast

    if not failed and coverage:
        if not run_gate(*COVERAGE_GATE):
            failed.append("coverage (pytest --cov)")
        elif not run_gate(*COVERAGE_CHECK):
            failed.append("coverage (guard)")

    if not failed and branch_coverage:
        if not run_gate(*BRANCH_COVERAGE_GATE):
            failed.append("branch coverage (pytest --cov-branch)")
        elif not run_gate(*BRANCH_COVERAGE_CHECK):
            failed.append("branch coverage (guard)")

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
