#!/usr/bin/env python3
# SCPN Phase Orchestrator — Local CI preflight
# Mirrors every Python-side CI gate so failures are caught before push.
# © 1998–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_SRC = "src/scpn_phase_orchestrator/"

GATES: list[tuple[str, list[str]]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"]),
    (
        "ruff format",
        [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"],
    ),
    ("version-sync", [_PY, "tools/check_version_sync.py"]),
    (
        "mypy",
        [_PY, "-m", "mypy", _SRC, "--ignore-missing-imports"],
    ),
    ("module-linkage", [_PY, "tools/check_test_module_linkage.py"]),
    (
        "pytest",
        [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
    ),
    (
        "bandit",
        [_PY, "-m", "bandit", "-r", "src/", "-c", "pyproject.toml", "--quiet"],
    ),
]

COVERAGE_GATE: tuple[str, list[str]] = (
    "coverage-guard",
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
    ],
)
COVERAGE_CHECK: tuple[str, list[str]] = (
    "coverage-guard (check)",
    [
        _PY,
        "tools/coverage_guard.py",
        "--coverage-xml",
        "coverage-python.xml",
    ],
)


def run_gate(name: str, cmd: list[str]) -> bool:
    t0 = time.monotonic()
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
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
    coverage = "--coverage" in sys.argv
    skip_tests = "--no-tests" in sys.argv

    gates = list(GATES)
    if skip_tests:
        gates = [(n, c) for n, c in gates if n != "pytest"]

    suffix = " + coverage" if coverage else ""
    print(f"preflight: {len(gates)} gates{suffix}")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd in gates:
        if not run_gate(name, cmd):
            failed.append(name)
            break  # fail fast

    if not failed and coverage:
        if not run_gate(*COVERAGE_GATE):
            failed.append("coverage-guard (pytest --cov)")
        elif not run_gate(*COVERAGE_CHECK):
            failed.append("coverage-guard (check)")

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
