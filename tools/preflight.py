#!/usr/bin/env python3
# SCPN Phase Orchestrator — Local CI preflight
# Mirrors every CI gate so failures are caught before push.
# © 1998–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

import shutil
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_SRC = "src/scpn_phase_orchestrator/"
_KERNEL = "spo-kernel"

PYTHON_GATES: list[tuple[str, list[str], Path]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"], ROOT),
    (
        "ruff format",
        [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"],
        ROOT,
    ),
    ("version-sync", [_PY, "tools/check_version_sync.py"], ROOT),
    (
        "mypy",
        [_PY, "-m", "mypy", _SRC, "--ignore-missing-imports"],
        ROOT,
    ),
    (
        "module-linkage",
        [_PY, "tools/check_test_module_linkage.py"],
        ROOT,
    ),
    (
        "pytest",
        [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
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
    ],
    ROOT,
)
COVERAGE_CHECK: tuple[str, list[str], Path] = (
    "coverage (guard)",
    [_PY, "tools/coverage_guard.py", "--coverage-xml", "coverage-python.xml"],
    ROOT,
)


def run_gate(name: str, cmd: list[str], cwd: Path) -> bool:
    t0 = time.monotonic()
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
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

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
