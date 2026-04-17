#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kaggle: mutation testing with mutmut
#
# Reproduces the mutation testing results in docs/VALIDATION_REPORT.md.
# Run on Kaggle (free CPU kernel, no GPU needed) or any Linux with Python 3.10+.
# mutmut does not support Windows natively.
#
# Usage (local Linux/WSL):
#   pip install -e ".[dev]" mutmut==2.4.5
#   python tools/kaggle_mutation_test.py
#
# Usage (Kaggle):
#   Upload as script kernel with internet enabled.

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    # If running on Kaggle, install from git
    if not os.path.exists("src"):
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "mutmut"],
            capture_output=True,
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "mutmut==2.4.5"]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "pytest",
                "hypothesis",
                "numpy",
                "scipy",
            ]
        )
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/anulum/scpn-phase-orchestrator.git",
                "spo",
            ]
        )
        os.chdir("spo")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"]
        )

    # Module → targeted test file (fast, no hypothesis)
    targets = [
        (
            "src/scpn_phase_orchestrator/upde/order_params.py",
            "tests/test_mutation_killers.py",
        ),
        (
            "src/scpn_phase_orchestrator/upde/numerics.py",
            "tests/test_upde_math.py",
        ),
        (
            "src/scpn_phase_orchestrator/monitor/npe.py",
            "tests/test_dimension.py",
        ),
        (
            "src/scpn_phase_orchestrator/coupling/spectral.py",
            "tests/test_coupling_modules.py",
        ),
    ]

    for mod, tests in targets:
        print(f"\n{'=' * 60}")
        print(f"MUTATION TESTING: {mod}")
        print(f"{'=' * 60}", flush=True)

        subprocess.run(["rm", "-rf", ".mutmut-cache"], capture_output=True)

        runner = (
            f"{sys.executable} -m pytest {tests} "
            f"-x -q --tb=no -p no:cacheprovider -p no:hypothesis"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mutmut",
                "run",
                "--paths-to-mutate",
                mod,
                "--tests-dir",
                "tests/",
                "--runner",
                runner,
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(result.stdout[-3000:] if result.stdout else "(no stdout)")
        if result.stderr:
            print("STDERR:", result.stderr[-500:])

        res = subprocess.run(
            [sys.executable, "-m", "mutmut", "results"],
            capture_output=True,
            text=True,
        )
        print(res.stdout[-2000:] if res.stdout else "(no results)")

    print("\n\nDONE — mutation testing complete")


if __name__ == "__main__":
    main()
