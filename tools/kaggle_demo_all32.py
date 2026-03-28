#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kaggle: demo all 32 domainpacks
#
# Reproduces the benchmark table in docs/galleries/domainpack_gallery.md.
# Run on Kaggle (free CPU kernel, no GPU needed) or any Linux with Python 3.10+.
#
# Usage (local):
#   pip install -e .
#   python tools/kaggle_demo_all32.py
#
# Usage (Kaggle):
#   Upload as script kernel with internet enabled.

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    # If running on Kaggle, install from git
    if not os.path.exists("domainpacks"):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numpy",
                "scipy",
                "click",
                "pyyaml",
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

    domains = sorted(
        d
        for d in os.listdir("domainpacks")
        if os.path.isdir(f"domainpacks/{d}")
        and os.path.exists(f"domainpacks/{d}/binding_spec.yaml")
    )

    print(f"Testing {len(domains)} domainpacks\n")
    print(f"{'Domain':<28s} {'Osc':>4s} {'Lay':>4s} {'R':>6s} {'Regime':<12s}")
    print("-" * 60)

    for domain in domains:
        result = subprocess.run(
            ["spo", "demo", "--domain", domain, "--steps", "20"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            osc = lay = r_val = regime = "?"
            for line in lines:
                if "Oscillators:" in line:
                    osc = line.split(":")[1].strip()
                if "Layers:" in line:
                    lay = line.split(":")[1].strip()
                if "Final R=" in line:
                    parts = line.split(",")
                    r_val = parts[0].split("=")[1]
                    regime = parts[1].split("=")[1]
            print(f"{domain:<28s} {osc:>4s} {lay:>4s} {r_val:>6s} {regime:<12s}")
        else:
            err = result.stderr.split("\n")[-2] if result.stderr else "unknown"
            print(f"{domain:<28s}  FAIL: {err[:40]}")

    print(f"\nDONE — all {len(domains)} domainpacks tested")


if __name__ == "__main__":
    main()
