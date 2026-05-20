#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tracked ignored file guard

"""Fail when ignored local-only paths have already entered the Git index."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent.parent


def _tracked_ignored_paths() -> list[str]:
    git = which("git")
    if git is None:
        print("FAIL: could not inspect tracked ignored files: git not found")
        raise SystemExit(1)
    result = subprocess.run(
        [git, "ls-files", "-ci", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"FAIL: could not inspect tracked ignored files: {stderr}")
        raise SystemExit(1)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    paths = _tracked_ignored_paths()
    if not paths:
        print("OK: no ignored files are tracked")
        return 0

    print("FAIL: ignored files are tracked by Git")
    print()
    for path in paths:
        print(f"  {path}")
    print()
    print("Remove these paths from the index with `git rm --cached -- <path>`.")
    print("Do not delete local session logs or internal notes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
