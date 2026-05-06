#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Release tag version guard

"""Fail release workflows when the Git tag and Python package version differ."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_version(pyproject: Path) -> str | None:
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def _release_tag_version(ref_name: str | None) -> str | None:
    if not ref_name:
        return None
    match = re.fullmatch(r"v(\d+(?:\.\d+){2}(?:[A-Za-z0-9.+-]+)?)", ref_name)
    return match.group(1) if match else None


def main() -> int:
    ref_name = os.environ.get("GITHUB_REF_NAME")
    tag_version = _release_tag_version(ref_name)
    if tag_version is None:
        print(f"FAIL: expected release tag like vX.Y.Z, got {ref_name!r}")
        return 1

    package_version = _project_version(ROOT / "pyproject.toml")
    if package_version is None:
        print("FAIL: could not extract project.version from pyproject.toml")
        return 1

    if tag_version != package_version:
        print(
            "FAIL: release tag/package version mismatch: "
            f"{ref_name} != {package_version}"
        )
        return 1

    print(f"OK: release tag {ref_name} matches package version {package_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
