#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Version sync guard

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _extract(path: Path, pattern: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def main() -> int:
    pyproject = _extract(ROOT / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    citation = _extract(ROOT / "CITATION.cff", r"^version:\s*(\S+)")
    cargo = _extract(ROOT / "spo-kernel" / "Cargo.toml", r'^version\s*=\s*"([^"]+)"')

    versions = {
        "pyproject.toml": pyproject,
        "CITATION.cff": citation,
        "Cargo.toml": cargo,
    }
    missing = [k for k, v in versions.items() if v is None]
    if missing:
        print(f"FAIL: could not extract version from: {', '.join(missing)}")
        return 1

    unique = set(versions.values())
    if len(unique) != 1:
        for name, ver in versions.items():
            print(f"  {name}: {ver}")
        print("FAIL: version mismatch")
        return 1

    print(f"OK: all versions = {unique.pop()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
