# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — type ignore justification regression

from __future__ import annotations

from pathlib import Path


def _has_type_ignore_reason(lines: list[str], index: int) -> bool:
    line = lines[index]
    directive = "# type: ignore"
    suffix = line.split(directive, 1)[1]
    if "type ignore:" in line:
        return True
    if "#" in suffix and "pragma:" not in suffix:
        return True
    previous = lines[max(0, index - 2) : index]
    return any("type ignore:" in item for item in previous)


def test_source_type_ignores_have_local_justification() -> None:
    missing: list[str] = []
    for path in sorted(Path("src/scpn_phase_orchestrator").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "# type: ignore" not in line:
                continue
            if not _has_type_ignore_reason(lines, index):
                missing.append(f"{path}:{index + 1}")

    assert missing == []
