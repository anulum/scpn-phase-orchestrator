# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API manifest regressions

from __future__ import annotations

from pathlib import Path

import scpn_phase_orchestrator as spo

MANIFEST = Path("docs/specs/public_api_manifest.txt")
API_INDEX = Path("docs/reference/api/index.md")


def _manifest_exports() -> list[str]:
    exports: list[str] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            exports.append(stripped)
    return exports


def test_top_level_public_api_matches_freeze_manifest() -> None:
    assert sorted(spo.__all__) == sorted(_manifest_exports())


def test_freeze_manifest_keeps_all_exports_resolvable() -> None:
    for name in _manifest_exports():
        assert getattr(spo, name) is not None


def test_api_index_lists_every_top_level_manifest_export() -> None:
    api_index = API_INDEX.read_text(encoding="utf-8")

    for name in _manifest_exports():
        assert name in api_index
