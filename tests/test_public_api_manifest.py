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
PHA_C_ACCEPTANCE_REFERENCE = Path("docs/reference/api/upde_pha_c_acceptance.md")
README = Path("README.md")
PERFORMANCE_GUIDE = Path("docs/guide/performance.md")


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


def test_api_index_surfaces_pha_c_acceptance_certificate_chain() -> None:
    api_index = API_INDEX.read_text(encoding="utf-8")

    required_markers = (
        "PHA-C acceptance evidence chain",
        "KinematicBounds.acceptanceCertificate",
        "acceptance_certificate_discharges_runtime_preconditions",
        "final-position, maximum-velocity, path-length",
        "Rust, Go, Julia,",
        "benchmark-isolation protocol",
    )

    for marker in required_markers:
        assert marker in api_index


def test_pha_c_acceptance_reference_names_combined_lean_certificate() -> None:
    reference = PHA_C_ACCEPTANCE_REFERENCE.read_text(encoding="utf-8")

    required_markers = (
        "KinematicBounds.acceptanceCertificate",
        "acceptance_certificate_discharges_runtime_preconditions",
        "acceptance kinematic-equation replay certificate",
        "combined spatial-budget, phase-budget, and acceptance-replay certificate",
    )

    for marker in required_markers:
        assert marker in reference


def test_public_pha_c_docs_name_combined_acceptance_boundary() -> None:
    required_markers = (
        "KinematicBounds.acceptanceCertificate",
        "acceptance_certificate_discharges_runtime_preconditions",
        "spatial Gronwall budget",
        "phase-budget certificate",
        "kinematic-equation replay certificate",
    )

    for path in (README, PERFORMANCE_GUIDE):
        text = " ".join(path.read_text(encoding="utf-8").split())
        for marker in required_markers:
            assert marker in text
