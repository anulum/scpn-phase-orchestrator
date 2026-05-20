# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Packaging script metadata tests

from __future__ import annotations

from pathlib import Path

import tomllib


def test_project_scripts_expose_review_only_meta_console_surface() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["spo"] == "scpn_phase_orchestrator.runtime.cli:main"
    assert (
        scripts["scpn-meta"]
        == "scpn_phase_orchestrator.runtime.cli:meta_transfer_manifest"
    )
