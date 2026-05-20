# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for check_meta_distribution.py

"""Unit tests for the non-publishing scpn-meta distribution evidence guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_meta_distribution.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_check_meta_distribution_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_meta_distribution_evidence_accepts_current_project_metadata() -> None:
    evidence = mod.build_meta_distribution_evidence()

    assert evidence["schema"] == "scpn_meta_distribution_evidence_v1"
    assert evidence["project_name"] == "scpn-phase-orchestrator"
    assert evidence["version"]
    assert evidence["scripts"]["scpn-meta"] == (
        "scpn_phase_orchestrator.runtime.cli:meta_transfer_manifest"
    )
    assert evidence["scpn_meta_command_name"] == "meta-transfer-manifest"
    assert evidence["checks"] == {
        "project_name": True,
        "publishing_permitted": False,
        "scpn_meta_entry_point": True,
        "scpn_meta_importable_click_command": True,
        "spo_entry_point": True,
        "version_present": True,
    }
    assert evidence["accepted"] is True


def test_main_returns_zero_for_current_project_metadata(capsys) -> None:
    assert mod.main() == 0
    captured = capsys.readouterr()

    assert '"schema": "scpn_meta_distribution_evidence_v1"' in captured.out
    assert '"accepted": true' in captured.out


def test_meta_distribution_evidence_rejects_wrong_scpn_meta_target(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "scpn-phase-orchestrator"',
                'version = "1.0.0"',
                "[project.scripts]",
                'spo = "scpn_phase_orchestrator.runtime.cli:main"',
                'scpn-meta = "scpn_phase_orchestrator.runtime.cli:main"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    evidence = mod.build_meta_distribution_evidence(pyproject)

    assert evidence["checks"]["scpn_meta_entry_point"] is False
    assert evidence["accepted"] is False
