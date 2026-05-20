# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Packaging script metadata tests

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import click
import tomllib
from click.testing import CliRunner


def test_project_scripts_expose_review_only_meta_console_surface() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["spo"] == "scpn_phase_orchestrator.runtime.cli:main"
    assert (
        scripts["scpn-meta"]
        == "scpn_phase_orchestrator.runtime.cli:meta_transfer_manifest"
    )


def test_project_script_targets_are_importable_click_commands() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    resolved = {
        name: _load_entry_point(target) for name, target in scripts.items()
    }

    assert isinstance(resolved["spo"], click.Group)
    assert resolved["spo"].name == "main"
    assert isinstance(resolved["scpn-meta"], click.Command)
    assert resolved["scpn-meta"].name == "meta-transfer-manifest"


def test_scpn_meta_entry_point_invokes_review_only_manifest_export(
    tmp_path: Path,
) -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    command = _load_entry_point(data["project"]["scripts"]["scpn-meta"])
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        json.dumps(
            {
                "domain": "power_grid",
                "features": {"coherence": 0.8, "event_rate": 0.2},
                "knobs": {"K": 0.04, "zeta": 0.06},
                "reward": 0.9,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(command, [str(audit_path), "--min-records", "1"])

    assert result.exit_code == 0
    manifest = json.loads(result.output)
    assert manifest["schema"] == "scpn_meta_package_manifest_v1"
    assert manifest["console_script"] == "scpn-meta"
    assert manifest["execution_permitted"] is False
    assert manifest["training_summary"]["record_count"] == 1


def _load_entry_point(target: str) -> object:
    module_name, separator, attribute_name = target.partition(":")
    if not separator:
        raise AssertionError(f"entry point target lacks ':' separator: {target}")
    module = import_module(module_name)
    return getattr(module, attribute_name)
