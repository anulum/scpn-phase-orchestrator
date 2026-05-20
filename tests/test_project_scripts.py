# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Packaging script metadata tests

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import click
import tomllib


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


def _load_entry_point(target: str) -> object:
    module_name, separator, attribute_name = target.partition(":")
    if not separator:
        raise AssertionError(f"entry point target lacks ':' separator: {target}")
    module = import_module(module_name)
    return getattr(module, attribute_name)
