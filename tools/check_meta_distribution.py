#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scpn-meta distribution evidence guard

"""Emit non-publishing release evidence for the optional scpn-meta surface."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import click
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src"
EXPECTED_PROJECT_NAME = "scpn-phase-orchestrator"
EXPECTED_META_SCRIPT = "scpn_phase_orchestrator.runtime.cli:meta_transfer_manifest"


def _pyproject_data(path: Path = PYPROJECT) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _resolve_entry_point(target: str) -> object:
    module_name, separator, attribute_name = target.partition(":")
    if not separator:
        raise ValueError(f"entry point target lacks ':' separator: {target}")
    sys.path.insert(0, str(SRC))
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)
    finally:
        with suppress(ValueError):
            sys.path.remove(str(SRC))


def build_meta_distribution_evidence(
    path: Path = PYPROJECT,
) -> dict[str, Any]:
    data = _pyproject_data(path)
    project = data.get("project", {})
    scripts = project.get("scripts", {})
    if not isinstance(project, dict):
        raise ValueError("[project] must be a table")
    if not isinstance(scripts, dict):
        raise ValueError("[project.scripts] must be a table")

    project_name = str(project.get("name", ""))
    version = str(project.get("version", ""))
    meta_target = str(scripts.get("scpn-meta", ""))
    spo_target = str(scripts.get("spo", ""))
    meta_command = _resolve_entry_point(meta_target)
    spo_command = _resolve_entry_point(spo_target)
    meta_is_command = isinstance(meta_command, click.Command)
    spo_is_group = isinstance(spo_command, click.Group)
    checks = {
        "project_name": project_name == EXPECTED_PROJECT_NAME,
        "version_present": bool(version),
        "spo_entry_point": spo_is_group,
        "scpn_meta_entry_point": meta_target == EXPECTED_META_SCRIPT,
        "scpn_meta_importable_click_command": meta_is_command,
        "publishing_permitted": False,
    }
    accepted = (
        all(
            value is True
            for key, value in checks.items()
            if key != "publishing_permitted"
        )
        and checks["publishing_permitted"] is False
    )
    return {
        "schema": "scpn_meta_distribution_evidence_v1",
        "project_name": project_name,
        "version": version,
        "scripts": {
            "spo": spo_target,
            "scpn-meta": meta_target,
        },
        "scpn_meta_command_name": (
            str(meta_command.name) if meta_is_command else ""
        ),
        "checks": checks,
        "accepted": accepted,
    }


def main() -> int:
    try:
        evidence = build_meta_distribution_evidence()
    except (
        OSError,
        ValueError,
        AttributeError,
        ImportError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
