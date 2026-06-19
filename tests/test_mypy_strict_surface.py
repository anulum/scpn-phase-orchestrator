# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — mypy strict-surface regression guard

"""Guard the committed mypy strict-surface snapshot against silent drift.

The snapshot enumerates the base ``strict`` flag and every
``[[tool.mypy.overrides]]`` relaxation. If a future change adds, widens, or
removes a relaxation without updating the snapshot, this test fails — making
strict-coverage changes a deliberate, reviewed act instead of a silent one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOL_PATH = _REPO_ROOT / "tools" / "mypy_strict_surface.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mypy_strict_surface", _TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_surface_matches_committed_snapshot() -> None:
    tool = _load_tool()
    surface = tool.compute_strict_surface(tool.PYPROJECT_PATH)
    rendered = tool.render_snapshot(surface)
    committed = tool.SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == committed, (
        "mypy strict surface drifted from tests/data/mypy_strict_surface.json; "
        "review the added or widened relaxation and, if intentional, run "
        "`python tools/mypy_strict_surface.py --write` to refresh the snapshot."
    )


def test_base_strict_flag_is_enabled() -> None:
    tool = _load_tool()
    surface = tool.compute_strict_surface(tool.PYPROJECT_PATH)
    assert surface["strict"] is True


def test_no_first_party_module_disables_type_checking() -> None:
    """No shipped ``scpn_phase_orchestrator`` module may set ``ignore_errors``.

    Generated gRPC stubs are the sole exception; everything else must stay
    type-checked.
    """
    tool = _load_tool()
    surface = tool.compute_strict_surface(tool.PYPROJECT_PATH)
    for override in surface["overrides"]:
        if not override["settings"].get("ignore_errors"):
            continue
        for module in override["modules"]:
            if not module.startswith("scpn_phase_orchestrator."):
                continue
            assert "grpc_gen" in module, (
                f"{module} disables type checking via ignore_errors; only "
                "generated grpc_gen stubs may do so."
            )
