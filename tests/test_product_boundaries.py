# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Product boundary tests

"""Regression tests for the product-boundary import guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_product_boundaries.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_product_boundary_guard", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()
ORIGINAL_SRC_ROOT = mod.SRC_ROOT


def _write_module(root: Path, rel: str, text: str) -> Path:
    mod.SRC_ROOT = root
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_core_package_imports_can_target_other_core_modules(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "binding/loader.py",
        "from scpn_phase_orchestrator.upde.engine import UPDEEngine\n",
    )

    assert mod.find_violations([core]) == []


def test_core_package_cannot_import_runtime_surface(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "upde/engine.py",
        "from scpn_phase_orchestrator.server import SimulationState\n",
    )

    violations = mod.find_violations([core])

    assert len(violations) == 1
    assert violations[0].target_boundary == "runtime"
    assert violations[0].imported_module == "scpn_phase_orchestrator.server"


def test_core_package_relative_import_cannot_target_runtime_surface(
    tmp_path: Path,
) -> None:
    core = _write_module(
        tmp_path,
        "upde/engine.py",
        "from ..server import create_app\n",
    )

    violations = mod.find_violations([core])

    assert len(violations) == 1
    assert violations[0].target_boundary == "runtime"
    assert violations[0].imported_module == "scpn_phase_orchestrator.server"


def test_core_package_cannot_import_integration_surface(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "monitor/boundaries.py",
        "from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter\n",
    )

    violations = mod.find_violations([core])

    assert len(violations) == 1
    assert violations[0].target_boundary == "integrations"


def test_language_shims_are_experimental_even_inside_core_dirs() -> None:
    assert mod.classify_module("scpn_phase_orchestrator.upde._engine_mojo") == (
        "experimental"
    )
    assert mod.classify_module("scpn_phase_orchestrator.coupling._spectral_go") == (
        "experimental"
    )


def test_new_core_to_experimental_imports_fail_without_legacy_allowlist(
    tmp_path: Path,
) -> None:
    core = _write_module(
        tmp_path,
        "upde/new_engine.py",
        "from scpn_phase_orchestrator.nn.ude import UDEKuramotoLayer\n",
    )

    violations = mod.find_violations([core])

    assert len(violations) == 1
    assert violations[0].target_boundary == "experimental"


def test_legacy_allowlist_usage_is_measured(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "upde/engine.py",
        "from scpn_phase_orchestrator.upde._engine_mojo import upde_run_mojo\n",
    )

    used = mod.find_legacy_accelerator_imports([core])

    assert used == {"scpn_phase_orchestrator.upde._engine_mojo"}
    assert mod.find_violations([core]) == []


def test_current_source_tree_respects_core_boundary_contract() -> None:
    mod.SRC_ROOT = ORIGINAL_SRC_ROOT
    violations = mod.find_violations(mod.iter_python_files())

    assert violations == []


def test_current_legacy_allowlist_has_no_stale_entries() -> None:
    mod.SRC_ROOT = ORIGINAL_SRC_ROOT
    used = mod.find_legacy_accelerator_imports(mod.iter_python_files())

    assert mod.LEGACY_CORE_ACCELERATOR_IMPORTS - used == set()
