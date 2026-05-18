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


def test_package_init_relative_import_stays_inside_package(tmp_path: Path) -> None:
    init_file = _write_module(tmp_path, "nn/__init__.py", "from .runtime import x\n")

    imports = mod.imported_modules(init_file)

    assert imports == [(1, "scpn_phase_orchestrator.nn.runtime")]


def test_runtime_namespace_is_runtime_boundary() -> None:
    assert mod.classify_module("scpn_phase_orchestrator.runtime.cli") == "runtime"
    assert (
        mod.classify_module("scpn_phase_orchestrator.runtime.network_security")
        == "runtime"
    )
    assert mod.classify_module("scpn_phase_orchestrator.runtime.replay") == "runtime"
    assert mod.classify_module("scpn_phase_orchestrator.runtime.server") == "runtime"
    assert mod.classify_module("scpn_phase_orchestrator.runtime.server_grpc") == (
        "runtime"
    )


def test_core_package_cannot_import_integration_surface(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "monitor/boundaries.py",
        "from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter\n",
    )

    violations = mod.find_violations([core])

    assert len(violations) == 1
    assert violations[0].target_boundary == "integrations"


def test_integration_package_can_import_core_surface(tmp_path: Path) -> None:
    adapter = _write_module(
        tmp_path,
        "adapters/prometheus.py",
        "from scpn_phase_orchestrator.upde.metrics import UPDEState\n",
    )

    assert mod.find_violations([adapter]) == []


def test_integration_package_cannot_import_runtime_surface(tmp_path: Path) -> None:
    adapter = _write_module(
        tmp_path,
        "adapters/prometheus.py",
        "from scpn_phase_orchestrator.server import SimulationState\n",
    )

    violations = mod.find_violations([adapter])

    assert len(violations) == 1
    assert violations[0].source_boundary == "integrations"
    assert violations[0].target_boundary == "runtime"


def test_integration_package_cannot_import_experimental_surface(
    tmp_path: Path,
) -> None:
    adapter = _write_module(
        tmp_path,
        "drivers/psi_physical.py",
        "from scpn_phase_orchestrator.nn.ude import UDEKuramotoLayer\n",
    )

    violations = mod.find_violations([adapter])

    assert len(violations) == 1
    assert violations[0].source_boundary == "integrations"
    assert violations[0].target_boundary == "experimental"


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


def test_core_accelerator_port_allowlist_usage_is_measured(tmp_path: Path) -> None:
    core = _write_module(
        tmp_path,
        "upde/engine.py",
        "from scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo "
        "import upde_run_mojo\n",
    )

    used = mod.find_core_accelerator_port_imports([core])

    assert used == {
        "scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo"
    }
    assert mod.find_violations([core]) == []


def test_unclassified_first_party_source_module_is_rejected(tmp_path: Path) -> None:
    unknown = _write_module(tmp_path, "new_surface/tool.py", "VALUE = 1\n")

    assert mod.find_unclassified_modules([unknown]) == {
        "scpn_phase_orchestrator.new_surface.tool"
    }


def test_unclassified_first_party_import_is_rejected(tmp_path: Path) -> None:
    runtime = _write_module(
        tmp_path,
        "server.py",
        "from scpn_phase_orchestrator.new_surface.tool import VALUE\n",
    )

    assert mod.find_unclassified_modules([runtime]) == {
        "scpn_phase_orchestrator.new_surface.tool"
    }


def test_current_source_tree_respects_core_boundary_contract() -> None:
    mod.SRC_ROOT = ORIGINAL_SRC_ROOT
    violations = mod.find_violations(mod.iter_python_files())

    assert violations == []


def test_current_source_tree_has_no_unclassified_modules() -> None:
    mod.SRC_ROOT = ORIGINAL_SRC_ROOT
    unclassified = mod.find_unclassified_modules(mod.iter_python_files())

    assert unclassified == set()


def test_current_core_accelerator_port_allowlist_has_no_stale_entries() -> None:
    mod.SRC_ROOT = ORIGINAL_SRC_ROOT
    used = mod.find_core_accelerator_port_imports(mod.iter_python_files())

    assert mod.CORE_ACCELERATOR_PORT_IMPORTS - used == set()
