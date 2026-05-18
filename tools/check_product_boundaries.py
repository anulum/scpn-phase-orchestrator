# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Product boundary import guard

"""Validate product-boundary imports for the Python source tree.

The boundary contract is intentionally narrower than a directory move: it keeps
legacy import paths stable while preventing the Core Engine layer from acquiring
new dependencies on runtime, integration, or experimental surfaces.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "scpn_phase_orchestrator"
PACKAGE = "scpn_phase_orchestrator"

CORE_PACKAGES = frozenset(
    {
        "actuation",
        "binding",
        "coupling",
        "exceptions",
        "imprint",
        "monitor",
        "oscillators",
        "ssgf",
        "supervisor",
        "upde",
        "_compat",
    }
)
RUNTIME_PACKAGES = frozenset(
    {
        "apps",
        "artifacts",
        "audit",
        "cli",
        "distributed",
        "grpc_gen",
        "meta",
        "network_security",
        "plugins",
        "reporting",
        "scaffold",
        "server",
        "server_grpc",
        "studio",
    }
)
INTEGRATION_PACKAGES = frozenset({"adapters", "drivers"})
EXPERIMENTAL_PACKAGES = frozenset({"nn", "visualization"})
EXPERIMENTAL_SUFFIXES = ("_go", "_julia", "_mojo", "_webgpu")
FORBIDDEN_CORE_TARGETS = frozenset({"runtime", "integrations", "experimental"})
LEGACY_CORE_ACCELERATOR_IMPORTS = frozenset(
    {
        "scpn_phase_orchestrator.coupling._attnres_go",
        "scpn_phase_orchestrator.coupling._attnres_julia",
        "scpn_phase_orchestrator.coupling._attnres_mojo",
        "scpn_phase_orchestrator.coupling._hodge_go",
        "scpn_phase_orchestrator.coupling._hodge_julia",
        "scpn_phase_orchestrator.coupling._hodge_mojo",
        "scpn_phase_orchestrator.coupling._spectral_go",
        "scpn_phase_orchestrator.coupling._spectral_julia",
        "scpn_phase_orchestrator.coupling._spectral_mojo",
        "scpn_phase_orchestrator.monitor._chimera_go",
        "scpn_phase_orchestrator.monitor._chimera_julia",
        "scpn_phase_orchestrator.monitor._chimera_mojo",
        "scpn_phase_orchestrator.monitor._dimension_go",
        "scpn_phase_orchestrator.monitor._dimension_julia",
        "scpn_phase_orchestrator.monitor._dimension_mojo",
        "scpn_phase_orchestrator.monitor._embedding_go",
        "scpn_phase_orchestrator.monitor._embedding_julia",
        "scpn_phase_orchestrator.monitor._embedding_mojo",
        "scpn_phase_orchestrator.monitor._entropy_prod_go",
        "scpn_phase_orchestrator.monitor._entropy_prod_julia",
        "scpn_phase_orchestrator.monitor._entropy_prod_mojo",
        "scpn_phase_orchestrator.monitor._itpc_go",
        "scpn_phase_orchestrator.monitor._itpc_julia",
        "scpn_phase_orchestrator.monitor._itpc_mojo",
        "scpn_phase_orchestrator.monitor._lyapunov_go",
        "scpn_phase_orchestrator.monitor._lyapunov_julia",
        "scpn_phase_orchestrator.monitor._lyapunov_mojo",
        "scpn_phase_orchestrator.monitor._npe_go",
        "scpn_phase_orchestrator.monitor._npe_julia",
        "scpn_phase_orchestrator.monitor._npe_mojo",
        "scpn_phase_orchestrator.monitor._poincare_go",
        "scpn_phase_orchestrator.monitor._poincare_julia",
        "scpn_phase_orchestrator.monitor._poincare_mojo",
        "scpn_phase_orchestrator.monitor._psychedelic_go",
        "scpn_phase_orchestrator.monitor._psychedelic_julia",
        "scpn_phase_orchestrator.monitor._psychedelic_mojo",
        "scpn_phase_orchestrator.monitor._recurrence_go",
        "scpn_phase_orchestrator.monitor._recurrence_julia",
        "scpn_phase_orchestrator.monitor._recurrence_mojo",
        "scpn_phase_orchestrator.monitor._te_go",
        "scpn_phase_orchestrator.monitor._te_julia",
        "scpn_phase_orchestrator.monitor._te_mojo",
        "scpn_phase_orchestrator.monitor._winding_go",
        "scpn_phase_orchestrator.monitor._winding_julia",
        "scpn_phase_orchestrator.monitor._winding_mojo",
        "scpn_phase_orchestrator.upde._basin_stability_go",
        "scpn_phase_orchestrator.upde._basin_stability_julia",
        "scpn_phase_orchestrator.upde._basin_stability_mojo",
        "scpn_phase_orchestrator.upde._engine_go",
        "scpn_phase_orchestrator.upde._engine_julia",
        "scpn_phase_orchestrator.upde._engine_mojo",
        "scpn_phase_orchestrator.upde._engine_webgpu",
        "scpn_phase_orchestrator.upde._envelope_go",
        "scpn_phase_orchestrator.upde._envelope_julia",
        "scpn_phase_orchestrator.upde._envelope_mojo",
        "scpn_phase_orchestrator.upde._geometric_go",
        "scpn_phase_orchestrator.upde._geometric_julia",
        "scpn_phase_orchestrator.upde._geometric_mojo",
        "scpn_phase_orchestrator.upde._hypergraph_go",
        "scpn_phase_orchestrator.upde._hypergraph_julia",
        "scpn_phase_orchestrator.upde._hypergraph_mojo",
        "scpn_phase_orchestrator.upde._inertial_go",
        "scpn_phase_orchestrator.upde._inertial_julia",
        "scpn_phase_orchestrator.upde._inertial_mojo",
        "scpn_phase_orchestrator.upde._market_go",
        "scpn_phase_orchestrator.upde._market_julia",
        "scpn_phase_orchestrator.upde._market_mojo",
        "scpn_phase_orchestrator.upde._order_params_go",
        "scpn_phase_orchestrator.upde._order_params_julia",
        "scpn_phase_orchestrator.upde._order_params_mojo",
        "scpn_phase_orchestrator.upde._pac_go",
        "scpn_phase_orchestrator.upde._pac_julia",
        "scpn_phase_orchestrator.upde._pac_mojo",
        "scpn_phase_orchestrator.upde._reduction_go",
        "scpn_phase_orchestrator.upde._reduction_julia",
        "scpn_phase_orchestrator.upde._reduction_mojo",
        "scpn_phase_orchestrator.upde._simplicial_go",
        "scpn_phase_orchestrator.upde._simplicial_julia",
        "scpn_phase_orchestrator.upde._simplicial_mojo",
        "scpn_phase_orchestrator.upde._splitting_go",
        "scpn_phase_orchestrator.upde._splitting_julia",
        "scpn_phase_orchestrator.upde._splitting_mojo",
        "scpn_phase_orchestrator.upde._swarmalator_go",
        "scpn_phase_orchestrator.upde._swarmalator_julia",
        "scpn_phase_orchestrator.upde._swarmalator_mojo",
    }
)


@dataclass(frozen=True)
class ImportViolation:
    """A single forbidden product-boundary dependency."""

    path: Path
    line: int
    source_boundary: str
    target_boundary: str
    imported_module: str


def iter_python_files(root: Path = SRC_ROOT) -> list[Path]:
    """Return tracked source-like Python files below ``root`` in stable order."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )


def module_name(path: Path) -> str:
    """Convert a source file path into its importable module name."""
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts)) if parts else PACKAGE


def _top_level(module: str) -> str | None:
    if module == PACKAGE:
        return None
    prefix = f"{PACKAGE}."
    if not module.startswith(prefix):
        return None
    rest = module[len(prefix) :]
    return rest.split(".", 1)[0]


def classify_module(module: str) -> str | None:
    """Classify ``module`` into one of the product boundaries.

    Returns ``None`` for external imports and the package root. Language shim
    modules keep their experimental classification even when physically stored
    under a core package during the migration window.
    """
    if not module.startswith(PACKAGE):
        return None
    leaf = module.rsplit(".", 1)[-1]
    if leaf.endswith(EXPERIMENTAL_SUFFIXES):
        return "experimental"

    top = _top_level(module)
    if top is None:
        return "shared"
    if top in CORE_PACKAGES:
        return "core"
    if top in RUNTIME_PACKAGES:
        return "runtime"
    if top in INTEGRATION_PACKAGES:
        return "integrations"
    if top in EXPERIMENTAL_PACKAGES:
        return "experimental"
    return "unclassified"


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """Extract absolute import module names from ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    current_module = module_name(path)
    current_package = current_module.rsplit(".", 1)[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            if node.level:
                base_parts = current_package.split(".")
                trim = node.level - 1
                if trim >= len(base_parts):
                    continue
                prefix = ".".join(base_parts[: len(base_parts) - trim])
                module = f"{prefix}.{node.module}" if node.module else prefix
            else:
                module = node.module or ""
            imports.append((node.lineno, module))
    return imports


def find_violations(paths: Iterable[Path]) -> list[ImportViolation]:
    """Find forbidden imports according to the product-boundary contract."""
    violations: list[ImportViolation] = []
    for path in paths:
        source_module = module_name(path)
        source_boundary = classify_module(source_module)
        if source_boundary != "core":
            continue
        for line, imported_module in imported_modules(path):
            target_boundary = classify_module(imported_module)
            if target_boundary not in FORBIDDEN_CORE_TARGETS:
                continue
            if imported_module in LEGACY_CORE_ACCELERATOR_IMPORTS:
                continue
            violations.append(
                ImportViolation(
                    path=path,
                    line=line,
                    source_boundary=source_boundary,
                    target_boundary=target_boundary,
                    imported_module=imported_module,
                )
            )
    return violations


def find_legacy_accelerator_imports(paths: Iterable[Path]) -> set[str]:
    """Return legacy accelerator imports still used by Core Engine modules."""
    used: set[str] = set()
    for path in paths:
        source_module = module_name(path)
        source_boundary = classify_module(source_module)
        if source_boundary != "core":
            continue
        for _, imported_module in imported_modules(path):
            if imported_module in LEGACY_CORE_ACCELERATOR_IMPORTS:
                used.add(imported_module)
    return used


def _format_path(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(arg) for arg in args] if args else iter_python_files()
    violations = find_violations(paths)
    stale_legacy_imports = (
        LEGACY_CORE_ACCELERATOR_IMPORTS - find_legacy_accelerator_imports(paths)
        if not args
        else frozenset()
    )
    if violations:
        print("ERROR: product-boundary import violations detected")
        for violation in violations:
            print(
                f"  {_format_path(violation.path)}:{violation.line}: "
                f"{violation.source_boundary} must not import "
                f"{violation.target_boundary}: {violation.imported_module}"
            )
        return 1

    if stale_legacy_imports:
        print("ERROR: stale legacy accelerator allowlist entries detected")
        for imported_module in sorted(stale_legacy_imports):
            print(f"  {imported_module}")
        return 1

    print(f"OK: product boundaries hold for {len(paths)} Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
