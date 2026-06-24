# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — environment readiness diagnostics

"""Runtime environment readiness checks for ``spo doctor``.

This module answers one question deterministically: *can this interpreter run
SCPN Phase Orchestrator, and which optional accelerators and feature extras are
usable right now?* It probes three layers without importing heavy optional
packages or touching the network:

* the interpreter version against the packaged ``requires-python`` window;
* the mandatory runtime dependencies (NumPy, SciPy, PyYAML, Click, protobuf,
  urllib3) that the core engine imports unconditionally;
* the optional native compute backends (Rust ``spo_kernel``, Julia
  ``juliacall``, the Go toolchain/shared libraries, the Mojo toolchain) and the
  optional feature extras (``nn``, ``studio``, ``queuewaves``, ``plot``,
  ``otel``, ``notebook``).

Detection uses :func:`importlib.util.find_spec` for Python modules (which
locates a distribution without executing its top-level import) and
:func:`shutil.which` / filesystem probes for external toolchains, so running the
diagnostics is cheap and free of side effects. A missing *required* dependency
or an out-of-range interpreter makes the overall status ``fail`` (non-zero exit
for the CLI); missing *optional* components are reported as ``warn`` and never
fail the run.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import platform
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "REQUIRED_PYTHON",
    "DependencyCheck",
    "DoctorReport",
    "run_environment_diagnostics",
]

# Mirrors ``requires-python = ">=3.10,<3.14"`` in ``pyproject.toml``. Kept as a
# pair of (major, minor) bounds so the check stays exact without parsing the
# packaging metadata at runtime.
REQUIRED_PYTHON: tuple[tuple[int, int], tuple[int, int]] = ((3, 10), (3, 14))

_STATUS_OK = "ok"
_STATUS_MISSING = "missing"
_STATUS_READY = "pass"
_STATUS_FAIL = "fail"
_STATUS_WARN = "warn"


@dataclass(frozen=True)
class DependencyCheck:
    """Outcome of probing a single dependency, backend, or toolchain.

    Attributes
    ----------
        name: Human-facing component name (for example ``numpy`` or ``rust``).
        category: Grouping used for rendering — ``interpreter``, ``core``,
            ``backend``, or one of the optional feature-extra names.
        required: Whether the component is mandatory for the core engine.
        available: Whether the component was detected as usable.
        detail: Short human-readable explanation (version string, path,
            install hint, or reason it is unavailable).
        version: Resolved distribution version when known, otherwise ``None``.
    """

    name: str
    category: str
    required: bool
    available: bool
    detail: str
    version: str | None = None

    @property
    def status(self) -> str:
        """Return ``ok``/``missing``/``warn`` for the component detection state.

        Returns
        -------
        str
            Return ``ok``/``missing``/``warn`` for the component detection state.
        """
        if self.available:
            return _STATUS_OK
        return _STATUS_MISSING if self.required else _STATUS_WARN

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping with deterministic key order.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable mapping with deterministic key order.
        """
        return {
            "name": self.name,
            "category": self.category,
            "required": self.required,
            "available": self.available,
            "status": self.status,
            "version": self.version,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate readiness report produced by :func:`run_environment_diagnostics`.

    Attributes
    ----------
        checks: Every :class:`DependencyCheck` in deterministic display order.
        python_version: The running interpreter version (``X.Y.Z``).
        platform: Short OS/architecture descriptor for the audit record.
    """

    checks: tuple[DependencyCheck, ...]
    python_version: str
    platform: str = field(default="")

    @property
    def missing_required(self) -> tuple[DependencyCheck, ...]:
        """Required components that were not detected.

        Returns
        -------
        tuple[DependencyCheck, ...]
            Required components that were not detected.
        """
        return tuple(c for c in self.checks if c.required and not c.available)

    @property
    def missing_optional(self) -> tuple[DependencyCheck, ...]:
        """Optional components that were not detected.

        Returns
        -------
        tuple[DependencyCheck, ...]
            Optional components that were not detected.
        """
        return tuple(c for c in self.checks if not c.required and not c.available)

    @property
    def ok(self) -> bool:
        """True when every required component is present (overall ``pass``).

        Returns
        -------
        bool
            True when every required component is present (overall ``pass``).
        """
        return not self.missing_required

    @property
    def status(self) -> str:
        """``pass`` when ready, otherwise ``fail``.

        Returns
        -------
        str
            ``pass`` when ready, otherwise ``fail``.
        """
        return _STATUS_READY if self.ok else _STATUS_FAIL

    @property
    def exit_code(self) -> int:
        """Process exit code: ``0`` when ready, ``1`` when a requirement is missing.

        Returns
        -------
        int
            Process exit code: ``0`` when ready, ``1`` when a requirement is missing.
        """
        return 0 if self.ok else 1

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-serialisable readiness record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-serialisable readiness record.
        """
        return {
            "report": "environment-diagnostics",
            "version": "1.0.0",
            "status": self.status,
            "python_version": self.python_version,
            "platform": self.platform,
            "required_present": len(
                [c for c in self.checks if c.required and c.available]
            ),
            "required_total": len([c for c in self.checks if c.required]),
            "optional_present": len(
                [c for c in self.checks if not c.required and c.available]
            ),
            "optional_total": len([c for c in self.checks if not c.required]),
            "missing_required": [c.name for c in self.missing_required],
            "missing_optional": [c.name for c in self.missing_optional],
            "checks": [c.to_record() for c in self.checks],
        }


def _distribution_version(dist_name: str) -> str | None:
    """Return the installed version of ``dist_name`` or ``None`` if absent."""
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _module_present(import_name: str) -> bool:
    """Return True when ``import_name`` can be located without importing it.

    ``find_spec`` raises :class:`ModuleNotFoundError` when a *parent* package of
    a dotted name is missing; that is treated as "not present" rather than an
    error so the probe never aborts the whole report.
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _check_python() -> DependencyCheck:
    """Return the Python environment health-check result."""
    info = sys.version_info
    current = (info.major, info.minor)
    low, high = REQUIRED_PYTHON
    in_range = low <= current < high
    version = f"{info.major}.{info.minor}.{info.micro}"
    window = f">={low[0]}.{low[1]},<{high[0]}.{high[1]}"
    detail = (
        f"Python {version} satisfies {window}"
        if in_range
        else f"Python {version} is outside the supported window {window}"
    )
    return DependencyCheck(
        name="python",
        category="interpreter",
        required=True,
        available=in_range,
        detail=detail,
        version=version,
    )


def _check_module(
    *,
    name: str,
    import_name: str,
    dist_name: str,
    category: str,
    required: bool,
    install_hint: str,
) -> DependencyCheck:
    """Return the health-check result for a Python module."""
    present = _module_present(import_name)
    version = _distribution_version(dist_name) if present else None
    if present:
        detail = f"{dist_name} {version}" if version else f"{dist_name} importable"
    else:
        detail = f"not installed — {install_hint}"
    return DependencyCheck(
        name=name,
        category=category,
        required=required,
        available=present,
        detail=detail,
        version=version,
    )


def _check_modules(
    specs: Iterable[tuple[str, str, str]],
    *,
    category: str,
    required: bool,
    install_hint: str,
) -> list[DependencyCheck]:
    """Return the health-check results for the required modules."""
    return [
        _check_module(
            name=name,
            import_name=import_name,
            dist_name=dist_name,
            category=category,
            required=required,
            install_hint=install_hint,
        )
        for name, import_name, dist_name in specs
    ]


# (display name, import name, distribution name) for the mandatory runtime deps.
_CORE_DEPS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "numpy"),
    ("scipy", "scipy", "scipy"),
    ("pyyaml", "yaml", "PyYAML"),
    ("click", "click", "click"),
    ("protobuf", "google.protobuf", "protobuf"),
    ("urllib3", "urllib3", "urllib3"),
)

# Optional feature extras keyed by the pip extra that installs them.
_OPTIONAL_EXTRAS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "nn": (
        ("jax", "jax", "jax"),
        ("equinox", "equinox", "equinox"),
        ("optax", "optax", "optax"),
    ),
    "studio": (("streamlit", "streamlit", "streamlit"),),
    "queuewaves": (
        ("fastapi", "fastapi", "fastapi"),
        ("uvicorn", "uvicorn", "uvicorn"),
    ),
    "plot": (("matplotlib", "matplotlib", "matplotlib"),),
    "otel": (("opentelemetry", "opentelemetry", "opentelemetry-api"),),
    "notebook": (("jupyter", "jupyter", "jupyter"),),
}


def _go_shared_libraries(repo_root: Path | None) -> list[str]:
    """Return the discovered Go shared-library paths."""
    if repo_root is None:
        return []
    go_dir = repo_root / "go"
    if not go_dir.is_dir():
        return []
    return sorted(p.name for p in go_dir.glob("*.so"))


def _check_rust() -> DependencyCheck:
    """Return the Rust backend health-check result."""
    present = _module_present("spo_kernel")
    version = _distribution_version("spo-kernel") if present else None
    if present:
        detail = (
            f"spo_kernel {version} importable (PyO3 FFI ready)"
            if version
            else "spo_kernel importable (PyO3 FFI ready)"
        )
    else:
        detail = "spo_kernel not importable — install the 'rust' extra (spo-kernel)"
    return DependencyCheck(
        name="rust",
        category="backend",
        required=False,
        available=present,
        detail=detail,
        version=version,
    )


def _check_julia() -> DependencyCheck:
    """Return the Julia backend health-check result."""
    present = _module_present("juliacall")
    version = _distribution_version("juliacall") if present else None
    detail = (
        f"juliacall {version} importable"
        if present
        else "juliacall not importable — install the 'julia' extra (juliacall)"
    )
    return DependencyCheck(
        name="julia",
        category="backend",
        required=False,
        available=present,
        detail=detail,
        version=version,
    )


def _check_go(repo_root: Path | None) -> DependencyCheck:
    """Return the Go backend health-check result."""
    toolchain = shutil.which("go")
    libraries = _go_shared_libraries(repo_root)
    available = toolchain is not None or bool(libraries)
    if libraries:
        detail = f"prebuilt shared libraries: {', '.join(libraries)}"
    elif toolchain is not None:
        detail = f"go toolchain at {toolchain} (build shared libs in ./go)"
    else:
        detail = "no 'go' toolchain on PATH and no prebuilt libraries in ./go"
    return DependencyCheck(
        name="go",
        category="backend",
        required=False,
        available=available,
        detail=detail,
        version=None,
    )


def _check_mojo() -> DependencyCheck:
    """Return the Mojo backend health-check result."""
    toolchain = shutil.which("mojo")
    available = toolchain is not None
    detail = (
        f"mojo toolchain at {toolchain}" if available else "no 'mojo' toolchain on PATH"
    )
    return DependencyCheck(
        name="mojo",
        category="backend",
        required=False,
        available=available,
        detail=detail,
        version=None,
    )


def _find_repo_root(start: Path | None = None) -> Path | None:
    """Locate a checkout root that carries the ``go`` backend sources.

    Returns the first ancestor of ``start`` (this module's file by default)
    containing both a ``go`` directory and ``pyproject.toml`` (a source
    checkout). Installed wheels ship no ``go`` directory, so this returns
    ``None`` there and the Go probe falls back to the toolchain-on-PATH signal.

    Parameters
    ----------
    start : Path | None
        Optional starting file/directory; tests pass a sandbox path to exercise the
        no-checkout fallback.

    Returns
    -------
    Path | None
        The result.
    """
    here = (start if start is not None else Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "go").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return None


def run_environment_diagnostics(*, repo_root: Path | None = None) -> DoctorReport:
    """Probe the interpreter, required dependencies, and optional components.

    Parameters
    ----------
    repo_root : Path | None
        Optional explicit checkout root for the Go shared-library probe. When ``None``
        the root is auto-detected from this module's location; pass a path in tests to
        exercise both branches.

    Returns
    -------
    DoctorReport
        A :class:`DoctorReport` whose :attr:`DoctorReport.status` is ``pass`` only when
        the interpreter is in range and every required dependency is importable.
    """
    resolved_root = repo_root if repo_root is not None else _find_repo_root()

    checks: list[DependencyCheck] = [_check_python()]
    checks.extend(
        _check_modules(
            _CORE_DEPS,
            category="core",
            required=True,
            install_hint="install scpn-phase-orchestrator core dependencies",
        )
    )
    checks.append(_check_rust())
    checks.append(_check_julia())
    checks.append(_check_go(resolved_root))
    checks.append(_check_mojo())
    for extra, specs in _OPTIONAL_EXTRAS.items():
        checks.extend(
            _check_modules(
                specs,
                category=extra,
                required=False,
                install_hint=f"install the '{extra}' extra",
            )
        )

    return DoctorReport(
        checks=tuple(checks),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}".strip(),
    )


def render_report(report: DoctorReport) -> Sequence[str]:
    """Render a :class:`DoctorReport` as aligned human-readable lines."""
    glyphs = {_STATUS_OK: "[ ok ]", _STATUS_MISSING: "[MISS]", _STATUS_WARN: "[warn]"}
    lines: list[str] = [
        f"SCPN Phase Orchestrator environment diagnostics — {report.status.upper()}",
        f"  python   {report.python_version}  ({report.platform})",
        "",
    ]
    width = max((len(c.name) for c in report.checks), default=0)
    for check in report.checks:
        glyph = glyphs.get(check.status, "[????]")
        lines.append(f"  {glyph} {check.name.ljust(width)}  {check.detail}")
    lines.append("")
    if report.missing_required:
        names = ", ".join(c.name for c in report.missing_required)
        lines.append(f"FAIL: missing required components: {names}")
    else:
        optional_present = len(
            [c for c in report.checks if not c.required and c.available]
        )
        optional_total = len([c for c in report.checks if not c.required])
        lines.append(
            "PASS: all required dependencies are present "
            f"({optional_present}/{optional_total} optional components available)."
        )
    return lines
