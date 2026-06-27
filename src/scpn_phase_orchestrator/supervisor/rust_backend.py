# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Rust supervisor backend readiness probe

"""Optional Rust supervisor FFI readiness checks.

The live Python supervisor remains the default runtime-control path. This module
answers a narrower packaging and operations question: when the optional
``spo_kernel`` wheel is installed, does it expose the ``spo-supervisor`` PyO3
surface and do the deterministic regime, boundary, and coherence primitives
behave like usable supervisor components?

All probes fail closed. Missing symbols, malformed smoke outputs, or import
failures produce an unavailable status and never mutate production supervisor
state.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

__all__ = [
    "SUPERVISOR_RUST_SYMBOLS",
    "RustSupervisorBackendStatus",
    "audit_rust_supervisor_backend",
]

SUPERVISOR_RUST_SYMBOLS: tuple[str, ...] = (
    "PyActiveInferenceAgent",
    "PyRegimeManager",
    "PyCoherenceMonitor",
    "PyBoundaryObserver",
    "PyActionProjector",
    "PySupervisorPolicy",
    "PyPetriNet",
    "PyRuleEngine",
)

_VALID_REGIMES = frozenset({"nominal", "degraded", "critical", "recovery"})


class _SupervisorFFIModule(Protocol):
    """Structural type for the subset of ``spo_kernel`` used by the smoke probe."""

    PyRegimeManager: Any
    PyBoundaryObserver: Any
    PyCoherenceMonitor: Any


@dataclass(frozen=True)
class RustSupervisorBackendStatus:
    """Readiness outcome for the optional Rust supervisor FFI backend.

    Attributes
    ----------
        available: True only when the module imports, all required symbols are
            present, and deterministic smoke validation succeeds.
        symbols: Required PyO3 symbols checked on the module.
        missing_symbols: Required symbols absent from the inspected module.
        detail: Human-facing readiness explanation for ``spo doctor`` and audit
            records.
        smoke: Deterministic smoke observations when validation succeeds.
    """

    available: bool
    symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    detail: str
    smoke: Mapping[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return ``ok`` when available, otherwise ``warn``.

        Returns
        -------
        str
            ``ok`` for a usable optional backend and ``warn`` for an unavailable
            optional backend.
        """
        return "ok" if self.available else "warn"

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-serialisable backend audit record.

        Returns
        -------
        dict[str, object]
            Backend readiness details suitable for doctor JSON output, release
            evidence, or internal handoff logs.
        """
        return {
            "backend": "rust-supervisor",
            "status": self.status,
            "available": self.available,
            "symbols": list(self.symbols),
            "missing_symbols": list(self.missing_symbols),
            "detail": self.detail,
            "smoke": dict(self.smoke),
        }


def audit_rust_supervisor_backend(
    module: object | None = None,
) -> RustSupervisorBackendStatus:
    """Probe the optional ``spo_kernel`` supervisor FFI surface.

    Parameters
    ----------
    module : object | None
        Optional module-like object used by tests. When ``None``, ``spo_kernel``
        is imported lazily.

    Returns
    -------
    RustSupervisorBackendStatus
        Fail-closed readiness status for the optional Rust supervisor backend.
    """
    if module is None:
        try:
            module = importlib.import_module("spo_kernel")
        except Exception as exc:
            return RustSupervisorBackendStatus(
                available=False,
                symbols=SUPERVISOR_RUST_SYMBOLS,
                missing_symbols=SUPERVISOR_RUST_SYMBOLS,
                detail=f"spo_kernel not importable: {type(exc).__name__}",
            )

    missing = tuple(
        symbol for symbol in SUPERVISOR_RUST_SYMBOLS if not hasattr(module, symbol)
    )
    if missing:
        return RustSupervisorBackendStatus(
            available=False,
            symbols=SUPERVISOR_RUST_SYMBOLS,
            missing_symbols=missing,
            detail=f"missing supervisor FFI symbols: {', '.join(missing)}",
        )

    try:
        smoke = _run_supervisor_smoke(module)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return RustSupervisorBackendStatus(
            available=False,
            symbols=SUPERVISOR_RUST_SYMBOLS,
            missing_symbols=(),
            detail=f"supervisor FFI smoke failed: {exc}",
        )

    return RustSupervisorBackendStatus(
        available=True,
        symbols=SUPERVISOR_RUST_SYMBOLS,
        missing_symbols=(),
        detail="spo-supervisor FFI symbols and deterministic smoke checks passed",
        smoke=smoke,
    )


def _run_supervisor_smoke(module: object) -> dict[str, object]:
    """Execute deterministic, non-actuating smoke checks against FFI classes."""
    ffi = cast(_SupervisorFFIModule, module)
    regime_manager_type = ffi.PyRegimeManager
    boundary_observer_type = ffi.PyBoundaryObserver
    coherence_monitor_type = ffi.PyCoherenceMonitor

    manager = regime_manager_type(cooldown_steps=0)
    nominal_regime = _validate_regime(
        manager.evaluate([0.9, 0.85], []), field_name="nominal_regime"
    )
    critical_regime = _validate_regime(
        manager.evaluate([0.1, 0.2], []), field_name="critical_regime"
    )
    forced_regime = _validate_regime(
        manager.force_transition("critical"), field_name="forced_regime"
    )

    observer = boundary_observer_type()
    boundary_record = _as_mapping(
        observer.observe(
            [("R_floor", "R", 0.3, None, "hard")],
            {"R": 0.1},
        ),
        field_name="boundary_record",
    )
    hard_boundaries = _as_sequence(
        boundary_record.get("hard_violations"), field_name="hard_violations"
    )

    monitor = coherence_monitor_type([0], [1])
    r_good = _as_finite_float(monitor.compute_r_good([0.8, 0.2]), field_name="r_good")
    r_bad = _as_finite_float(monitor.compute_r_bad([0.8, 0.2]), field_name="r_bad")

    return {
        "nominal_regime": nominal_regime,
        "critical_regime": critical_regime,
        "forced_regime": forced_regime,
        "hard_boundary_count": len(hard_boundaries),
        "r_good": r_good,
        "r_bad": r_bad,
    }


def _validate_regime(value: object, *, field_name: str) -> str:
    """Return a validated regime string from an FFI smoke result."""
    if not isinstance(value, str) or value not in _VALID_REGIMES:
        raise ValueError(f"unexpected regime for {field_name}: {value!r}")
    return value


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise a smoke validation error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _as_sequence(value: object, *, field_name: str) -> Sequence[object]:
    """Return ``value`` as a non-string sequence or raise a validation error."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _as_finite_float(value: Any, *, field_name: str) -> float:
    """Return ``value`` as a finite float or raise a smoke validation error."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number
