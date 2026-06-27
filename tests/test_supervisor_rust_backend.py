# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Rust supervisor backend readiness

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from scpn_phase_orchestrator.supervisor.rust_backend import (
    SUPERVISOR_RUST_SYMBOLS,
    audit_rust_supervisor_backend,
)


class _FakeRegimeManager:
    def __init__(
        self,
        *,
        hysteresis: float = 0.05,
        cooldown_steps: int = 0,
        hysteresis_hold_steps: int = 0,
    ) -> None:
        self.hysteresis = hysteresis
        self.cooldown_steps = cooldown_steps
        self.hysteresis_hold_steps = hysteresis_hold_steps
        self.current = "nominal"

    def evaluate(
        self, layer_rs: Sequence[float], hard_violations: Sequence[str]
    ) -> str:
        if hard_violations or min(layer_rs) < 0.3:
            self.current = "critical"
        elif min(layer_rs) < 0.6:
            self.current = "degraded"
        else:
            self.current = "nominal"
        return self.current

    def force_transition(self, regime: str) -> str:
        self.current = regime
        return self.current


class _FakeBoundaryObserver:
    def observe(
        self,
        definitions: Sequence[tuple[str, str, float | None, float | None, str]],
        values: Mapping[str, float],
    ) -> dict[str, object]:
        hard_violations: list[dict[str, object]] = []
        for name, variable, lower, upper, severity in definitions:
            value = values[variable]
            breached = (lower is not None and value < lower) or (
                upper is not None and value > upper
            )
            if breached and severity == "hard":
                hard_violations.append({"name": name, "variable": variable})
        return {
            "violations": hard_violations,
            "soft_violations": [],
            "hard_violations": hard_violations,
        }


class _FakeCoherenceMonitor:
    def __init__(self, good_layers: Sequence[int], bad_layers: Sequence[int]) -> None:
        self.good_layers = tuple(good_layers)
        self.bad_layers = tuple(bad_layers)

    def compute_r_good(self, layer_rs: Sequence[float]) -> float:
        return layer_rs[self.good_layers[0]]

    def compute_r_bad(self, layer_rs: Sequence[float]) -> float:
        return layer_rs[self.bad_layers[0]]


class _NonMappingBoundaryObserver(_FakeBoundaryObserver):
    def observe(
        self,
        definitions: Sequence[tuple[str, str, float | None, float | None, str]],
        values: Mapping[str, float],
    ) -> list[object]:
        del definitions, values
        return []


class _StringHardBoundaryObserver(_FakeBoundaryObserver):
    def observe(
        self,
        definitions: Sequence[tuple[str, str, float | None, float | None, str]],
        values: Mapping[str, float],
    ) -> dict[str, object]:
        del definitions, values
        return {"hard_violations": "R_floor"}


class _NonFiniteCoherenceMonitor(_FakeCoherenceMonitor):
    def compute_r_good(self, layer_rs: Sequence[float]) -> float:
        del layer_rs
        return math.inf


class _MalformedRegimeManager(_FakeRegimeManager):
    def evaluate(
        self, layer_rs: Sequence[float], hard_violations: Sequence[str]
    ) -> str:
        del layer_rs, hard_violations
        return "unknown"


def _fake_module(**overrides: object) -> SimpleNamespace:
    symbols: dict[str, object] = {
        symbol: type(symbol, (), {}) for symbol in SUPERVISOR_RUST_SYMBOLS
    }
    symbols.update(
        {
            "PyRegimeManager": _FakeRegimeManager,
            "PyBoundaryObserver": _FakeBoundaryObserver,
            "PyCoherenceMonitor": _FakeCoherenceMonitor,
        }
    )
    symbols.update(overrides)
    return SimpleNamespace(**symbols)


def test_audit_rust_supervisor_backend_accepts_valid_ffi_contract() -> None:
    status = audit_rust_supervisor_backend(module=_fake_module())

    assert status.available is True
    assert status.missing_symbols == ()
    assert status.smoke["nominal_regime"] == "nominal"
    assert status.smoke["critical_regime"] == "critical"
    assert status.smoke["forced_regime"] == "critical"
    assert status.smoke["hard_boundary_count"] == 1
    assert status.smoke["r_good"] == 0.8
    assert status.smoke["r_bad"] == 0.2
    assert status.to_audit_record()["status"] == "ok"


def test_audit_rust_supervisor_backend_reports_missing_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "spo_kernel":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    status = audit_rust_supervisor_backend()

    assert status.available is False
    assert status.missing_symbols == SUPERVISOR_RUST_SYMBOLS
    assert status.smoke == {}
    assert "not importable" in status.detail


def test_audit_rust_supervisor_backend_reports_missing_symbols() -> None:
    module = _fake_module()
    delattr(module, "PyCoherenceMonitor")

    status = audit_rust_supervisor_backend(module=module)

    assert status.available is False
    assert status.missing_symbols == ("PyCoherenceMonitor",)
    assert status.smoke == {}
    assert "missing supervisor FFI symbols" in status.detail


def test_audit_rust_supervisor_backend_fails_closed_on_malformed_smoke() -> None:
    status = audit_rust_supervisor_backend(
        module=_fake_module(PyRegimeManager=_MalformedRegimeManager)
    )

    assert status.available is False
    assert status.missing_symbols == ()
    assert status.smoke == {}
    assert "unexpected regime" in status.detail


def test_audit_rust_supervisor_backend_fails_closed_on_non_mapping_boundary() -> None:
    status = audit_rust_supervisor_backend(
        module=_fake_module(PyBoundaryObserver=_NonMappingBoundaryObserver)
    )

    assert status.available is False
    assert status.smoke == {}
    assert "boundary_record must be a mapping" in status.detail


def test_audit_rust_supervisor_backend_fails_closed_on_bad_boundary_sequence() -> None:
    status = audit_rust_supervisor_backend(
        module=_fake_module(PyBoundaryObserver=_StringHardBoundaryObserver)
    )

    assert status.available is False
    assert status.smoke == {}
    assert "hard_violations must be a sequence" in status.detail


def test_audit_rust_supervisor_backend_fails_closed_on_non_finite_coherence() -> None:
    status = audit_rust_supervisor_backend(
        module=_fake_module(PyCoherenceMonitor=_NonFiniteCoherenceMonitor)
    )

    assert status.available is False
    assert status.smoke == {}
    assert "r_good must be finite" in status.detail
