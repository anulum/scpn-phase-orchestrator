# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD backend parity tests

"""Per-backend parity gate for the Koopman EDMD-with-control solve.

Every available accelerator backend must reproduce the NumPy reference solve to
a tight tolerance on the same lifted snapshot data; the dispatched solve is also
checked to agree with the reference so the active backend never silently
diverges from the published least-squares result.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import koopman_edmd as ke

_TOLERANCE = 1.0e-9

_NON_PYTHON_BACKENDS = [name for name in ke.AVAILABLE_BACKENDS if name != "python"]


def _snapshots() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260622)
    samples, n_lift, n_input, n_state = 240, 6, 2, 3
    x_lift = rng.standard_normal((samples, n_lift))
    inputs = rng.standard_normal((samples, n_input))
    y_lift = rng.standard_normal((samples, n_lift))
    states = rng.standard_normal((samples, n_state))
    return x_lift, inputs, y_lift, states


@pytest.mark.skipif(not _NON_PYTHON_BACKENDS, reason="no accelerator backend is built")
@pytest.mark.parametrize("backend", _NON_PYTHON_BACKENDS)
def test_backend_matches_the_reference_solve(backend: str) -> None:
    x_lift, inputs, y_lift, states = _snapshots()
    regularisation = 1.0e-8
    reference = ke._edmd_solve_reference(x_lift, inputs, y_lift, states, regularisation)
    solver = ke._load_backend(backend)["edmd_solve"]
    produced = solver(x_lift, inputs, y_lift, states, regularisation)  # type: ignore[operator]
    for got, expected in zip(produced, reference, strict=True):
        np.testing.assert_allclose(got, expected, atol=_TOLERANCE, rtol=_TOLERANCE)


def test_active_dispatch_agrees_with_the_reference() -> None:
    x_lift, inputs, y_lift, states = _snapshots()
    regularisation = 1.0e-8
    reference = ke._edmd_solve_reference(x_lift, inputs, y_lift, states, regularisation)
    dispatched = ke._edmd_solve(x_lift, inputs, y_lift, states, regularisation)
    for got, expected in zip(dispatched, reference, strict=True):
        np.testing.assert_allclose(got, expected, atol=_TOLERANCE, rtol=_TOLERANCE)


def test_edmd_solve_falls_back_to_the_python_reference(monkeypatch) -> None:
    monkeypatch.setattr(ke, "_dispatch", lambda _name: None)
    x_lift, inputs, y_lift, states = _snapshots()
    fallback = ke._edmd_solve(x_lift, inputs, y_lift, states, 1.0e-8)
    reference = ke._edmd_solve_reference(x_lift, inputs, y_lift, states, 1.0e-8)
    for got, expected in zip(fallback, reference, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_dispatch_skips_a_backend_missing_the_kernel(monkeypatch) -> None:
    monkeypatch.setattr(ke, "ACTIVE_BACKEND", "phantom")
    monkeypatch.setattr(ke, "AVAILABLE_BACKENDS", ["phantom", "python"])
    monkeypatch.setattr(ke, "_load_backend", lambda _name: {"edmd_solve": None})
    # 'phantom' lacks the kernel and 'python' signals the reference path.
    assert ke._dispatch("edmd_solve") is None


def test_dispatch_returns_none_when_every_backend_is_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(ke, "ACTIVE_BACKEND", "phantom")
    monkeypatch.setattr(ke, "AVAILABLE_BACKENDS", ["phantom"])
    monkeypatch.setattr(ke, "_load_backend", lambda _name: {"edmd_solve": None})
    assert ke._dispatch("edmd_solve") is None


def test_dispatch_skips_a_backend_that_fails_to_load(monkeypatch) -> None:
    def _raise(_name: str) -> dict[str, object]:
        raise ImportError("backend absent")

    monkeypatch.setattr(ke, "ACTIVE_BACKEND", "phantom")
    monkeypatch.setattr(ke, "AVAILABLE_BACKENDS", ["phantom", "python"])
    monkeypatch.setattr(ke, "_load_backend", _raise)
    assert ke._dispatch("edmd_solve") is None


def test_resolve_backends_skips_a_loader_that_fails(monkeypatch) -> None:
    def _boom() -> dict[str, object]:
        raise ImportError("toolchain absent")

    monkeypatch.setattr(ke, "_BACKEND_NAMES", ("phantom", "python"))
    monkeypatch.setitem(ke._LOADERS, "phantom", _boom)
    active, available = ke._resolve_backends()
    assert active == "python"
    assert available == ["python"]


def test_benchmark_polyglot_parity_gate_passes() -> None:
    from benchmarks.koopman_edmd_benchmark import (
        benchmark_koopman_edmd_polyglot_parity_gate,
    )

    result = benchmark_koopman_edmd_polyglot_parity_gate(calls=1)
    assert result["parity_ok"] is True
    present = [r for r in result["backend_records"] if r.get("available")]
    assert {r["backend"] for r in present} >= {"python"}


_MODULE_LINKAGE_PATHS = (
    "scpn_phase_orchestrator.monitor.koopman_edmd",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_validation",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_mojo",
)


def test_module_linkage_paths_cover_the_koopman_chain() -> None:
    import importlib

    for import_path in _MODULE_LINKAGE_PATHS:
        assert importlib.import_module(import_path).__name__ == import_path
