# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for delayed Kuramoto

"""Cross-backend parity for :class:`DelayedEngine.run`.

Every accelerated backend must reproduce the NumPy reference final phases
within 1e-9 (Rust / Julia / Go) or 1e-6 (Mojo, subprocess text round-trip).
Direct backend adapters validate the state arrays, scalars, and step counts
before any runtime loads.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _delay_validation as delay_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_go import (
    delayed_kuramoto_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_julia import (
    delayed_kuramoto_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_mojo import (
    delayed_kuramoto_run_mojo,
)
from scpn_phase_orchestrator.upde import delay as delay_mod
from scpn_phase_orchestrator.upde.delay import AVAILABLE_BACKENDS, DelayedEngine

TWO_PI = 2.0 * np.pi


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.uniform(-1.0, 1.0, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(0.0, 0.3, (n, n))
    return phases, omegas, knm, alpha


def _reference(problem, delay, steps):
    phases, omegas, knm, alpha = problem
    prev = delay_mod.ACTIVE_BACKEND
    delay_mod.ACTIVE_BACKEND = "python"
    try:
        eng = DelayedEngine(len(phases), dt=0.05, delay_steps=delay)
        return eng.run(phases, omegas, knm, 0.3, 0.7, alpha, n_steps=steps)
    finally:
        delay_mod.ACTIVE_BACKEND = prev


def _assert_parity(backend: str, seed: int, atol: float, delay: int = 2) -> None:
    problem = _problem(seed)
    ref = _reference(problem, delay, 40)
    phases, omegas, knm, alpha = problem
    prev = delay_mod.ACTIVE_BACKEND
    delay_mod.ACTIVE_BACKEND = backend
    try:
        eng = DelayedEngine(len(phases), dt=0.05, delay_steps=delay)
        got = eng.run(phases, omegas, knm, 0.3, 0.7, alpha, n_steps=40)
    finally:
        delay_mod.ACTIVE_BACKEND = prev
    np.testing.assert_allclose(got, ref, atol=atol)


def test__delay_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(delay_validation.validate_delay_backend_inputs)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("rust", seed, atol=1e-9)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("julia", seed, atol=1e-9)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @pytest.mark.parametrize("seed", [0, 13])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("go", seed, atol=1e-9)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("mojo", seed, atol=1e-6)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        problem = _problem(2026, n=8)
        ref = _reference(problem, 3, 60)
        phases, omegas, knm, alpha = problem
        tol = {"rust": 1e-9, "julia": 1e-9, "go": 1e-9, "mojo": 1e-6, "python": 0.0}
        for backend in AVAILABLE_BACKENDS:
            prev = delay_mod.ACTIVE_BACKEND
            delay_mod.ACTIVE_BACKEND = backend
            try:
                eng = DelayedEngine(len(phases), dt=0.05, delay_steps=3)
                got = eng.run(phases, omegas, knm, 0.3, 0.7, alpha, n_steps=60)
            finally:
                delay_mod.ACTIVE_BACKEND = prev
            np.testing.assert_allclose(got, ref, atol=tol[backend])


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        "backend",
        [
            delayed_kuramoto_run_go,
            delayed_kuramoto_run_julia,
            delayed_kuramoto_run_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"phases": np.array([True, False])}, "phases"),
            ({"phases": np.array([0.0, np.nan])}, "finite"),
            ({"phases": np.array([0.0, 1.0 + 0j])}, "real-valued"),
            ({"phases": np.array(["a", "b"])}, "finite float array"),
            ({"phases": np.zeros((2, 1))}, "one-dimensional"),
            ({"phases": np.zeros(3)}, "phases length"),
            ({"omegas": np.zeros(3)}, "omegas length"),
            ({"knm_flat": np.zeros(3)}, "knm_flat length"),
            ({"alpha_flat": np.zeros(3)}, "alpha_flat length"),
            ({"n": 0}, "n"),
            ({"n": 1.5}, "n"),
            ({"dt": 0.0}, "dt"),
            ({"delay_steps": 0}, "delay_steps"),
            ({"zeta": np.inf}, "zeta"),
            ({"zeta": None}, "zeta"),
        ],
    )
    def test_validation_precedes_runtime_load(self, backend, kwargs, match) -> None:
        base = {
            "phases": np.zeros(2),
            "omegas": np.zeros(2),
            "knm_flat": np.zeros(4),
            "alpha_flat": np.zeros(4),
            "n": 2,
            "zeta": 0.0,
            "psi": 0.0,
            "dt": 0.05,
            "delay_steps": 1,
            "n_steps": 5,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            backend(
                base["phases"],
                base["omegas"],
                base["knm_flat"],
                base["alpha_flat"],
                base["n"],
                base["zeta"],
                base["psi"],
                base["dt"],
                base["delay_steps"],
                base["n_steps"],
            )


class TestBackendLoaderDispatch:
    def test_rust_loader_wraps_spo_kernel(self, monkeypatch) -> None:
        calls: list[tuple] = []

        def fake_rust(phases, omegas, knm, alpha, n, zeta, psi, dt, delay, steps):
            calls.append((n, delay, steps))
            return np.zeros(n, dtype=np.float64)

        fake_module = types.ModuleType("spo_kernel")
        fake_module.delayed_kuramoto_run_rust = fake_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)

        backend = delay_mod._load_rust_fn()
        out = backend(
            np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 0.0, 0.0, 0.05, 1, 5
        )
        assert out.shape == (2,)
        assert calls[0] == (2, 1, 5)

    def test_resolve_backends_falls_back_to_python(self, monkeypatch) -> None:
        def _fail() -> delay_mod._DelayBackend:
            raise RuntimeError("unavailable")

        for name in ("rust", "mojo", "julia", "go"):
            monkeypatch.setitem(delay_mod._LOADERS, name, _fail)
        active, available = delay_mod._resolve_backends()
        assert active == "python"
        assert available == ["python"]


class _ArrayRaises:
    """Object whose array conversion always fails (probes the defensive guards)."""

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise ValueError("no array interface")


class TestDispatchFallthrough:
    def test_dispatch_skips_failing_active_and_dedups(self, monkeypatch) -> None:
        sentinel: delay_mod._DelayBackend = lambda *_a, **_k: np.zeros(2)  # noqa: E731

        def fake_load(name: str) -> delay_mod._DelayBackend:
            if name == "rust":
                raise RuntimeError("rust unavailable at dispatch")
            return sentinel

        monkeypatch.setattr(delay_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(delay_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(delay_mod, "_load_backend", fake_load)

        # ordered = [rust(active), rust, go, python]: active rust raises, the
        # second rust is deduplicated, go resolves.
        assert delay_mod._dispatch() is sentinel

    def test_dispatch_returns_none_when_only_python(self, monkeypatch) -> None:
        monkeypatch.setattr(delay_mod, "ACTIVE_BACKEND", "python")
        monkeypatch.setattr(delay_mod, "AVAILABLE_BACKENDS", ["python"])
        assert delay_mod._dispatch() is None

    def test_dispatch_none_when_all_accelerators_fail(self, monkeypatch) -> None:
        def fake_load(_name: str) -> delay_mod._DelayBackend:
            raise RuntimeError("no accelerator resolves")

        monkeypatch.setattr(delay_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(delay_mod, "AVAILABLE_BACKENDS", ["rust", "go"])
        monkeypatch.setattr(delay_mod, "_load_backend", fake_load)
        # No "python" entry and every accelerator fails: dispatch must still
        # signal the NumPy fallback rather than raise.
        assert delay_mod._dispatch() is None


class TestBackendOutputContract:
    """A backend that violates the phase-vector contract falls back to NumPy."""

    @pytest.mark.parametrize(
        "bad_output",
        [
            np.zeros(3),  # wrong oscillator count
            np.array([0.0, 100.0]),  # outside [0, 2*pi)
            np.array(["a", "b"]),  # not convertible to float
            np.array([0.0, np.nan]),  # non-finite
        ],
    )
    def test_invalid_backend_output_falls_back(self, monkeypatch, bad_output) -> None:
        def bad_backend(*_a: object, **_k: object) -> np.ndarray:
            return bad_output

        monkeypatch.setattr(delay_mod, "_dispatch", lambda: bad_backend)
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        out = eng.run(np.array([0.4, 1.1]), np.zeros(2), np.zeros((2, 2)), n_steps=5)
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out < TWO_PI))


class TestStateArrayDefensiveGuards:
    def test_step_rejects_unconvertible_object(self) -> None:
        eng = DelayedEngine(2, dt=0.05, delay_steps=1)
        with pytest.raises(ValueError, match="finite float array"):
            eng.step(_ArrayRaises(), np.zeros(2), np.zeros((2, 2)))

    def test_boolean_alias_probe_handles_unconvertible(self) -> None:
        assert delay_mod._contains_boolean_alias(_ArrayRaises()) is False
        assert delay_validation._contains_boolean_alias(_ArrayRaises()) is False
