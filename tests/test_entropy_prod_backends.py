# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for entropy production rate

"""Cross-backend parity for :func:`entropy_production_rate`.

All backends must agree with the Python reference on the same input.
Tolerances:

* Rust / Julia / Go — 1e-12 (shared f64).
* Mojo — 1e-9 (subprocess text round-trip; empirically 1e-19).
"""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _entropy_prod_go,
    _entropy_prod_julia,
    _entropy_prod_mojo,
    _entropy_prod_validation,
)
from scpn_phase_orchestrator.monitor import entropy_prod as ep_mod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    AVAILABLE_BACKENDS,
    entropy_production_rate,
)
from tests.typing_contracts import assert_precise_ndarray_hint

entropy_production_rate_go = _entropy_prod_go.entropy_production_rate_go
entropy_production_rate_julia = _entropy_prod_julia.entropy_production_rate_julia
entropy_production_rate_mojo = _entropy_prod_mojo.entropy_production_rate_mojo

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = ep_mod.ACTIVE_BACKEND
    ep_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    ep_mod.ACTIVE_BACKEND = prev


def _reference(phases, omegas, knm, alpha, dt):
    prev = _force("python")
    try:
        return entropy_production_rate(phases, omegas, knm, alpha, dt)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 0.9, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    return phases, omegas, knm


def test__entropy_prod_validation_rejects_boolean_alias_inputs() -> None:
    with pytest.raises(ValueError, match="knm must not contain boolean values"):
        _entropy_prod_validation.validate_entropy_prod_backend_inputs(
            np.zeros(2),
            np.ones(2),
            np.array([[0.0, True], [0.0, 0.0]], dtype=object),
            1.0,
            0.01,
        )


def test__entropy_prod_validation_rejects_complex_inputs() -> None:
    with pytest.raises(ValueError, match="phases must contain real-valued samples"):
        _entropy_prod_validation.validate_entropy_prod_backend_inputs(
            np.array([0.0 + 1.0j, 1.0 + 0.0j]),
            np.zeros(2),
            np.zeros((2, 2)),
            1.0,
            0.01,
        )

    with pytest.raises(ValueError, match="knm must contain real-valued couplings"):
        _entropy_prod_validation.validate_entropy_prod_backend_inputs(
            np.zeros(2),
            np.zeros(2),
            np.array([[0.0 + 0.0j, 1.0 + 1.0j], [1.0, 0.0]]),
            1.0,
            0.01,
        )


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        entropy_production_rate_go,
        entropy_production_rate_julia,
        entropy_production_rate_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        for key in ("phases", "omegas", "knm"):
            assert_precise_ndarray_hint(hints[key])
            assert "float64" in str(hints[key])


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, omegas, knm = _problem(seed, n)
        ref = _reference(phases, omegas, knm, 0.6, 0.01)
        prev = _force("rust")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.6, 0.01)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm = _problem(seed)
        ref = _reference(phases, omegas, knm, 0.5, 0.01)
        prev = _force("julia")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, omegas, knm = _problem(seed, n)
        ref = _reference(phases, omegas, knm, 0.4, 0.02)
        prev = _force("go")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.4, 0.02)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm = _problem(seed)
        ref = _reference(phases, omegas, knm, 0.7, 0.01)
        prev = _force("mojo")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.7, 0.01)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases, omegas, knm = _problem(2026, n=10)
        ref = _reference(phases, omegas, knm, 0.5, 0.01)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
            finally:
                _reset(prev)
            assert abs(got - ref) <= tolerances[backend], (
                f"{backend} diverged: {got} vs {ref}"
            )


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("go", entropy_production_rate_go),
        ("julia", entropy_production_rate_julia),
        ("mojo", entropy_production_rate_mojo),
    ],
)
class TestEntropyProductionAdapterContracts:
    """Backend adapters reject invalid monitor inputs before polyglot execution."""

    def test_rejects_boolean_alias_arrays_before_runtime(
        self,
        name: str,
        backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if name == "go":
            monkeypatch.setattr(_entropy_prod_go, "_load_lib", lambda: None)
        elif name == "julia":
            monkeypatch.setattr(_entropy_prod_julia, "_ensure", lambda: None)
        else:
            monkeypatch.setattr(_entropy_prod_mojo, "_run", lambda _payload: 0.0)

        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            backend(
                np.array([0.0, np.bool_(True)], dtype=object),
                np.zeros(2),
                np.zeros((2, 2)),
                1.0,
                0.01,
            )

    def test_rejects_shape_mismatch_before_runtime(
        self,
        name: str,
        backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if name == "go":
            monkeypatch.setattr(_entropy_prod_go, "_load_lib", lambda: None)
        elif name == "julia":
            monkeypatch.setattr(_entropy_prod_julia, "_ensure", lambda: None)
        else:
            monkeypatch.setattr(_entropy_prod_mojo, "_run", lambda _payload: 0.0)

        with pytest.raises(ValueError, match="omegas shape"):
            backend(
                np.zeros(2),
                np.zeros(1),
                np.zeros((2, 2)),
                1.0,
                0.01,
            )

    def test_rejects_complex_arrays_before_runtime(
        self,
        name: str,
        backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if name == "go":
            monkeypatch.setattr(_entropy_prod_go, "_load_lib", lambda: None)
        elif name == "julia":
            monkeypatch.setattr(_entropy_prod_julia, "_ensure", lambda: None)
        else:
            monkeypatch.setattr(_entropy_prod_mojo, "_run", lambda _payload: 0.0)

        with pytest.raises(ValueError, match="omegas must contain real-valued samples"):
            backend(
                np.zeros(2),
                np.array([0.0 + 1.0j, 1.0 + 0.0j]),
                np.zeros((2, 2)),
                1.0,
                0.01,
            )

    def test_rejects_boolean_scalar_before_runtime(
        self,
        name: str,
        backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if name == "go":
            monkeypatch.setattr(_entropy_prod_go, "_load_lib", lambda: None)
        elif name == "julia":
            monkeypatch.setattr(_entropy_prod_julia, "_ensure", lambda: None)
        else:
            monkeypatch.setattr(_entropy_prod_mojo, "_run", lambda _payload: 0.0)

        with pytest.raises(ValueError, match="alpha must be a finite real"):
            backend(
                np.zeros(2),
                np.zeros(2),
                np.zeros((2, 2)),
                np.bool_(True),
                0.01,
            )
