# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for entropy_from_phases

"""Cross-backend parity for :func:`entropy_from_phases`.

Tolerances: Rust / Julia / Go within 1e-12; Mojo within 1e-9 due to
the subprocess text round-trip on the bin-edge float comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _psychedelic_go,
    _psychedelic_julia,
    _psychedelic_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _psychedelic_validation as psychedelic_validation,
)
from scpn_phase_orchestrator.monitor import psychedelic as py_mod
from scpn_phase_orchestrator.monitor.psychedelic import (
    AVAILABLE_BACKENDS,
    entropy_from_phases,
)
from tests.typing_contracts import assert_precise_ndarray_hint

entropy_from_phases_go = _psychedelic_go.entropy_from_phases_go
entropy_from_phases_julia = _psychedelic_julia.entropy_from_phases_julia
entropy_from_phases_mojo = _psychedelic_mojo.entropy_from_phases_mojo

TWO_PI = 2.0 * np.pi
EntropyBackend = Callable[[np.ndarray, object], float]


def _force(backend: str) -> str:
    prev = py_mod.ACTIVE_BACKEND
    py_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    py_mod.ACTIVE_BACKEND = prev


def _reference(phases: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return entropy_from_phases(phases, n_bins)
    finally:
        _reset(prev)


def _phases(seed: int, n: int = 500) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, TWO_PI, n)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        entropy_from_phases_go,
        entropy_from_phases_julia,
        entropy_from_phases_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        assert_precise_ndarray_hint(hints["phases"])
        assert "float64" in str(hints["phases"])


def test__psychedelic_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(psychedelic_validation.validate_psychedelic_backend_inputs)
    assert callable(psychedelic_validation.validate_psychedelic_entropy_backend_output)


class _FakeGoPsychedelic:
    def __init__(self, entropy: float) -> None:
        self.entropy = entropy

    def EntropyFromPhases(
        self,
        _phases_ptr: object,
        _n: object,
        _n_bins: object,
        out_ptr: object,
    ) -> int:
        out_ptr._obj.value = self.entropy
        return 0


class _FakeJuliaPsychedelic:
    def __init__(self, entropy: object) -> None:
        self.entropy = entropy

    def entropy_from_phases(
        self,
        _phase_values: np.ndarray,
        _bin_count: int,
    ) -> object:
        return self.entropy


class TestDirectBackendBoundaryContracts:
    """Direct optional psychedelic backends validate before runtime loading."""

    @pytest.mark.parametrize(
        "backend",
        [
            entropy_from_phases_go,
            entropy_from_phases_julia,
            entropy_from_phases_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("phases", "n_bins", "error", "match"),
        [
            (np.array([0.0, True], dtype=object), 4, ValueError, "boolean"),
            (np.array([0.0, 1.0 + 0.0j]), 4, ValueError, "real-valued"),
            (
                np.array([0.0, complex(1.0, 0.0)], dtype=object),
                4,
                ValueError,
                "real-valued",
            ),
            (np.array([0.0, np.inf]), 4, ValueError, "finite"),
            (np.array([[0.0, 1.0]]), 4, ValueError, "one-dimensional"),
            (np.linspace(0.0, 1.0, 4), np.bool_(True), TypeError, "n_bins"),
            (np.linspace(0.0, 1.0, 4), 1, ValueError, "n_bins"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: EntropyBackend,
        phases: np.ndarray,
        n_bins: object,
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            backend(phases, n_bins)

    @pytest.mark.parametrize(
        "backend",
        [
            entropy_from_phases_go,
            entropy_from_phases_julia,
            entropy_from_phases_mojo,
        ],
    )
    def test_empty_phase_entropy_returns_zero_before_runtime_load(
        self,
        backend: EntropyBackend,
    ) -> None:
        assert backend(np.array([], dtype=np.float64), 36) == 0.0

    @pytest.mark.parametrize(
        ("backend_name", "payload", "match"),
        [
            ("go", np.inf, "finite"),
            ("julia", 2.0, r"\[0, log\(n_bins\)\]"),
            ("mojo", -0.1, r"\[0, log\(n_bins\)\]"),
            ("julia", complex(1.0, 0.0), "real-valued"),
        ],
    )
    def test_invalid_entropy_backend_outputs_are_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_name: str,
        payload: float,
        match: str,
    ) -> None:
        phases = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        if backend_name == "go":
            monkeypatch.setattr(
                _psychedelic_go,
                "_load_lib",
                lambda: _FakeGoPsychedelic(float(payload)),
            )
            backend = entropy_from_phases_go
        elif backend_name == "julia":
            monkeypatch.setattr(
                _psychedelic_julia,
                "_ensure",
                lambda: _FakeJuliaPsychedelic(payload),
            )
            backend = entropy_from_phases_julia
        else:
            monkeypatch.setattr(_psychedelic_mojo, "_ensure_exe", lambda: "fake")

            def fake_run(*_args: object, **_kwargs: object) -> object:
                class Result:
                    returncode = 0
                    stderr = ""
                    stdout = f"{payload}\n"

                return Result()

            monkeypatch.setattr(_psychedelic_mojo.subprocess, "run", fake_run)
            backend = entropy_from_phases_mojo

        with pytest.raises(ValueError, match=match):
            backend(phases, 4)

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("\n0.1\n", "Mojo entropy returned 2"),
            ("0.1\n\n", "Mojo entropy returned 2"),
            ("0.1\n0.2\n", "Mojo entropy returned 2"),
            ("not-a-scalar\n", "non-scalar psychedelic value"),
        ],
    )
    def test_mojo_rejects_malformed_entropy_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_psychedelic_mojo, "_ensure_exe", lambda: "fake")

        def fake_run(*_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(returncode=0, stderr="", stdout=stdout)

        monkeypatch.setattr(_psychedelic_mojo.subprocess, "run", fake_run)

        with pytest.raises(ValueError, match=match):
            entropy_from_phases_mojo(np.linspace(0.0, 1.0, 8), 4)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=10, max_value=2000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases = _phases(seed, n)
        ref = _reference(phases, 36)
        prev = _force("rust")
        try:
            got = entropy_from_phases(phases, 36)
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
        phases = _phases(seed)
        ref = _reference(phases, 36)
        prev = _force("julia")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=10, max_value=2000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases = _phases(seed, n)
        ref = _reference(phases, 36)
        prev = _force("go")
        try:
            got = entropy_from_phases(phases, 36)
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
        phases = _phases(seed)
        ref = _reference(phases, 36)
        prev = _force("mojo")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases = _phases(2026, n=1000)
        ref = _reference(phases, 36)
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
                got = entropy_from_phases(phases, 36)
            finally:
                _reset(prev)
            assert abs(got - ref) <= tolerances[backend], (
                f"{backend} diverged: {got} vs {ref}"
            )
