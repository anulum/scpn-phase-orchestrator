# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for fractal-dimension kernels

"""Cross-backend parity for :func:`correlation_integral` and
:func:`kaplan_yorke_dimension`.

Parity tests run only against the **full-pairs** branch of
``correlation_integral`` — the subsampled branch is RNG-driven and
Rust keeps its own in-kernel RNG for API stability, so the only
universally identical output comes from the deterministic
triu-indices pair list. Tolerances:

* Rust / Julia / Go — 1e-12 (shared f64 on integer pair lists).
* Mojo — 1e-9 (subprocess text round-trip).
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
    _dimension_julia as dimension_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _dimension_mojo as dimension_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _dimension_validation as dimension_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_go import (
    correlation_integral_go,
    kaplan_yorke_dimension_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_julia import (
    correlation_integral_julia,
    kaplan_yorke_dimension_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_mojo import (
    _run as run_dimension_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_mojo import (
    correlation_integral_mojo,
    kaplan_yorke_dimension_mojo,
)
from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    AVAILABLE_BACKENDS,
    correlation_integral,
    kaplan_yorke_dimension,
)
from tests.typing_contracts import assert_precise_ndarray_hint

CiBackend = Callable[
    [np.ndarray, object, object, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
KyBackend = Callable[[np.ndarray], float]


def test__dimension_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(dimension_validation.expected_correlation_integral_backend_output)
    assert callable(dimension_validation.expected_kaplan_yorke_backend_output)
    assert callable(dimension_validation.validate_correlation_integral_backend_inputs)
    assert callable(dimension_validation.validate_correlation_integral_backend_output)
    assert callable(dimension_validation.validate_kaplan_yorke_backend_input)
    assert callable(dimension_validation.validate_kaplan_yorke_backend_output)


def _force(backend: str) -> str:
    prev = dim_mod.ACTIVE_BACKEND
    dim_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    dim_mod.ACTIVE_BACKEND = prev


def _reference_ci(
    traj: np.ndarray, eps: np.ndarray, max_pairs: int = 10_000
) -> np.ndarray:
    prev = _force("python")
    try:
        return correlation_integral(traj, eps, max_pairs=max_pairs)
    finally:
        _reset(prev)


def _reference_ky(le: np.ndarray) -> float:
    prev = _force("python")
    try:
        return kaplan_yorke_dimension(le)
    finally:
        _reset(prev)


def _trajectory(seed: int, t: int = 40, d: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (t, d))


def _eps(n_k: int = 10) -> np.ndarray:
    return np.logspace(-1, 0.5, n_k)


def test_backend_array_contracts_are_parameterised() -> None:
    ci_functions = (
        correlation_integral_go,
        correlation_integral_julia,
        correlation_integral_mojo,
    )
    ky_functions = (
        kaplan_yorke_dimension_go,
        kaplan_yorke_dimension_julia,
        kaplan_yorke_dimension_mojo,
    )
    for fn in ci_functions:
        hints = get_type_hints(fn)
        for key in ("traj_flat", "epsilons", "return"):
            assert_precise_ndarray_hint(hints[key])
            assert "float64" in str(hints[key])
        for key in ("idx_i", "idx_j"):
            assert_precise_ndarray_hint(hints[key])
            assert "int64" in str(hints[key])
    for fn in ky_functions:
        hints = get_type_hints(fn)
        assert_precise_ndarray_hint(hints["lyapunov_exponents"])
        assert "float64" in str(hints["lyapunov_exponents"])


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 1, "KY", "exactly 1 scalar"),
            ("2.0\n3.0\n", 1, "KY", "exactly 1 scalar"),
            ("0.0\n\n0.5\n1.0\n", 3, "CI", "exactly 3 scalar"),
            ("not-a-number\n", 1, "KY", "non-scalar"),
        ],
    )
    def test_mojo_dimension_stdout_contract_rejects_malformed_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(dimension_mojo, "_ensure_exe", lambda: "dimension_mojo")
        monkeypatch.setattr(
            dimension_mojo.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=stdout, stderr=""
            ),
        )

        with pytest.raises(ValueError, match=match):
            run_dimension_mojo(
                "KY 3 0.2 0.0 -0.5\n",
                expected_count=expected_count,
                label=label,
            )

    @pytest.mark.parametrize(
        "fn",
        [
            correlation_integral_go,
            correlation_integral_julia,
            correlation_integral_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("traj_flat", "t", "d", "idx_i", "idx_j", "epsilons", "message"),
        [
            (
                np.array([0.0, True], dtype=object),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "traj_flat",
            ),
            (
                np.array([0.0 + 1.0j, 1.0 + 0.0j]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "traj_flat",
            ),
            (
                np.array([0.0 + 1.0j, 1.0], dtype=object),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "traj_flat",
            ),
            (
                np.array([0.0, np.nan]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "traj_flat",
            ),
            (
                np.array([0.0, 1.0]),
                np.bool_(True),
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "t",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                0,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
                "d",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0, 1]),
                np.array([1]),
                np.array([0.1]),
                "same length",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0]),
                np.array([0]),
                np.array([0.1]),
                "self-pairs",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0]),
                np.array([2]),
                np.array([0.1]),
                "idx_j",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0 + 1j], dtype=object),
                np.array([1]),
                np.array([0.1]),
                "idx_i",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([-0.1]),
                "epsilons",
            ),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1 + 1.0j], dtype=object),
                "epsilons",
            ),
        ],
    )
    def test_correlation_integral_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: CiBackend,
        traj_flat: np.ndarray,
        t: object,
        d: object,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        epsilons: np.ndarray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(traj_flat, t, d, idx_i, idx_j, epsilons)

    @pytest.mark.parametrize(
        "fn",
        [
            kaplan_yorke_dimension_go,
            kaplan_yorke_dimension_julia,
            kaplan_yorke_dimension_mojo,
        ],
    )
    @pytest.mark.parametrize(
        "lyapunov_exponents",
        [
            np.array([0.1, np.bool_(False)], dtype=object),
            np.array([0.1 + 0.0j, -0.2 + 1.0j]),
            np.array([0.1 + 0.0j, -0.2], dtype=object),
            np.array([0.1, np.inf]),
            np.array([[0.1, -0.2]]),
        ],
    )
    def test_kaplan_yorke_backend_rejects_invalid_spectrum_before_runtime_load(
        self,
        fn: KyBackend,
        lyapunov_exponents: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="lyapunov_exponents"):
            fn(lyapunov_exponents)

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            (np.array([0.0, np.nan]), "finite"),
            (np.array([0.0, np.bool_(True)], dtype=object), "boolean"),
            (np.array([0.0, 0.5 + 0.1j], dtype=np.complex128), "real"),
            (np.array([0.0, 0.5 + 0.1j], dtype=object), "real"),
            (np.array([0.0, 1.2]), "\\[0, 1\\]"),
            (np.array([0.5, 0.4]), "non-decreasing"),
            (np.array([0.0]), "length"),
        ],
    )
    def test_correlation_integral_output_validation_rejects_nonphysical_values(
        self, values: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            dimension_validation.validate_correlation_integral_backend_output(
                values,
                np.array([0.1, 0.2]),
            )

    def test_correlation_integral_output_validation_rejects_exact_divergence(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="exact reference"):
            dimension_validation.validate_correlation_integral_backend_output(
                np.array([0.0, 0.5]),
                np.array([0.1, 0.2]),
                expected=np.array([0.0, 1.0]),
            )

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.bool_(True), "boolean"),
            (1.0 + 0.1j, "real"),
            (np.array(1.0 + 0.1j, dtype=object), "real"),
            (np.inf, "finite"),
            (-0.5, "\\[0, spectrum length\\]"),
            (4.5, "\\[0, spectrum length\\]"),
        ],
    )
    def test_kaplan_yorke_output_validation_rejects_nonphysical_values(
        self, value: object, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            dimension_validation.validate_kaplan_yorke_backend_output(
                value,
                np.array([0.2, 0.0, -0.5]),
            )

    def test_kaplan_yorke_output_validation_rejects_exact_divergence(self) -> None:
        with pytest.raises(ValueError, match="exact reference"):
            dimension_validation.validate_kaplan_yorke_backend_output(
                1.0,
                np.array([0.5, 0.1, -0.2, -0.5]),
                expected=3.2,
            )

    def test_julia_backend_rejects_nonmonotone_correlation_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def correlation_integral(*args: object) -> np.ndarray:
                return np.array([0.4, 0.3])

        monkeypatch.setattr(dimension_julia, "_ensure", lambda: _FakeJulia())

        with pytest.raises(ValueError, match="non-decreasing"):
            correlation_integral_julia(
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1, 0.2]),
            )

    def test_julia_backend_rejects_exact_correlation_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def correlation_integral(*args: object) -> np.ndarray:
                return np.array([0.0])

        monkeypatch.setattr(dimension_julia, "_ensure", lambda: _FakeJulia())

        with pytest.raises(ValueError, match="exact reference"):
            correlation_integral_julia(
                np.array([0.0, 0.05]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
            )

    def test_mojo_backend_rejects_exact_correlation_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            dimension_mojo,
            "_run",
            lambda payload, *, expected_count, label: [0.0],
        )

        with pytest.raises(ValueError, match="exact reference"):
            correlation_integral_mojo(
                np.array([0.0, 0.05]),
                2,
                1,
                np.array([0]),
                np.array([1]),
                np.array([0.1]),
            )

    def test_mojo_backend_rejects_out_of_bounds_kaplan_yorke_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            dimension_mojo,
            "_run",
            lambda payload, *, expected_count, label: [4.0],
        )

        with pytest.raises(ValueError, match="\\[0, spectrum length\\]"):
            kaplan_yorke_dimension_mojo(np.array([0.2, 0.0, -0.5]))

    def test_mojo_backend_rejects_exact_kaplan_yorke_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            dimension_mojo,
            "_run",
            lambda payload, *, expected_count, label: [1.0],
        )

        with pytest.raises(ValueError, match="exact reference"):
            kaplan_yorke_dimension_mojo(np.array([0.5, 0.1, -0.2, -0.5]))

    def test_public_correlation_integral_rejects_exact_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _bad_backend(
            traj_flat: np.ndarray,
            t: int,
            d: int,
            idx_i: np.ndarray,
            idx_j: np.ndarray,
            epsilons: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(epsilons.size, dtype=np.float64)

        monkeypatch.setattr(dim_mod, "_dispatch", lambda name: _bad_backend)
        prev = _force("go")
        try:
            with pytest.raises(ValueError, match="exact reference"):
                correlation_integral(
                    np.array([[0.0], [0.05]]),
                    np.array([0.1]),
                    max_pairs=10,
                )
        finally:
            _reset(prev)

    def test_public_kaplan_yorke_rejects_exact_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(dim_mod, "_dispatch", lambda name: lambda le: 1.0)
        prev = _force("go")
        try:
            with pytest.raises(ValueError, match="exact reference"):
                kaplan_yorke_dimension(np.array([0.4, -0.2, -0.8]))
        finally:
            _reset(prev)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        t=st.integers(min_value=5, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_ci_full_pairs(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("rust")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.5, 0.1, -0.2, -0.5, -0.9])
        ref = _reference_ky(le)
        prev = _force("rust")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_ci_full_pairs(self, seed: int) -> None:
        traj = _trajectory(seed)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("julia")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.3, 0.0, -0.1, -0.8])
        ref = _reference_ky(le)
        prev = _force("julia")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        t=st.integers(min_value=5, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_ci_full_pairs(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("go")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.7, 0.0, -1.2])
        ref = _reference_ky(le)
        prev = _force("go")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_ci_full_pairs(self, seed: int) -> None:
        traj = _trajectory(seed, t=20)
        eps = _eps(n_k=6)
        ref = _reference_ci(traj, eps)
        prev = _force("mojo")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)

    def test_ky(self) -> None:
        le = np.array([0.25, 0.05, -0.4])
        ref = _reference_ky(le)
        prev = _force("mojo")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        traj = _trajectory(2026, t=30)
        eps = _eps()
        le = np.array([0.4, 0.1, -0.2, -0.5])
        ref_ci = _reference_ci(traj, eps)
        ref_ky = _reference_ky(le)
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
                got_ci = correlation_integral(traj, eps, max_pairs=10_000)
                got_ky = kaplan_yorke_dimension(le)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                got_ci,
                ref_ci,
                atol=tolerances[backend],
                err_msg=f"{backend} CI diverged from python",
            )
            assert abs(got_ky - ref_ky) <= tolerances[backend], (
                f"{backend} KY diverged: {got_ky} vs {ref_ky}"
            )
