# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for recurrence kernels

"""Cross-backend parity for :func:`recurrence_matrix` and
:func:`cross_recurrence_matrix`.

Outputs are booleans — tolerance is exact array equality. Both
``euclidean`` and ``angular`` metric branches are exercised.
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
    _recurrence_go,
    _recurrence_julia,
    _recurrence_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _recurrence_validation as recurrence_validation,
)
from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    AVAILABLE_BACKENDS,
    cross_recurrence_matrix,
    recurrence_matrix,
)
from tests.typing_contracts import assert_precise_ndarray_hint

RmBackend = Callable[[np.ndarray, object, object, object, object], np.ndarray]
CrossBackend = Callable[
    [np.ndarray, np.ndarray, object, object, object, object],
    np.ndarray,
]

cross_recurrence_matrix_go = _recurrence_go.cross_recurrence_matrix_go
cross_recurrence_matrix_julia = _recurrence_julia.cross_recurrence_matrix_julia
cross_recurrence_matrix_mojo = _recurrence_mojo.cross_recurrence_matrix_mojo
recurrence_matrix_go = _recurrence_go.recurrence_matrix_go
recurrence_matrix_julia = _recurrence_julia.recurrence_matrix_julia
recurrence_matrix_mojo = _recurrence_mojo.recurrence_matrix_mojo


def test__recurrence_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(recurrence_validation.expected_recurrence_backend_output)
    assert callable(recurrence_validation.validate_recurrence_backend_inputs)
    assert callable(recurrence_validation.validate_cross_recurrence_backend_inputs)
    assert callable(recurrence_validation.validate_recurrence_backend_output)


def _force(backend: str) -> str:
    prev = r_mod.ACTIVE_BACKEND
    r_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    r_mod.ACTIVE_BACKEND = prev


def _reference_rm(
    traj: np.ndarray,
    epsilon: float,
    metric: str = "euclidean",
) -> np.ndarray:
    prev = _force("python")
    try:
        return recurrence_matrix(traj, epsilon, metric)
    finally:
        _reset(prev)


def _reference_cross(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    prev = _force("python")
    try:
        return cross_recurrence_matrix(a, b, epsilon)
    finally:
        _reset(prev)


def _trajectory(seed: int, t: int = 30, d: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, (t, d))


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        recurrence_matrix_go,
        cross_recurrence_matrix_go,
        recurrence_matrix_julia,
        cross_recurrence_matrix_julia,
        recurrence_matrix_mojo,
        cross_recurrence_matrix_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        checked_hints = [hints["return"]]
        checked_hints.extend(
            value
            for key, value in hints.items()
            if key in {"traj_flat", "traj_a_flat", "traj_b_flat"}
        )
        for hint in checked_hints:
            assert_precise_ndarray_hint(hint)
        float_hint = (
            hints["traj_flat"] if "traj_flat" in hints else hints["traj_a_flat"]
        )
        assert "float64" in str(float_hint)
        assert "uint8" in str(hints["return"])


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 4, "REC", "exactly 4 integer"),
            ("1\n0\n0\n1\n1\n", 4, "REC", "exactly 4 integer"),
            ("1\n\n0\n0\n1\n", 4, "CROSS", "exactly 4 integer"),
            ("1\nnot-an-int\n0\n1\n", 4, "REC", "non-integer"),
        ],
    )
    def test_mojo_recurrence_stdout_contract_rejects_malformed_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_recurrence_mojo, "_ensure_exe", lambda: "recurrence_mojo")
        monkeypatch.setattr(
            _recurrence_mojo.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=stdout, stderr=""
            ),
        )

        with pytest.raises(ValueError, match=match):
            _recurrence_mojo._run(
                "REC 2 1 0 0.5 0.0 1.0\n",
                expected_count=expected_count,
                label=label,
            )

    @pytest.mark.parametrize(
        "fn",
        [
            recurrence_matrix_go,
            recurrence_matrix_julia,
            recurrence_matrix_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("traj_flat", "t", "d", "epsilon", "angular", "message"),
        [
            (np.array([0.0, True], dtype=object), 2, 1, 0.5, False, "traj_flat"),
            (np.array(["0.0", "1.0"], dtype=str), 2, 1, 0.5, False, "numeric-string"),
            (
                np.array([0.0, "1.0"], dtype=object),
                2,
                1,
                0.5,
                False,
                "numeric-string",
            ),
            (np.array([0.0 + 1.0j, 1.0 + 0.0j]), 2, 1, 0.5, False, "traj_flat"),
            (np.array([0.0, np.inf]), 2, 1, 0.5, False, "traj_flat"),
            (np.array([[0.0, 1.0]]), 2, 1, 0.5, False, "traj_flat"),
            (np.array([0.0, 1.0]), 3, 1, 0.5, False, "traj_flat"),
            (np.array([0.0, 1.0]), np.bool_(True), 1, 0.5, False, "t"),
            (np.array([0.0, 1.0]), 2, 0, 0.5, False, "d"),
            (np.array([0.0, 1.0]), 2, 1, np.nan, False, "epsilon"),
            (np.array([0.0, 1.0]), 2, 1, -0.1, False, "epsilon"),
            (np.array([0.0, 1.0]), 2, 1, 0.5, 1, "angular"),
        ],
    )
    def test_recurrence_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: RmBackend,
        traj_flat: np.ndarray,
        t: object,
        d: object,
        epsilon: object,
        angular: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(traj_flat, t, d, epsilon, angular)

    @pytest.mark.parametrize(
        "fn",
        [
            cross_recurrence_matrix_go,
            cross_recurrence_matrix_julia,
            cross_recurrence_matrix_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("traj_a", "traj_b", "t", "d", "epsilon", "angular", "message"),
        [
            (
                np.array([0.0, 1.0]),
                np.array([0.0, np.bool_(True)], dtype=object),
                2,
                1,
                0.5,
                False,
                "traj_b_flat",
            ),
            (
                np.array([0.0, 1.0]),
                np.array(["0.0", "1.0"], dtype=str),
                2,
                1,
                0.5,
                False,
                "numeric-string",
            ),
            (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0j]),
                2,
                1,
                0.5,
                False,
                "traj_b_flat",
            ),
            (
                np.array([0.0, 1.0]),
                np.array([0.0]),
                2,
                1,
                0.5,
                False,
                "traj_b_flat",
            ),
            (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                2,
                1,
                np.bool_(False),
                False,
                "epsilon",
            ),
        ],
    )
    def test_cross_recurrence_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: CrossBackend,
        traj_a: np.ndarray,
        traj_b: np.ndarray,
        t: object,
        d: object,
        epsilon: object,
        angular: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(traj_a, traj_b, t, d, epsilon, angular)

    @pytest.mark.parametrize(
        ("value", "name", "match"),
        [
            (np.array([0, 1, 1]), "recurrence_matrix", "size"),
            (np.array([0, 1, 2, 1]), "recurrence_matrix", "0/1"),
            (np.array([1, np.inf, 0, 1]), "recurrence_matrix", "finite"),
            (
                np.array(["1", "0", "0", "1"], dtype=str),
                "recurrence_matrix",
                "numeric-string",
            ),
            (np.array([0, 1, 1, 1]), "recurrence_matrix", "true diagonal"),
            (np.array([1, 1, 0, 1]), "recurrence_matrix", "symmetric"),
            (np.array([0, 1, 2, 1]), "cross_recurrence_matrix", "0/1"),
        ],
    )
    def test_output_validation_rejects_nonphysical_recurrence_values(
        self, value: np.ndarray, name: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            recurrence_validation.validate_recurrence_backend_output(
                value,
                t=2,
                name=name,
            )

    def test_output_validation_rejects_numeric_string_expected(self) -> None:
        with pytest.raises(ValueError, match="expected output .*numeric-string"):
            recurrence_validation.validate_recurrence_backend_output(
                np.array([1, 0, 0, 1], dtype=np.uint8),
                t=2,
                name="recurrence_matrix",
                expected=np.array(["1", "0", "0", "1"], dtype=str),
            )

    def test_output_validation_rejects_exact_threshold_divergence(self) -> None:
        traj = np.array([0.0, 2.0, 5.0], dtype=np.float64)
        expected = recurrence_validation.expected_recurrence_backend_output(
            traj,
            traj,
            t=3,
            d=1,
            epsilon=0.5,
            angular=False,
        )
        symmetric_wrong = np.ones((3, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="exact recurrence threshold"):
            recurrence_validation.validate_recurrence_backend_output(
                symmetric_wrong,
                t=3,
                name="recurrence_matrix",
                expected=expected,
            )

    def test_cross_output_validation_rejects_exact_threshold_divergence(self) -> None:
        traj_a = np.array([0.0, 2.0], dtype=np.float64)
        traj_b = np.array([0.0, 3.0], dtype=np.float64)
        expected = recurrence_validation.expected_recurrence_backend_output(
            traj_a,
            traj_b,
            t=2,
            d=1,
            epsilon=0.5,
            angular=False,
        )
        binary_wrong = np.array([1, 1, 0, 1], dtype=np.uint8)

        with pytest.raises(ValueError, match="exact recurrence threshold"):
            recurrence_validation.validate_recurrence_backend_output(
                binary_wrong,
                t=2,
                name="cross_recurrence_matrix",
                expected=expected,
            )

    def test_julia_backend_rejects_asymmetric_recurrence_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def recurrence_matrix(*args: object) -> np.ndarray:
                return np.array([1, 1, 0, 1], dtype=np.uint8)

        monkeypatch.setattr(_recurrence_julia, "_ensure", lambda: _FakeJulia())

        with pytest.raises(ValueError, match="symmetric"):
            recurrence_matrix_julia(np.array([0.0, 1.0]), 2, 1, 0.5, False)

    def test_julia_backend_rejects_exact_threshold_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def recurrence_matrix(*args: object) -> np.ndarray:
                return np.ones((3, 3), dtype=np.uint8)

        monkeypatch.setattr(_recurrence_julia, "_ensure", lambda: _FakeJulia())

        with pytest.raises(ValueError, match="exact recurrence threshold"):
            recurrence_matrix_julia(np.array([0.0, 2.0, 5.0]), 3, 1, 0.5, False)

    def test_mojo_backend_rejects_nonbinary_cross_recurrence_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _recurrence_mojo,
            "_run",
            lambda payload, *, expected_count, label: [0, 1, 2, 1],
        )

        with pytest.raises(ValueError, match="0/1"):
            cross_recurrence_matrix_mojo(
                np.array([0.0, 1.0]),
                np.array([1.0, 2.0]),
                2,
                1,
                0.5,
                False,
            )

    def test_public_recurrence_rejects_shape_correct_wrong_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        traj = np.array([0.0, 2.0, 5.0], dtype=np.float64)

        def _wrong_backend(
            traj_flat: np.ndarray,
            t: int,
            d: int,
            epsilon: float,
            angular: bool,
        ) -> np.ndarray:
            del traj_flat, d, epsilon, angular
            return np.ones((t, t), dtype=np.uint8)

        monkeypatch.setattr(r_mod, "_dispatch", lambda _name: _wrong_backend)

        with pytest.raises(ValueError, match="exact recurrence threshold"):
            recurrence_matrix(traj, 0.5)

    def test_public_cross_recurrence_rejects_wrong_binary_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        traj_a = np.array([0.0, 2.0], dtype=np.float64)
        traj_b = np.array([0.0, 3.0], dtype=np.float64)

        def _wrong_backend(
            traj_a_flat: np.ndarray,
            traj_b_flat: np.ndarray,
            t: int,
            d: int,
            epsilon: float,
            angular: bool,
        ) -> np.ndarray:
            del traj_a_flat, traj_b_flat, t, d, epsilon, angular
            return np.array([1, 1, 0, 1], dtype=np.uint8)

        monkeypatch.setattr(r_mod, "_dispatch", lambda _name: _wrong_backend)

        with pytest.raises(ValueError, match="exact recurrence threshold"):
            cross_recurrence_matrix(traj_a, traj_b, 0.5)


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
    def test_rm_euclidean(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        ref = _reference_rm(traj, 0.8)
        prev = _force("rust")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_rm_angular(self) -> None:
        traj = _trajectory(3, t=20, d=2)
        ref = _reference_rm(traj, 0.5, metric="angular")
        prev = _force("rust")
        try:
            got = recurrence_matrix(traj, 0.5, metric="angular")
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_cross(self) -> None:
        a = _trajectory(7, t=25)
        b = _trajectory(11, t=25)
        ref = _reference_cross(a, b, 1.0)
        prev = _force("rust")
        try:
            got = cross_recurrence_matrix(a, b, 1.0)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_rm(self, seed: int) -> None:
        traj = _trajectory(seed)
        ref = _reference_rm(traj, 0.8)
        prev = _force("julia")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_angular(self) -> None:
        traj = _trajectory(3, t=20, d=2)
        ref = _reference_rm(traj, 0.5, metric="angular")
        prev = _force("julia")
        try:
            got = recurrence_matrix(traj, 0.5, metric="angular")
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        t=st.integers(min_value=5, max_value=35),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rm_euclidean(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        ref = _reference_rm(traj, 0.8)
        prev = _force("go")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_cross(self) -> None:
        a = _trajectory(2, t=20)
        b = _trajectory(3, t=20)
        ref = _reference_cross(a, b, 1.0)
        prev = _force("go")
        try:
            got = cross_recurrence_matrix(a, b, 1.0)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_rm(self, seed: int) -> None:
        traj = _trajectory(seed, t=15)
        ref = _reference_rm(traj, 0.8)
        prev = _force("mojo")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree_rm(self) -> None:
        traj = _trajectory(2026, t=30)
        ref = _reference_rm(traj, 1.0)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = recurrence_matrix(traj, 1.0)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} RM diverged from python",
            )

    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree_cross(self) -> None:
        a = _trajectory(2026, t=24)
        b = _trajectory(1337, t=24)
        ref = _reference_cross(a, b, 1.0)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = cross_recurrence_matrix(a, b, 1.0)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} CROSS diverged from python",
            )
