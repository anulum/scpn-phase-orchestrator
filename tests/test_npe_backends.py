# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for NPE

"""Per-backend parity tests for ``monitor/npe.py``."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _npe_validation as npe_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._npe_go import (
    compute_npe_go,
    phase_distance_matrix_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._npe_julia import (
    compute_npe_julia,
    phase_distance_matrix_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._npe_mojo import (
    _run as run_npe_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._npe_mojo import (
    compute_npe_mojo,
    phase_distance_matrix_mojo,
)
from scpn_phase_orchestrator.monitor import npe as npe_mod
from scpn_phase_orchestrator.monitor.npe import (
    AVAILABLE_BACKENDS,
    compute_npe,
    phase_distance_matrix,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi
PdmBackend = Callable[[np.ndarray], np.ndarray]
NpeBackend = Callable[[np.ndarray, object], float]


def test__npe_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(npe_validation.expected_npe_backend_output)
    assert callable(npe_validation.expected_phase_distance_backend_output)
    assert callable(npe_validation.validate_phase_distance_backend_input)
    assert callable(npe_validation.validate_npe_backend_inputs)


def _force(backend: str) -> str:
    prev = npe_mod.ACTIVE_BACKEND
    npe_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    npe_mod.ACTIVE_BACKEND = prev


def _reference(
    phases: np.ndarray,
) -> tuple[np.ndarray, float]:
    prev = _force("python")
    try:
        pdm = phase_distance_matrix(phases)
        npe = compute_npe(phases)
    finally:
        _reset(prev)
    return pdm, npe


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        phase_distance_matrix_go,
        phase_distance_matrix_julia,
        phase_distance_matrix_mojo,
        compute_npe_go,
        compute_npe_julia,
        compute_npe_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        assert_precise_ndarray_hint(hints["phases"])
        assert "float64" in str(hints["phases"])
        if fn.__name__.startswith("phase_distance_matrix"):
            assert_precise_ndarray_hint(hints["return"])
            assert "float64" in str(hints["return"])


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 1, "NPE", "exactly 1 scalar"),
            ("0.1\n0.2\n", 1, "NPE", "exactly 1 scalar"),
            ("0.0\n\n0.5\n0.5\n0.0\n", 4, "PDM", "exactly 4 scalar"),
            ("not-a-number\n", 1, "NPE", "non-scalar"),
        ],
    )
    def test_mojo_runner_rejects_malformed_npe_stdout(
        self,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_npe_mojo._ensure_exe",
            lambda: "npe_mojo",
        )
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_npe_mojo.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            run_npe_mojo(
                "NPE 2 3.141592653589793 0 1\n",
                expected_count=expected_count,
                label=label,
            )

    def test_phase_distance_backend_output_accepts_flat_or_matrix_contract(
        self,
    ) -> None:
        distances = np.array(
            [
                0.0,
                0.25,
                0.5,
                0.25,
                0.0,
                0.75,
                0.5,
                0.75,
                0.0,
            ]
        )

        matrix = npe_validation.validate_phase_distance_backend_output(
            distances,
            n_phases=3,
        )

        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix, matrix.T, atol=0.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=0.0)

    @pytest.mark.parametrize(
        ("distances", "message"),
        [
            (np.array([0.0, 1.0, 1.0]), "size"),
            (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
            (np.array([[0.0, np.pi + 1.0], [np.pi + 1.0, 0.0]]), r"\[0, pi\]"),
            (np.array([[0.0, 0.25], [0.5, 0.0]]), "symmetric"),
            (np.array([[0.1, 0.25], [0.25, 0.0]]), "diagonal"),
            (np.array([[False, 0.25], [0.25, False]], dtype=object), "booleans"),
            (np.array([[0.0, 0.25j], [0.25j, 0.0]]), "real values"),
            (np.array([[0.0, 0.25j], [0.25j, 0.0]], dtype=object), "real values"),
        ],
    )
    def test_phase_distance_backend_output_rejects_invalid_physics(
        self,
        distances: np.ndarray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            npe_validation.validate_phase_distance_backend_output(
                distances,
                n_phases=2,
            )

    def test_phase_distance_backend_output_rejects_exact_distance_divergence(
        self,
    ) -> None:
        phases = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
        expected = npe_validation.expected_phase_distance_backend_output(phases)
        bounded_symmetric_wrong = np.array(
            [
                [0.0, 0.25, 0.5],
                [0.25, 0.0, 0.25],
                [0.5, 0.25, 0.0],
            ],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="exact circular phase distances"):
            npe_validation.validate_phase_distance_backend_output(
                bounded_symmetric_wrong,
                n_phases=3,
                expected=expected,
            )

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, 1.0 + 5.0e-13])
    def test_npe_backend_output_accepts_unit_interval_scalars(
        self,
        score: float,
    ) -> None:
        assert 0.0 <= npe_validation.validate_npe_backend_output(score) <= 1.0

    @pytest.mark.parametrize(
        "score",
        [np.bool_(False), np.nan, -1.0e-3, 1.0 + 1.0e-3, 0.5 + 0.0j],
    )
    def test_npe_backend_output_rejects_invalid_scalars(self, score: object) -> None:
        with pytest.raises(ValueError, match="NPE backend output"):
            npe_validation.validate_npe_backend_output(score)

    def test_npe_backend_output_rejects_exact_entropy_divergence(self) -> None:
        phases = np.array([0.0, 0.1, 2.8, 3.0], dtype=np.float64)
        expected = npe_validation.expected_npe_backend_output(phases, np.pi)
        wrong_score = 0.0 if expected > 0.1 else 0.75

        with pytest.raises(ValueError, match="exact NPE"):
            npe_validation.validate_npe_backend_output(
                wrong_score,
                expected=expected,
            )

    @pytest.mark.parametrize(
        "fn",
        [
            phase_distance_matrix_go,
            phase_distance_matrix_julia,
            phase_distance_matrix_mojo,
        ],
    )
    @pytest.mark.parametrize(
        "phases",
        [
            np.array([0.0, True], dtype=object),
            np.array([0.0 + 1.0j, 1.0 + 0.0j]),
            np.array([0.0, 1.0j], dtype=object),
            np.array([0.0, np.inf]),
            np.array([[0.0, 1.0]]),
        ],
    )
    def test_phase_distance_backend_rejects_invalid_phases_before_runtime_load(
        self,
        fn: PdmBackend,
        phases: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="phases"):
            fn(phases)

    @pytest.mark.parametrize(
        "fn",
        [
            compute_npe_go,
            compute_npe_julia,
            compute_npe_mojo,
        ],
    )
    @pytest.mark.parametrize(
        "phases",
        [
            np.array([0.0, np.bool_(False)], dtype=object),
            np.array([0.0, 1.0j]),
            np.array([0.0, 1.0j], dtype=object),
            np.array([0.0, np.nan]),
            np.array([[0.0, 1.0]]),
        ],
    )
    def test_npe_backend_rejects_invalid_phases_before_runtime_load(
        self,
        fn: NpeBackend,
        phases: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="phases"):
            fn(phases, np.pi)

    @pytest.mark.parametrize(
        "fn",
        [
            compute_npe_go,
            compute_npe_julia,
            compute_npe_mojo,
        ],
    )
    @pytest.mark.parametrize(
        "max_radius",
        [
            np.bool_(True),
            np.nan,
            -1.0,
            np.pi + 1.0e-6,
            1.0 + 0.0j,
        ],
    )
    def test_npe_backend_rejects_invalid_radius_before_runtime_load(
        self,
        fn: NpeBackend,
        max_radius: object,
    ) -> None:
        phases = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="max_radius"):
            fn(phases, max_radius)

    def test_julia_backend_rejects_exact_phase_distance_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def phase_distance_matrix(phases: np.ndarray) -> np.ndarray:
                n = phases.size
                wrong = np.full((n, n), 0.25, dtype=np.float64)
                np.fill_diagonal(wrong, 0.0)
                return wrong

        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_npe_julia._ensure",
            lambda: _FakeJulia(),
        )

        with pytest.raises(ValueError, match="exact circular phase distances"):
            phase_distance_matrix_julia(np.array([0.0, np.pi / 2.0, np.pi]))

    def test_mojo_backend_rejects_exact_npe_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor._npe_mojo._run",
            lambda payload, *, expected_count, label: [0.0],
        )

        with pytest.raises(ValueError, match="exact NPE"):
            compute_npe_mojo(np.array([0.0, 0.1, 2.8, 3.0]), np.pi)

    def test_public_phase_distance_rejects_wrong_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float64)

        def _wrong_backend(phases_in: np.ndarray) -> np.ndarray:
            n = phases_in.size
            wrong = np.full((n, n), 0.25, dtype=np.float64)
            np.fill_diagonal(wrong, 0.0)
            return wrong

        monkeypatch.setattr(npe_mod, "_dispatch", lambda _name: _wrong_backend)

        with pytest.raises(ValueError, match="exact circular phase distances"):
            phase_distance_matrix(phases)

    def test_public_compute_npe_rejects_wrong_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases = np.array([0.0, 0.1, 2.8, 3.0], dtype=np.float64)
        _, ref_npe = _reference(phases)

        def _wrong_backend(phases_in: np.ndarray, radius: float) -> float:
            del phases_in, radius
            return 0.0 if ref_npe > 0.1 else 0.75

        monkeypatch.setattr(npe_mod, "_dispatch", lambda _name: _wrong_backend)

        with pytest.raises(ValueError, match="exact persistent-entropy reference"):
            compute_npe(phases)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=4, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=12, deadline=None)
    def test_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("rust")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [8, 24, 64])
    def test_parity(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("julia")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=4, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("go")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [6, 12, 24])
    def test_parity(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("mojo")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        # text-protocol budget amplifies in log() across N lifetimes
        assert abs(npe - ref_npe) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 20
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            atol = tolerances[backend]
            prev = _force(backend)
            try:
                pdm = phase_distance_matrix(phases)
                npe = compute_npe(phases)
            finally:
                _reset(prev)
            np.testing.assert_allclose(pdm, ref_pdm, atol=atol)
            assert abs(npe - ref_npe) <= atol
