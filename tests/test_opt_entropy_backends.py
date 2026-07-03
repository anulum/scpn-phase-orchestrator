# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for OPT-entropy

"""Per-backend parity tests for ``monitor/opt_entropy.py``."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _opt_entropy_go,
    _opt_entropy_julia,
    _opt_entropy_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _opt_entropy_validation as oe_validation,
)
from scpn_phase_orchestrator.monitor import opt_entropy as oe_mod
from scpn_phase_orchestrator.monitor.opt_entropy import (
    AVAILABLE_BACKENDS,
    ordinal_pattern_sequence,
    transition_entropy,
)
from tests.typing_contracts import assert_precise_ndarray_hint

ordinal_pattern_sequence_go = _opt_entropy_go.ordinal_pattern_sequence_go
transition_entropy_go = _opt_entropy_go.transition_entropy_go
ordinal_pattern_sequence_julia = _opt_entropy_julia.ordinal_pattern_sequence_julia
transition_entropy_julia = _opt_entropy_julia.transition_entropy_julia
ordinal_pattern_sequence_mojo = _opt_entropy_mojo.ordinal_pattern_sequence_mojo
transition_entropy_mojo = _opt_entropy_mojo.transition_entropy_mojo
run_opt_entropy_mojo = _opt_entropy_mojo._run

OpsBackend = Callable[[np.ndarray, int, int], np.ndarray]
TeBackend = Callable[[np.ndarray, int, int], float]


def test__opt_entropy_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(oe_validation.expected_ordinal_pattern_backend_output)
    assert callable(oe_validation.expected_transition_entropy_backend_output)
    assert callable(oe_validation.validate_series_backend_input)
    assert callable(oe_validation.validate_transition_entropy_backend_inputs)


def _force(backend: str) -> str:
    prev = oe_mod.ACTIVE_BACKEND
    oe_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    oe_mod.ACTIVE_BACKEND = prev


def _reference(
    series: np.ndarray, dimension: int, delay: int
) -> tuple[np.ndarray, float]:
    prev = _force("python")
    try:
        codes = ordinal_pattern_sequence(series, dimension, delay)
        value = transition_entropy(series, dimension, delay)
    finally:
        _reset(prev)
    return codes, value


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        ordinal_pattern_sequence_go,
        ordinal_pattern_sequence_julia,
        ordinal_pattern_sequence_mojo,
        transition_entropy_go,
        transition_entropy_julia,
        transition_entropy_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        assert_precise_ndarray_hint(hints["series"])
        assert "float64" in str(hints["series"])
        if fn.__name__.startswith("ordinal_pattern_sequence"):
            assert_precise_ndarray_hint(hints["return"])
            assert "int64" in str(hints["return"])


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 1, "OTE", "exactly 1 scalar"),
            ("0.1\n0.2\n", 1, "OTE", "exactly 1 scalar"),
            ("0\n\n1\n", 4, "OPS", "exactly 4 scalar"),
            ("not-a-number\n", 1, "OTE", "non-scalar"),
        ],
    )
    def test_mojo_runner_rejects_malformed_stdout(
        self,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_opt_entropy_mojo._ensure_exe",
            lambda: "opt_entropy_mojo",
        )
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_opt_entropy_mojo.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )
        with pytest.raises(ValueError, match=match):
            run_opt_entropy_mojo(
                "OTE 6 3 1 0 1 0.5 2 1.5 3\n",
                expected_count=expected_count,
                label=label,
            )

    def test_ordinal_backend_output_accepts_valid_codes(self) -> None:
        codes = oe_validation.validate_ordinal_pattern_backend_output(
            np.array([0, 5, 3], dtype=np.int64),
            n_windows=3,
            dimension=3,
        )
        assert codes.dtype == np.int64
        assert codes.tolist() == [0, 5, 3]

    @pytest.mark.parametrize(
        ("codes", "message"),
        [
            (np.array([0.0, 1.5], dtype=np.float64), "integer-valued"),
            (np.array([0, 1, 2], dtype=np.int64), "does not match"),
            (np.array([0, 99], dtype=np.int64), r"\[0, "),
            (np.array([np.nan, 1.0]), "finite"),
            (np.array([True, False]), "booleans"),
            (np.array([0.0 + 1.0j, 1.0]), "real values"),
            (np.array(["0", "1"]), "numeric-string"),
        ],
    )
    def test_ordinal_backend_output_rejects_invalid(
        self, codes: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            oe_validation.validate_ordinal_pattern_backend_output(
                codes, n_windows=2, dimension=3
            )

    def test_ordinal_backend_output_rejects_divergence(self) -> None:
        series = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float64)
        expected = oe_validation.expected_ordinal_pattern_backend_output(series, 3, 1)
        wrong = np.array([abs(int(expected[0]) - 4)] * expected.size, dtype=np.int64)
        with pytest.raises(ValueError, match="exact reference"):
            oe_validation.validate_ordinal_pattern_backend_output(
                wrong, n_windows=expected.size, dimension=3, expected=expected
            )

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, 1.0 + 5.0e-13])
    def test_transition_entropy_output_accepts_unit_interval(
        self, score: float
    ) -> None:
        assert (
            0.0
            <= oe_validation.validate_transition_entropy_backend_output(score)
            <= 1.0
        )

    @pytest.mark.parametrize(
        "score", [np.bool_(False), np.nan, -1.0e-3, 1.0 + 1.0e-3, 0.5 + 0.0j]
    )
    def test_transition_entropy_output_rejects_invalid(self, score: object) -> None:
        with pytest.raises(ValueError, match="transition entropy backend output"):
            oe_validation.validate_transition_entropy_backend_output(score)

    def test_transition_entropy_output_rejects_divergence(self) -> None:
        series = np.sin(np.linspace(0.0, 30.0, 200))
        expected = oe_validation.expected_transition_entropy_backend_output(
            series, 3, 1
        )
        wrong = 0.0 if expected > 0.1 else 0.9
        with pytest.raises(ValueError, match="exact reference"):
            oe_validation.validate_transition_entropy_backend_output(
                wrong, expected=expected
            )

    @pytest.mark.parametrize(
        "fn",
        [
            ordinal_pattern_sequence_go,
            ordinal_pattern_sequence_julia,
            ordinal_pattern_sequence_mojo,
            transition_entropy_go,
            transition_entropy_julia,
            transition_entropy_mojo,
        ],
    )
    @pytest.mark.parametrize(
        "series",
        [
            np.array([0.0, True], dtype=object),
            np.array([0.0 + 1.0j, 1.0 + 0.0j]),
            np.array([0.0, np.inf]),
            np.array([[0.0, 1.0]]),
            np.array(["0.0", "1.0", "0.5"]),
            np.array([0.0, "1.0", 0.5], dtype=object),
        ],
    )
    def test_backend_rejects_invalid_series_before_runtime_load(
        self, fn: OpsBackend | TeBackend, series: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="series"):
            fn(series, 3, 1)

    @pytest.mark.parametrize(
        "fn",
        [
            transition_entropy_go,
            transition_entropy_julia,
            transition_entropy_mojo,
        ],
    )
    @pytest.mark.parametrize("dimension", [1, 8, True, 3.5])
    def test_backend_rejects_invalid_dimension_before_runtime_load(
        self, fn: TeBackend, dimension: object
    ) -> None:
        with pytest.raises(ValueError, match="dimension"):
            fn(np.arange(20, dtype=np.float64), dimension, 1)

    def test_julia_backend_rejects_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def transition_entropy(series: object, dimension: int, delay: int) -> float:
                del series, dimension, delay
                return 0.0

        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_opt_entropy_julia._ensure",
            lambda: _FakeJulia(),
        )
        with pytest.raises(ValueError, match="exact reference"):
            transition_entropy_julia(np.sin(np.linspace(0.0, 30.0, 200)), 3, 1)

    def test_mojo_backend_rejects_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_opt_entropy_mojo._run",
            lambda payload, *, expected_count, label: [0.0],
        )
        with pytest.raises(ValueError, match="exact reference"):
            transition_entropy_mojo(np.sin(np.linspace(0.0, 30.0, 200)), 3, 1)

    def test_public_transition_entropy_rejects_wrong_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = np.sin(np.linspace(0.0, 30.0, 200))
        _, ref = _reference(series, 3, 1)

        def _wrong(_s: np.ndarray, _d: int, _t: int) -> float:
            return 0.0 if ref > 0.1 else 0.9

        monkeypatch.setattr(oe_mod, "_dispatch", lambda _name: _wrong)
        with pytest.raises(ValueError, match="exact reference"):
            transition_entropy(series, 3, 1)


class TestGoBridgeErrorPaths:
    def test_missing_library_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pathlib

        monkeypatch.setattr(_opt_entropy_go, "_LIB", None)
        monkeypatch.setattr(
            _opt_entropy_go, "_LIB_PATH", pathlib.Path("/nonexistent/libopt.so")
        )
        with pytest.raises(ImportError, match="libopt_entropy.so not found"):
            _opt_entropy_go._load_lib()

    def test_nonzero_return_code_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeLib:
            @staticmethod
            def TransitionEntropy(*_args: object) -> int:
                return 7

            @staticmethod
            def OrdinalPatternSequence(*_args: object) -> int:
                return 9

        monkeypatch.setattr(_opt_entropy_go, "_load_lib", lambda: _FakeLib())
        with pytest.raises(ValueError, match="rc=7"):
            transition_entropy_go(np.arange(20, dtype=np.float64), 3, 1)
        with pytest.raises(ValueError, match="rc=9"):
            ordinal_pattern_sequence_go(np.arange(20, dtype=np.float64), 3, 1)


class TestMojoBridgeEdgeCases:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    def test_too_short_series_returns_empty(self) -> None:
        codes = ordinal_pattern_sequence_mojo(np.array([1.0, 2.0]), 3, 1)
        assert codes.shape == (0,)
        assert codes.dtype == np.int64


class TestMojoBridgeColdPaths:
    def test_missing_executable_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pathlib

        monkeypatch.setattr(
            _opt_entropy_mojo,
            "_EXE_PATH",
            pathlib.Path("/nonexistent/opt_entropy_mojo"),
        )
        with pytest.raises(ImportError, match="not built"):
            _opt_entropy_mojo._ensure_exe()

    def test_nonzero_exit_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _opt_entropy_mojo, "_ensure_exe", lambda: "opt_entropy_mojo"
        )
        monkeypatch.setattr(
            _opt_entropy_mojo.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr="boom"),
        )
        with pytest.raises(ValueError, match="exit 2"):
            run_opt_entropy_mojo(
                "OTE 6 3 1 0 1 2 3 4 5\n", expected_count=1, label="OTE"
            )


class TestJuliaBridgeColdPaths:
    def test_missing_side_file_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a stub ``juliacall`` so the file-existence branch is exercised
        # without booting a real Julia runtime (which cannot initialise under
        # the coverage tracer).
        import pathlib
        import sys
        import types

        fake_juliacall = types.ModuleType("juliacall")
        fake_juliacall.Main = object()
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setattr(_opt_entropy_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(
            _opt_entropy_julia,
            "_JULIA_FILE",
            pathlib.Path("/nonexistent/opt_entropy.jl"),
        )
        with pytest.raises(ImportError, match="julia side-file not found"):
            _opt_entropy_julia._ensure()


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=8, max_value=600),
        dimension=st.integers(min_value=2, max_value=6),
        delay=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_parity(self, n: int, dimension: int, delay: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        series = rng.standard_normal(n)
        ref_codes, ref_te = _reference(series, dimension, delay)
        prev = _force("rust")
        try:
            codes = ordinal_pattern_sequence(series, dimension, delay)
            te = transition_entropy(series, dimension, delay)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(codes, ref_codes)
        assert abs(te - ref_te) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("dimension", [2, 3, 4])
    @pytest.mark.parametrize("n", [40, 200])
    def test_parity(self, n: int, dimension: int) -> None:
        rng = np.random.default_rng(7 + n + dimension)
        series = rng.standard_normal(n)
        ref_codes, ref_te = _reference(series, dimension, 1)
        prev = _force("julia")
        try:
            codes = ordinal_pattern_sequence(series, dimension, 1)
            te = transition_entropy(series, dimension, 1)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(codes, ref_codes)
        assert abs(te - ref_te) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=8, max_value=400),
        dimension=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_parity(self, n: int, dimension: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        series = rng.standard_normal(n)
        ref_codes, ref_te = _reference(series, dimension, 1)
        prev = _force("go")
        try:
            codes = ordinal_pattern_sequence(series, dimension, 1)
            te = transition_entropy(series, dimension, 1)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(codes, ref_codes)
        assert abs(te - ref_te) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("dimension", [2, 3, 4])
    @pytest.mark.parametrize("n", [20, 60])
    def test_parity(self, n: int, dimension: int) -> None:
        rng = np.random.default_rng(17 + n + dimension)
        series = rng.standard_normal(n)
        ref_codes, ref_te = _reference(series, dimension, 1)
        prev = _force("mojo")
        try:
            codes = ordinal_pattern_sequence(series, dimension, 1)
            te = transition_entropy(series, dimension, 1)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(codes, ref_codes)
        # text-protocol round-trip amplifies the entropy log() floor
        assert abs(te - ref_te) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        series = rng.standard_normal(120)
        ref_codes, ref_te = _reference(series, 3, 1)
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
                codes = ordinal_pattern_sequence(series, 3, 1)
                te = transition_entropy(series, 3, 1)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(codes, ref_codes)
            assert abs(te - ref_te) <= atol
