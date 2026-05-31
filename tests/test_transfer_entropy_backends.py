# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for transfer entropy

"""Per-backend parity tests for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _te_validation as te_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_go import (
    phase_te_go,
    te_matrix_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_julia import (
    phase_te_julia,
    te_matrix_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_mojo import (
    _run as run_te_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_mojo import (
    phase_te_mojo,
    te_matrix_mojo,
)
from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    AVAILABLE_BACKENDS,
    phase_transfer_entropy,
    transfer_entropy_matrix,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


def test__te_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(te_validation.validate_phase_te_backend_inputs)
    assert callable(te_validation.expected_phase_te_backend_output)
    assert callable(te_validation.validate_te_matrix_backend_inputs)
    assert callable(te_validation.expected_te_matrix_backend_output)


def _force(backend: str) -> str:
    prev = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    te_mod.ACTIVE_BACKEND = prev


def _reference_te(src: np.ndarray, tgt: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return phase_transfer_entropy(src, tgt, n_bins)
    finally:
        _reset(prev)


def _reference_matrix(series: np.ndarray, n_bins: int) -> np.ndarray:
    prev = _force("python")
    try:
        return transfer_entropy_matrix(series, n_bins)
    finally:
        _reset(prev)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        phase_te_go,
        te_matrix_go,
        phase_te_julia,
        te_matrix_julia,
        phase_te_mojo,
        te_matrix_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        checked_hints = [
            value
            for key, value in hints.items()
            if key in {"source", "target", "phase_series"}
        ]
        if fn.__name__.startswith("te_matrix"):
            checked_hints.append(hints["return"])
        for hint in checked_hints:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


def test_backend_resolution_records_first_available_backend(monkeypatch) -> None:
    calls: list[str] = []

    def unavailable(name: str):
        def _loader() -> dict[str, object]:
            calls.append(name)
            raise ImportError(name)

        return _loader

    def rust_loader() -> dict[str, object]:
        calls.append("rust")
        return {
            "phase_te": lambda _src, _tgt, _bins: 0.125,
            "te_matrix": lambda _flat, n_osc, _n_time, _bins: np.eye(n_osc).ravel(),
        }

    monkeypatch.setitem(te_mod._LOADERS, "rust", rust_loader)
    monkeypatch.setitem(te_mod._LOADERS, "mojo", unavailable("mojo"))
    monkeypatch.setitem(te_mod._LOADERS, "julia", unavailable("julia"))
    monkeypatch.setitem(te_mod._LOADERS, "go", unavailable("go"))

    active, available = te_mod._resolve_backends()

    assert active == "rust"
    assert available == ["rust", "python"]
    assert calls == ["rust", "mojo", "julia", "go"]


def test_rust_loader_exposes_phase_and_matrix_kernels(monkeypatch) -> None:
    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.phase_transfer_entropy_rust = lambda _src, _tgt, _bins: 0.25
    fake_spo.transfer_entropy_matrix_rust = lambda _flat, n_osc, _n_time, _bins: (
        np.zeros(n_osc * n_osc)
    )
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

    kernels = te_mod._load_rust_fns()

    assert kernels["phase_te"] is fake_spo.phase_transfer_entropy_rust
    assert kernels["te_matrix"] is fake_spo.transfer_entropy_matrix_rust


def test_dispatch_calls_active_backend_with_contiguous_arrays(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def phase_te(src: np.ndarray, tgt: np.ndarray, n_bins: int) -> float:
        captured["phase_src_contiguous"] = src.flags.c_contiguous
        captured["phase_tgt_contiguous"] = tgt.flags.c_contiguous
        captured["phase_bins"] = n_bins
        return _reference_te(src, tgt, n_bins)

    def te_matrix(flat: np.ndarray, n_osc: int, n_time: int, n_bins: int) -> np.ndarray:
        captured["matrix_flat_contiguous"] = flat.flags.c_contiguous
        captured["matrix_shape"] = (n_osc, n_time)
        captured["matrix_bins"] = n_bins
        return _reference_matrix(flat.reshape(n_osc, n_time), n_bins).ravel()

    kernels = {"phase_te": phase_te, "te_matrix": te_matrix}
    monkeypatch.setattr(te_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(te_mod._BACKEND_CACHE, "rust", kernels)
    monkeypatch.setitem(
        te_mod._LOADERS,
        "rust",
        lambda: kernels,
    )
    previous = _force("rust")
    try:
        src = np.arange(12, dtype=np.float64)[::2]
        tgt = np.arange(12, dtype=np.float64)[1::2]
        assert phase_transfer_entropy(src, tgt, n_bins=7) == _reference_te(
            src,
            tgt,
            7,
        )

        series = np.arange(30, dtype=np.float64).reshape(3, 10)[:, ::2]
        matrix = transfer_entropy_matrix(series, n_bins=5)
    finally:
        _reset(previous)

    assert captured == {
        "phase_src_contiguous": True,
        "phase_tgt_contiguous": True,
        "phase_bins": 7,
        "matrix_flat_contiguous": True,
        "matrix_shape": (3, 5),
        "matrix_bins": 5,
    }
    np.testing.assert_array_equal(matrix, _reference_matrix(series, 5))


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        try:
            rust_spec = importlib.util.find_spec("spo_kernel")
        except (ImportError, ValueError):
            rust_spec = None
        if rust_spec is None:
            pytest.skip("Rust backend not built")
        te_mod._BACKEND_CACHE.pop("rust", None)
        try:
            te_mod._load_rust_fns()
        except (ImportError, RuntimeError, OSError):
            pytest.skip("Rust transfer-entropy backend not built")

    @given(
        n=st.integers(min_value=50, max_value=400),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_phase_te(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = 0.5 * np.roll(src, -1) + 0.5 * rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
        prev = _force("rust")
        try:
            result = phase_transfer_entropy(src, tgt, 16)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_matrix(self) -> None:
        rng = np.random.default_rng(0)
        n_osc, n_time = 6, 200
        series = rng.uniform(0.0, TWO_PI, size=(n_osc, n_time))
        ref = _reference_matrix(series, 8)
        prev = _force("rust")
        try:
            result = transfer_entropy_matrix(series, 8)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [100, 400])
    def test_phase_te(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 12)
        prev = _force("julia")
        try:
            result = phase_transfer_entropy(src, tgt, 12)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=50, max_value=300),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_phase_te(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
        prev = _force("go")
        try:
            result = phase_transfer_entropy(src, tgt, 16)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [80, 200])
    def test_phase_te(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 12)
        prev = _force("mojo")
        try:
            result = phase_transfer_entropy(src, tgt, 12)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 250
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
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
                result = phase_transfer_entropy(src, tgt, 16)
            finally:
                _reset(prev)
            assert abs(result - ref) <= tolerances[backend]


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 1, "PTE", "exactly 1 scalar"),
            ("0.1\n0.2\n", 1, "PTE", "exactly 1 scalar"),
            ("0.0\n\n0.1\n0.0\n0.0\n", 4, "MAT", "exactly 4 scalar"),
            ("not-a-number\n", 1, "PTE", "non-scalar"),
        ],
    )
    def test_mojo_runner_rejects_malformed_transfer_entropy_stdout(
        self,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_te_mojo._ensure_exe",
            lambda: "transfer_entropy_mojo",
        )
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_te_mojo.subprocess.run",
            lambda *args, **kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            run_te_mojo(
                "PTE 3 4 0 1 2 1 2 3\n",
                expected_count=expected_count,
                label=label,
            )

    @pytest.mark.parametrize("value", [0.0, 0.25, np.log(4) + 5.0e-13])
    def test_pairwise_te_backend_output_accepts_entropy_bounds(
        self,
        value: float,
    ) -> None:
        assert te_validation.validate_te_backend_output(value, n_bins=4) >= 0.0

    @pytest.mark.parametrize(
        "value",
        [
            np.bool_(False),
            -1.0e-3,
            np.inf,
            np.log(4) + 1.0e-3,
            0.1 + 0.0j,
            np.array([0.1]),
        ],
    )
    def test_pairwise_te_backend_output_rejects_invalid_scalars(
        self,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="transfer entropy backend output"):
            te_validation.validate_te_backend_output(value, n_bins=4)

    def test_pairwise_te_backend_output_rejects_exact_estimator_divergence(
        self,
    ) -> None:
        source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
        target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
        expected = te_validation.expected_phase_te_backend_output(source, target, 2)
        assert expected > 0.0

        with pytest.raises(ValueError, match="exact transfer-entropy reference"):
            te_validation.validate_te_backend_output(
                0.0,
                n_bins=2,
                expected=expected,
            )

    def test_te_matrix_backend_output_accepts_directed_matrix(self) -> None:
        matrix = te_validation.validate_te_matrix_backend_output(
            np.array(
                [
                    0.0,
                    0.25,
                    0.5,
                    0.125,
                    0.0,
                    0.375,
                    0.2,
                    0.3,
                    0.0,
                ],
            ),
            n_osc=3,
            n_bins=4,
        )

        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=0.0)

    @pytest.mark.parametrize(
        ("matrix", "match"),
        [
            (np.array([0.0, 0.1, 0.2]), "size"),
            (np.array([[0.0, np.nan], [0.1, 0.0]]), "finite"),
            (np.array([[0.0, -0.1], [0.1, 0.0]]), "non-negative"),
            (np.array([[0.0, np.log(4) + 1.0], [0.1, 0.0]]), "log"),
            (np.array([[0.1, 0.0], [0.0, 0.0]]), "diagonal"),
            (np.array([[False, 0.1], [0.2, False]], dtype=object), "boolean"),
            (np.array([[0.0, 0.1j], [0.2j, 0.0]]), "real"),
        ],
    )
    def test_te_matrix_backend_output_rejects_invalid_payloads(
        self,
        matrix: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            te_validation.validate_te_matrix_backend_output(
                matrix,
                n_osc=2,
                n_bins=4,
            )

    def test_te_matrix_backend_output_rejects_exact_estimator_divergence(
        self,
    ) -> None:
        source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
        target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
        series = np.vstack([source, target])
        expected = te_validation.expected_te_matrix_backend_output(
            series.ravel(),
            2,
            source.size,
            2,
        )
        wrong = np.zeros_like(expected)
        assert expected[0, 1] > 0.0

        with pytest.raises(ValueError, match="exact transfer-entropy reference"):
            te_validation.validate_te_matrix_backend_output(
                wrong,
                n_osc=2,
                n_bins=2,
                expected=expected,
            )

    @pytest.mark.parametrize("backend", ["go", "julia", "mojo"])
    def test_direct_pairwise_backends_reject_exact_estimator_divergence(
        self,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
        target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
        wrong = 0.0

        if backend == "go":
            from scpn_phase_orchestrator.experimental.accelerators.monitor import (
                _te_go as te_go,
            )

            class _FakeGo:
                @staticmethod
                def PhaseTransferEntropy(
                    _source_ptr: object,
                    _target_ptr: object,
                    _n: object,
                    _n_bins: object,
                    out_ptr: object,
                ) -> int:
                    out_ptr._obj.value = wrong
                    return 0

            monkeypatch.setattr(te_go, "_load_lib", lambda: _FakeGo())
            target_fn = phase_te_go
        elif backend == "julia":

            class _FakeJulia:
                @staticmethod
                def phase_transfer_entropy(
                    _source: np.ndarray,
                    _target: np.ndarray,
                    _n_bins: int,
                ) -> float:
                    return wrong

            monkeypatch.setattr(
                "scpn_phase_orchestrator.experimental.accelerators.monitor."
                "_te_julia._ensure",
                lambda: _FakeJulia(),
            )
            target_fn = phase_te_julia
        else:
            monkeypatch.setattr(
                "scpn_phase_orchestrator.experimental.accelerators.monitor."
                "_te_mojo._run",
                lambda payload, *, expected_count, label: [wrong],
            )
            target_fn = phase_te_mojo

        with pytest.raises(ValueError, match="exact transfer-entropy reference"):
            target_fn(source, target, 2)

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (phase_te_go, "go"),
            (phase_te_julia, "julia"),
            (phase_te_mojo, "mojo"),
        ],
    )
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("source", np.array([0.0, np.bool_(True)], dtype=object), "source"),
            ("target", np.array([0.0, np.inf], dtype=np.float64), "target"),
            ("source", np.array([0.0 + 0.0j, 1.0 + 0.0j]), "source"),
            ("n_bins", np.bool_(True), "n_bins"),
            ("n_bins", 1, "n_bins"),
        ],
    )
    def test_phase_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn,
        label: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "source": np.array([0.0, 0.1, 0.2], dtype=np.float64),
            "target": np.array([0.2, 0.3, 0.4], dtype=np.float64),
            "n_bins": 4,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=match):
            fn(**kwargs)

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (te_matrix_go, "go"),
            (te_matrix_julia, "julia"),
            (te_matrix_mojo, "mojo"),
        ],
    )
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            (
                "phase_series",
                np.array([0.0, np.bool_(True)], dtype=object),
                "phase_series",
            ),
            ("phase_series", np.array([0.0 + 0.0j, 1.0 + 0.0j]), "phase_series"),
            ("phase_series", np.array([0.0, np.nan], dtype=np.float64), "phase_series"),
            ("n_osc", np.bool_(True), "n_osc"),
            ("n_time", np.bool_(True), "n_time"),
            ("n_bins", np.bool_(True), "n_bins"),
            ("n_bins", 1, "n_bins"),
        ],
    )
    def test_matrix_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn,
        label: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "phase_series": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            "n_osc": 2,
            "n_time": 2,
            "n_bins": 4,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=match):
            fn(**kwargs)


def test_public_transfer_entropy_falls_back_on_exact_contract_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
    target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])

    def fake_phase_te(_src: np.ndarray, _tgt: np.ndarray, _n_bins: int) -> float:
        return 0.0

    kernels = {"phase_te": fake_phase_te, "te_matrix": lambda *_args: np.zeros(4)}
    monkeypatch.setattr(te_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(te_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(te_mod._BACKEND_CACHE, "go", kernels)
    monkeypatch.setitem(te_mod._LOADERS, "go", lambda: kernels)

    assert phase_transfer_entropy(source, target, 2) == _reference_te(
        source,
        target,
        2,
    )


def test_public_transfer_entropy_matrix_falls_back_on_exact_contract_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
    target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
    series = np.vstack([source, target])

    def fake_matrix(
        _flat: np.ndarray,
        _n_osc: int,
        _n_time: int,
        _n_bins: int,
    ) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)

    kernels = {"phase_te": lambda *_args: 0.0, "te_matrix": fake_matrix}
    monkeypatch.setattr(te_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(te_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(te_mod._BACKEND_CACHE, "go", kernels)
    monkeypatch.setitem(te_mod._LOADERS, "go", lambda: kernels)

    np.testing.assert_allclose(
        transfer_entropy_matrix(series, 2),
        _reference_matrix(series, 2),
        atol=0.0,
    )
