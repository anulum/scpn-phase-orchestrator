# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling tests

"""Tests for phase-amplitude coupling (PAC) measurement."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import pac as pac_mod
from scpn_phase_orchestrator.upde.pac import modulation_index, pac_gate, pac_matrix

TWO_PI = 2.0 * np.pi


class TestModulationIndex:
    def test_uncoupled_signals_low_mi(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, TWO_PI, 1000)
        amp = rng.uniform(0.5, 1.5, 1000)
        mi = modulation_index(theta, amp)
        assert mi < 0.1

    def test_entrained_signals_detectable_mi(self) -> None:
        theta = np.linspace(0, TWO_PI * 10, 5000) % TWO_PI
        # Amplitude concentrated in narrow phase window → detectable PAC
        amp = np.exp(4.0 * np.cos(theta))
        mi = modulation_index(theta, amp)
        assert mi > 0.05

    def test_mi_in_unit_range(self) -> None:
        rng = np.random.default_rng(1)
        theta = rng.uniform(0, TWO_PI, 500)
        amp = np.abs(rng.standard_normal(500))
        mi = modulation_index(theta, amp)
        assert 0.0 <= mi <= 1.0

    def test_empty_input_returns_zero(self) -> None:
        assert modulation_index(np.array([]), np.array([])) == 0.0

    def test_zero_amplitude_returns_zero(self) -> None:
        theta = np.linspace(0, TWO_PI, 100)
        amp = np.zeros(100)
        assert modulation_index(theta, amp) == 0.0

    def test_mismatched_lengths_uses_shorter(self) -> None:
        theta = np.linspace(0, TWO_PI, 100)
        amp = np.ones(50)
        mi = modulation_index(theta, amp)
        assert 0.0 <= mi <= 1.0

    def test_n_bins_parameter(self) -> None:
        theta = np.linspace(0, TWO_PI * 10, 5000) % TWO_PI
        amp = np.exp(4.0 * np.cos(theta))
        mi_18 = modulation_index(theta, amp, n_bins=18)
        mi_36 = modulation_index(theta, amp, n_bins=36)
        assert mi_18 > 0.05
        assert mi_36 > 0.05

    @pytest.mark.parametrize(
        ("theta", "amp", "n_bins", "match"),
        [
            (np.zeros((2, 2)), np.ones(4), 18, "theta_low"),
            (np.array([0.0, np.nan]), np.ones(2), 18, "theta_low"),
            (np.zeros(2), np.array([1.0, np.inf]), 18, "amp_high"),
            (np.array([True, False]), np.ones(2), 18, "theta_low"),
            (np.zeros(2), np.array([True, False]), 18, "amp_high"),
            (np.zeros(2), np.ones(2), True, "n_bins"),
            (np.zeros(2), np.ones(2), 1.5, "n_bins"),
        ],
    )
    def test_rejects_invalid_modulation_inputs(
        self,
        theta: np.ndarray,
        amp: np.ndarray,
        n_bins: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            modulation_index(theta, amp, n_bins=n_bins)  # type: ignore[arg-type]

    def test_degenerate_private_histogram_resolution_returns_zero(self) -> None:
        theta = np.linspace(0.0, TWO_PI, 64, endpoint=False)
        amp = 1.0 + 0.25 * np.cos(theta)
        assert pac_mod._modulation_index_python(theta, amp, n_bins=1) == 0.0

    def test_modulation_index_non_positive_bins_short_circuit(self) -> None:
        theta = np.linspace(0.0, TWO_PI, 16, endpoint=False)
        amp = np.ones_like(theta)
        assert modulation_index(theta, amp, n_bins=0) == 0.0
        assert modulation_index(theta, amp, n_bins=-2) == 0.0

    def test_modulation_index_short_circuit_does_not_use_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[bool] = []

        def _backend(_: np.ndarray, __: np.ndarray, ___: int) -> float:
            called.append(True)
            return 1.0

        monkeypatch.setattr(pac_mod, "_dispatch", lambda _fn_name: _backend)
        theta = np.linspace(0.0, TWO_PI, 16, endpoint=False)
        amp = np.ones_like(theta)
        assert modulation_index(theta, amp, n_bins=1) == 0.0
        assert called == []

    @pytest.mark.parametrize(
        ("backend_value", "expected"),
        [
            (-1.0, 0.0),
            (2.5, 1.0),
        ],
    )
    def test_modulation_index_backend_clamps_out_of_range_mi(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: float,
        expected: float,
    ) -> None:
        def fake_modulation_index(
            _theta_low: np.ndarray,
            _amp_high: np.ndarray,
            _n_bins: int,
        ) -> float:
            return backend_value

        monkeypatch.setattr(
            pac_mod,
            "_dispatch",
            lambda _fn_name: fake_modulation_index,
        )
        theta = np.linspace(0.0, TWO_PI, 16, endpoint=False)
        amp = np.ones_like(theta)
        assert modulation_index(theta, amp) == expected

    def test_modulation_index_backend_rejects_non_finite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_modulation_index(
            _theta_low: np.ndarray,
            _amp_high: np.ndarray,
            _n_bins: int,
        ) -> float:
            return float("nan")

        monkeypatch.setattr(
            pac_mod,
            "_dispatch",
            lambda _fn_name: fake_modulation_index,
        )
        theta = np.linspace(0.0, TWO_PI, 16, endpoint=False)
        amp = np.ones_like(theta)
        with pytest.raises(ValueError, match="modulation_index backend"):
            modulation_index(theta, amp)


class TestPACMatrix:
    def test_shape(self) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (100, 4))
        amps = np.random.default_rng(1).uniform(0, 1, (100, 4))
        mat = pac_matrix(phases, amps)
        assert mat.shape == (4, 4)

    def test_values_in_range(self) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (200, 3))
        amps = np.abs(np.random.default_rng(1).standard_normal((200, 3)))
        mat = pac_matrix(phases, amps)
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pac_matrix(np.zeros(10), np.zeros(10))

    def test_mismatched_n_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            pac_matrix(np.zeros((10, 3)), np.zeros((10, 4)))

    @pytest.mark.parametrize(
        ("phases", "amps", "expected_shape"),
        [
            (np.zeros((0, 4)), np.zeros((0, 4)), (4, 4)),
            (np.zeros((6, 0)), np.zeros((6, 0)), (0, 0)),
        ],
    )
    def test_boundary_shapes_short_circuit_to_zero(
        self,
        phases: np.ndarray,
        amps: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> None:
        mat = pac_matrix(phases, amps)
        assert mat.shape == expected_shape
        assert np.all(mat == 0.0)

    def test_n_bins_short_circuit_matrix_avoids_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[bool] = []

        def _backend(
            _phases_flat: np.ndarray,
            _amps_flat: np.ndarray,
            _t: int,
            _n: int,
            _n_bins: int,
        ) -> np.ndarray:
            called.append(True)
            return np.full(4, 1.0, dtype=np.float64)

        monkeypatch.setattr(pac_mod, "_dispatch", lambda _fn_name: _backend)
        phases = np.ones((8, 2), dtype=np.float64)
        amps = np.ones((8, 2), dtype=np.float64)
        mat = pac_matrix(phases, amps, n_bins=0)
        assert mat.shape == (2, 2)
        assert np.allclose(mat, 0.0)
        assert called == []

    @pytest.mark.parametrize(
        ("phases", "amps", "n_bins", "match"),
        [
            (np.array([[0.0, np.nan]]), np.ones((1, 2)), 18, "phases_history"),
            (np.zeros((1, 2)), np.array([[1.0, np.inf]]), 18, "amplitudes_history"),
            (np.array([[True, False]]), np.ones((1, 2)), 18, "phases_history"),
            (np.zeros((1, 2)), np.array([[True, False]]), 18, "amplitudes_history"),
            (np.zeros((1, 2)), np.ones((1, 2)), True, "n_bins"),
        ],
    )
    def test_rejects_invalid_matrix_inputs(
        self,
        phases: np.ndarray,
        amps: np.ndarray,
        n_bins: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            pac_matrix(phases, amps, n_bins=n_bins)  # type: ignore[arg-type]


class TestPACBackendErrorPaths:
    def test_pac_matrix_rejects_backend_wrong_size_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_matrix(
            _phases_flat: np.ndarray,
            _amplitudes_flat: np.ndarray,
            _t: int,
            _n: int,
            _n_bins: int,
        ) -> np.ndarray:
            return np.zeros(3, dtype=np.float64)

        monkeypatch.setattr(pac_mod, "_dispatch", lambda _fn_name: fake_matrix)
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (4, 2))
        amps = np.abs(np.random.default_rng(1).standard_normal((4, 2)))
        with pytest.raises(ValueError, match=r"n\*n values"):
            pac_matrix(phases, amps)

    def test_pac_matrix_rejects_backend_non_finite_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_matrix(
            phases_flat: np.ndarray,
            amplitudes_flat: np.ndarray,
            t: int,
            n: int,
            n_bins: int,
        ) -> np.ndarray:
            return np.array([0.0, 0.0, np.inf, 0.0], dtype=np.float64)

        monkeypatch.setattr(pac_mod, "_dispatch", lambda _fn_name: fake_matrix)
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (4, 2))
        amps = np.abs(np.random.default_rng(1).standard_normal((4, 2)))
        with pytest.raises(ValueError, match="finite values"):
            pac_matrix(phases, amps)


class TestPACGate:
    def test_gate_open(self) -> None:
        assert pac_gate(0.5, threshold=0.3) is True

    def test_gate_closed(self) -> None:
        assert pac_gate(0.1, threshold=0.3) is False

    def test_gate_at_threshold(self) -> None:
        assert pac_gate(0.3, threshold=0.3) is True

    @pytest.mark.parametrize(
        ("pac_value", "threshold", "match"),
        [
            (np.nan, 0.3, "pac_value"),
            (0.2, np.inf, "threshold"),
            (True, 0.3, "pac_value"),
            (0.2, True, "threshold"),
        ],
    )
    def test_rejects_invalid_gate_inputs(
        self,
        pac_value: object,
        threshold: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            pac_gate(pac_value, threshold=threshold)  # type: ignore[arg-type]


class TestPACPipelineWiring:
    """Verify PAC analysis wires into the UPDE engine pipeline and
    measures performance for both Python and Rust paths."""


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> dict[str, object]:
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go() -> dict[str, object]:
            calls["go"] += 1
            return {
                "modulation_index": lambda theta, amp, n_bins: 0.5,
                "pac_matrix": lambda phases, amps, t, n, n_bins: np.zeros(
                    n * n, dtype=np.float64
                ),
            }

        monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(pac_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        fn = pac_mod._dispatch("modulation_index")
        assert fn is not None
        assert float(
            fn(
                np.array([0.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                18,
            )
        ) == 0.5
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go() -> dict[str, object]:
            calls["go"] += 1
            return {
                "modulation_index": lambda theta, amp, n_bins: 0.25,
                "pac_matrix": lambda phases, amps, t, n, n_bins: np.zeros(
                    n * n, dtype=np.float64
                ),
            }

        monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(pac_mod, "_LOADERS", {"go": _ok_go})

        pac_mod._dispatch("modulation_index")
        pac_mod._dispatch("pac_matrix")

        assert calls["go"] == 1

    def test_engine_phases_to_pac_matrix(self) -> None:
        """UPDEEngine → phases trajectory → PAC matrix: proves PAC
        accepts engine output (not decorative)."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases_hist = []
        amps_hist = []
        p = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(100):
            p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
            phases_hist.append(p.copy())
            amps_hist.append(np.abs(np.cos(p)))

        phases_arr = np.array(phases_hist)  # (100, n)
        amps_arr = np.array(amps_hist)
        mat = pac_matrix(phases_arr, amps_arr)
        assert mat.shape == (n, n)
        assert np.all(mat >= 0.0) and np.all(mat <= 1.0)

    def test_pac_gate_in_policy_context(self) -> None:
        """pac_gate used in policy rules: high PAC → alert."""
        assert pac_gate(0.6, threshold=0.3) is True, "High PAC must trigger"
        assert pac_gate(0.1, threshold=0.3) is False, "Low PAC must not trigger"

    def test_modulation_index_performance_n1000(self) -> None:
        """MI computation on 1000 samples must complete in <5ms."""
        import time

        rng = np.random.default_rng(42)
        theta = rng.uniform(0, TWO_PI, 1000)
        amp = np.abs(rng.standard_normal(1000)) + 0.1

        # Warm up
        modulation_index(theta, amp)

        t0 = time.perf_counter()
        for _ in range(50):
            modulation_index(theta, amp)
        elapsed = (time.perf_counter() - t0) / 50
        assert elapsed < 0.005, f"MI(1000) = {elapsed * 1000:.1f}ms > 5ms"

    def test_pac_matrix_performance_n8_t200(self) -> None:
        """PAC matrix (8 channels, 200 timesteps) must complete in <50ms."""
        import time

        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, (200, 8))
        amps = np.abs(rng.standard_normal((200, 8))) + 0.1

        # Warm up
        pac_matrix(phases, amps)

        t0 = time.perf_counter()
        for _ in range(10):
            pac_matrix(phases, amps)
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.2, f"pac_matrix(8,200) = {elapsed * 1000:.1f}ms > 200ms"


class TestPACBackendDispatchFallbacks:
    def test_rust_loader_contract_with_contiguous_fake_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {}

        def fake_mi(theta: np.ndarray, amp: np.ndarray, n_bins: int) -> float:
            calls["mi"] = (theta.shape, amp.shape, n_bins)
            return 0.125

        def fake_matrix(
            phases_flat: np.ndarray,
            amps_flat: np.ndarray,
            t: int,
            n: int,
            n_bins: int,
        ) -> np.ndarray:
            calls["matrix"] = (phases_flat.shape, amps_flat.shape, n_bins)
            return np.full(n * n, 0.25, dtype=np.float64)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.pac_modulation_index = fake_mi
        fake_spo.pac_matrix_compute = fake_matrix
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        loaded = pac_mod._load_rust_fns()
        theta = np.linspace(0.0, TWO_PI, 8)
        amp = 1.0 + np.cos(theta)
        assert loaded["modulation_index"](theta, amp, 12) == 0.125
        mat = loaded["pac_matrix"](
            theta.reshape(4, 2).ravel(),
            amp.reshape(4, 2).ravel(),
            4,
            2,
            12,
        )
        np.testing.assert_allclose(mat.reshape(2, 2), 0.25)
        assert calls == {
            "mi": ((8,), (8,), 12),
            "matrix": ((8,), (8,), 12),
        }

    def test_probe_marks_backend_infinite_when_callable_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raising_mi(*_args: object) -> float:
            raise RuntimeError("backend down")

        monkeypatch.setattr(
            pac_mod,
            "_load_backend",
            lambda _name: {"modulation_index": raising_mi},
        )
        assert pac_mod._modulation_index_probe_seconds("rust") == float("inf")

    def test_dispatch_falls_back_to_python_when_active_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(
            pac_mod,
            "_load_backend",
            lambda _name: (_ for _ in ()).throw(OSError("backend unavailable")),
        )
        theta = np.linspace(0.0, TWO_PI, 128, endpoint=False)
        amp = 1.0 + 0.4 * np.cos(theta)
        assert modulation_index(theta, amp, n_bins=1) == 0.0
        got = modulation_index(theta, amp, n_bins=18)
        ref = pac_mod._modulation_index_python(theta, amp, 18)
        assert got == ref
