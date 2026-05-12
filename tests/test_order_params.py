# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithm tests for order parameters

"""Algorithm-level tests for ``upde/order_params.py``.

Covers the three compute kernels — ``compute_order_parameter``,
``compute_plv``, ``compute_layer_coherence`` — plus dispatcher
invariants. Per-backend parity lives in
``test_order_params_backends.py``; stability tests live in
``test_order_params_stability.py``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scpn_phase_orchestrator.upde.order_params import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi

phase_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)


# ---------------------------------------------------------------------
# compute_order_parameter
# ---------------------------------------------------------------------


class TestOrderParameter:
    @given(phases=phase_arrays)
    @settings(max_examples=30, deadline=None)
    def test_r_bounded_unit_interval(self, phases: np.ndarray) -> None:
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0 + 1e-12

    @given(phases=phase_arrays)
    @settings(max_examples=30, deadline=None)
    def test_psi_in_zero_2pi(self, phases: np.ndarray) -> None:
        _, psi = compute_order_parameter(phases)
        assert 0.0 <= psi < TWO_PI + 1e-12

    def test_empty_returns_zero_zero(self) -> None:
        r, psi = compute_order_parameter(np.array([], dtype=np.float64))
        assert r == 0.0 and psi == 0.0

    def test_full_synchrony_r_one(self) -> None:
        phases = np.full(32, 1.234)
        r, _ = compute_order_parameter(phases)
        assert r == pytest.approx(1.0, abs=1e-12)

    def test_antiphase_pairs_r_zero(self) -> None:
        phases = np.concatenate([np.zeros(8), np.full(8, np.pi)])
        r, _ = compute_order_parameter(phases)
        assert r == pytest.approx(0.0, abs=1e-12)

    def test_uniform_distribution_r_small(self) -> None:
        phases = np.linspace(0.0, TWO_PI, 100, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert r < 1e-10

    def test_single_oscillator_r_one(self) -> None:
        r, psi = compute_order_parameter(np.array([0.7]))
        assert r == pytest.approx(1.0, abs=1e-12)
        assert psi == pytest.approx(0.7, abs=1e-12)


# ---------------------------------------------------------------------
# compute_plv
# ---------------------------------------------------------------------


class TestPLV:
    @given(a=phase_arrays, b=phase_arrays)
    @settings(max_examples=20, deadline=None)
    def test_plv_bounded_unit_interval(self, a: np.ndarray, b: np.ndarray) -> None:
        n = min(a.size, b.size)
        val = compute_plv(a[:n], b[:n])
        assert 0.0 <= val <= 1.0 + 1e-12

    def test_empty_returns_zero(self) -> None:
        assert compute_plv(np.array([]), np.array([])) == 0.0

    def test_identical_series_plv_one(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0.0, TWO_PI, size=50)
        assert compute_plv(phases, phases) == pytest.approx(1.0, abs=1e-12)

    def test_constant_offset_plv_one(self) -> None:
        rng = np.random.default_rng(1)
        phases = rng.uniform(0.0, TWO_PI, size=50)
        offset = phases + 0.7
        assert compute_plv(phases, offset) == pytest.approx(1.0, abs=1e-12)

    def test_uncorrelated_plv_small(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.uniform(0.0, TWO_PI, size=2000)
        b = rng.uniform(0.0, TWO_PI, size=2000)
        assert compute_plv(a, b) < 0.08

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------
# compute_layer_coherence
# ---------------------------------------------------------------------


class TestLayerCoherence:
    def test_full_synchrony_of_subset(self) -> None:
        phases = np.array([0.0, 0.1, np.pi, np.pi + 0.1, 2.0, 3.0], dtype=np.float64)
        r = compute_layer_coherence(phases, np.array([0, 1], dtype=np.int64))
        assert r > 0.99

    def test_empty_mask_returns_zero(self) -> None:
        phases = np.array([0.0, 1.0, 2.0])
        assert compute_layer_coherence(phases, np.array([], dtype=np.int64)) == 0.0

    def test_bool_mask_supported(self) -> None:
        phases = np.array([0.0, np.pi, 0.05, np.pi + 0.05])
        bool_mask = np.array([True, False, True, False])
        idx_mask = np.array([0, 2], dtype=np.int64)
        r_bool = compute_layer_coherence(phases, bool_mask)
        r_idx = compute_layer_coherence(phases, idx_mask)
        assert r_bool == pytest.approx(r_idx, abs=1e-12)

    def test_single_index_r_one(self) -> None:
        phases = np.array([0.7, 1.0, 2.0])
        r = compute_layer_coherence(phases, np.array([0], dtype=np.int64))
        assert r == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


class TestDispatcher:
    def test_rust_loader_maps_spo_kernel_symbols(self, monkeypatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def order_parameter(phases: np.ndarray) -> tuple[float, float]:
            return float(phases.size), 0.0

        def plv(a: np.ndarray, b: np.ndarray) -> float:
            return float(a.size == b.size)

        def compute_layer_coherence_rust(
            phases: np.ndarray, indices: np.ndarray
        ) -> float:
            return float(indices.size) / max(float(phases.size), 1.0)

        fake_spo_kernel = types.ModuleType("spo_kernel")
        fake_spo_kernel.order_parameter = order_parameter
        fake_spo_kernel.plv = plv
        fake_spo_kernel.compute_layer_coherence_rust = compute_layer_coherence_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo_kernel)

        loaded = op_mod._load_rust_fns()

        phases = np.arange(4.0)
        assert loaded["order_parameter"](phases) == (4.0, 0.0)
        assert loaded["plv"](phases, phases) == 1.0
        assert loaded["layer_coherence"](phases, np.array([0, 2])) == 0.5

    def test_python_is_always_available(self) -> None:
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        assert ACTIVE_BACKEND in AVAILABLE_BACKENDS

    def test_available_backends_keep_canonical_fallback_order(self) -> None:
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices)

    def test_probe_marks_faulty_backend_unusable(self, monkeypatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def broken_backend(name: str) -> dict[str, object]:
            raise RuntimeError(f"{name} unavailable during probe")

        monkeypatch.setattr(op_mod, "_load_backend", broken_backend)

        elapsed = op_mod._order_parameter_probe_seconds("rust")

        assert elapsed == float("inf")

    def test_resolve_backends_keeps_working_backend_before_python(
        self, monkeypatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_loader(name: str) -> dict[str, object]:
            if name == "rust":
                return {
                    "order_parameter": lambda phases: (
                        float(phases.size),
                        float(phases[0]),
                    )
                }
            raise ImportError(name)

        monkeypatch.setattr(op_mod, "_BACKEND_NAMES", ("rust", "mojo", "python"))
        monkeypatch.setattr(op_mod, "_load_backend", fake_loader)
        monkeypatch.setattr(
            op_mod,
            "_order_parameter_probe_seconds",
            lambda name: 0.0 if name == "rust" else 1.0,
        )

        active, available = op_mod._resolve_backends()

        assert active == "rust"
        assert available == ["rust", "python"]

    def test_dispatch_ignores_rust_when_rust_flag_is_false(self, monkeypatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", False)

        assert op_mod._dispatch("order_parameter") is None

    def test_dispatch_falls_back_when_active_backend_raises(self, monkeypatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def failing_loader(name: str) -> dict[str, object]:
            raise OSError(f"{name} backend failed after selection")

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "mojo")
        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        monkeypatch.setattr(op_mod, "_load_backend", failing_loader)

        assert op_mod._dispatch("plv") is None

    def test_public_kernels_use_active_backend_with_flat_contiguous_payloads(
        self, monkeypatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        observed: dict[str, tuple[tuple[int, ...], bool, str]] = {}

        def fake_order(phases: np.ndarray) -> tuple[float, float]:
            observed["order"] = (
                phases.shape,
                phases.flags.c_contiguous,
                phases.dtype.name,
            )
            return 0.25, 1.5

        def fake_plv(a: np.ndarray, b: np.ndarray) -> float:
            observed["plv_a"] = (a.shape, a.flags.c_contiguous, a.dtype.name)
            observed["plv_b"] = (b.shape, b.flags.c_contiguous, b.dtype.name)
            return 0.75

        def fake_layer(phases: np.ndarray, indices: np.ndarray) -> float:
            observed["layer_phases"] = (
                phases.shape,
                phases.flags.c_contiguous,
                phases.dtype.name,
            )
            observed["layer_indices"] = (
                indices.shape,
                indices.flags.c_contiguous,
                indices.dtype.name,
            )
            return 0.5

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {
                "order_parameter": fake_order,
                "plv": fake_plv,
                "layer_coherence": fake_layer,
            },
        )

        phases_2d = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
        r, psi = op_mod.compute_order_parameter(phases_2d)
        plv = op_mod.compute_plv(phases_2d, phases_2d + 0.25)
        layer = op_mod.compute_layer_coherence(
            phases_2d,
            np.array([True, False, True, False]),
        )

        assert (r, psi) == (0.25, 1.5)
        assert plv == 0.75
        assert layer == 0.5
        assert observed["order"] == ((4,), True, "float64")
        assert observed["plv_a"] == ((4,), True, "float64")
        assert observed["plv_b"] == ((4,), True, "float64")
        assert observed["layer_phases"] == ((4,), True, "float64")
        assert observed["layer_indices"] == ((2,), True, "int64")

    def test_layer_coherence_empty_selected_subarray_returns_zero(
        self, monkeypatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "python")
        phases = np.empty((1, 0), dtype=np.float64)
        indices = np.array([0], dtype=np.int64)

        assert op_mod.compute_layer_coherence(phases, indices) == 0.0

    def test_python_fallback_kernels_preserve_physical_invariants(
        self, monkeypatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "python")
        phases = np.array([0.0, np.pi], dtype=np.float64)
        shifted = phases + 0.5

        r, psi = op_mod.compute_order_parameter(phases)
        plv = op_mod.compute_plv(phases, shifted)
        layer = op_mod.compute_layer_coherence(
            phases,
            np.array([0, 1], dtype=np.int64),
        )

        assert r == pytest.approx(0.0, abs=1e-12)
        assert 0.0 <= psi < TWO_PI
        assert plv == pytest.approx(1.0, abs=1e-12)
        assert layer == pytest.approx(0.0, abs=1e-12)
