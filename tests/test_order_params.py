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
    def test_python_is_always_available(self) -> None:
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        assert AVAILABLE_BACKENDS[0] == ACTIVE_BACKEND

    def test_fastest_first_ordering(self) -> None:
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices)
