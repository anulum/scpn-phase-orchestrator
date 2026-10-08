# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for chimera detection

"""Check public local-order and classification algorithm contracts.

Covered: bounded ``[0, 1]`` output, perfect-sync ``R_i = 1``,
antiphase ``R_i = 0``, disconnected-oscillator ``R_i = 0``, correct
partition on a staged chimera (half coherent, half incoherent),
chimera-index is a fraction, empty-input safety, Hypothesis
property coverage.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import chimera as ch_mod
from scpn_phase_orchestrator.monitor.chimera import (
    ChimeraState,
    detect_chimera,
    local_order_parameter,
)

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi


def _all_to_all(n: int) -> FloatArray:
    """Construct unweighted positive non-self all-to-all coupling."""
    k = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(k, 0.0)
    return k


class TestLocalOrderParameter:
    """Check local-order topology, numerical bounds and shape admission."""

    def test_perfect_sync_gives_one(self) -> None:
        """Return unit coherence for equal neighbour phases."""
        n = 8
        phases = np.full(n, 0.3)
        r = local_order_parameter(phases, _all_to_all(n), backend="python")
        assert np.all(np.abs(r - 1.0) < 1e-12)

    def test_uniform_on_circle_gives_1_over_N_minus_1(self) -> None:
        """Check the exact local-order magnitude of evenly spaced oscillators.

        With all-to-all coupling, excluding self leaves an imbalance of
        exactly ``-1 · e^{iθ_i}``, so ``R_i = 1 / (N-1)`` for every
        ``i`` — asymptotically zero as N grows.
        """
        n = 32
        phases = np.linspace(0.0, TWO_PI, n, endpoint=False)
        r = local_order_parameter(phases, _all_to_all(n), backend="python")
        expected = 1.0 / (n - 1)
        np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_bounded_unit_interval(self) -> None:
        """Keep seeded sparse-graph magnitudes in the unit interval."""
        rng = np.random.default_rng(0)
        n = 16
        phases = rng.uniform(0, TWO_PI, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        knm = (knm > 0.3).astype(np.float64) * knm
        np.fill_diagonal(knm, 0.0)
        r = local_order_parameter(phases, knm, backend="python")
        assert np.all(r >= 0.0 - 1e-12)
        assert np.all(r <= 1.0 + 1e-12)

    def test_global_phase_shift_invariance(self) -> None:
        """R_i depends on phase differences, not the absolute phase gauge."""
        rng = np.random.default_rng(12)
        n = 10
        phases = rng.uniform(-math.pi, math.pi, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        np.fill_diagonal(knm, 0.0)

        base = local_order_parameter(phases, knm, backend="python")
        shifted = local_order_parameter(phases + 23.0, knm, backend="python")

        np.testing.assert_allclose(shifted, base, atol=1e-12)

    def test_permutation_equivariance(self) -> None:
        """Relabelling oscillators relabels local order parameters exactly."""
        rng = np.random.default_rng(13)
        n = 9
        phases = rng.uniform(0.0, TWO_PI, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        np.fill_diagonal(knm, 0.0)
        permutation = np.array([2, 5, 1, 8, 0, 6, 3, 7, 4])

        base = local_order_parameter(phases, knm, backend="python")
        relabelled = local_order_parameter(
            phases[permutation], knm[np.ix_(permutation, permutation)], backend="python"
        )

        np.testing.assert_allclose(relabelled, base[permutation], atol=1e-12)

    def test_negative_couplings_are_not_local_neighbours(self) -> None:
        """The public contract defines neighbours by K_ij > 0 only."""
        phases = np.array([0.0, 0.3, 1.1])
        knm = np.array(
            [
                [0.0, -2.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        r = local_order_parameter(phases, knm, backend="python")
        assert r[0] == 0.0
        assert r[1] == pytest.approx(1.0)
        assert r[2] == pytest.approx(1.0)

    def test_isolated_oscillator_returns_zero(self) -> None:
        # Oscillator 0 has no outgoing edges.
        """Give zero coherence to an oscillator without positive outgoing edges."""
        n = 4
        knm = np.zeros((n, n))
        knm[1, 2] = knm[2, 1] = knm[1, 3] = knm[3, 1] = 1.0
        phases = np.array([0.1, 0.2, 0.2, 0.2])
        r = local_order_parameter(phases, knm, backend="python")
        assert r[0] == 0.0

    def test_empty_input(self) -> None:
        """Preserve the empty local-order identity."""
        r = local_order_parameter(np.array([]), np.zeros((0, 0)), backend="python")
        assert r.shape == (0,)

    @pytest.mark.parametrize(
        ("phases", "knm", "match"),
        [
            (np.zeros((2, 2), dtype=np.float64), np.zeros((4, 4)), "phases"),
            (np.array([0.0, np.nan]), np.zeros((2, 2)), "phases"),
            (np.array([True, False]), np.zeros((2, 2)), "phases"),
            (np.array([0.0 + 0.0j, 0.5 + 0.25j]), np.zeros((2, 2)), "real-valued"),
            (np.zeros(3), np.zeros((2, 2)), "knm shape"),
            (np.zeros(2), np.array([[0.0, np.inf], [0.0, 0.0]]), "knm"),
            (np.zeros(2), np.array([[True, False], [False, True]]), "knm"),
            (np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]]), "diagonal"),
            (
                np.zeros(2),
                np.array([[0.0 + 0.0j, 1.0 + 0.25j], [1.0, 0.0 + 0.0j]]),
                "real-valued",
            ),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        phases: FloatArray,
        knm: FloatArray,
        match: str,
    ) -> None:
        """Refuse original aliases and invalid finite graph dimensions."""
        with pytest.raises(ValueError, match=match):
            local_order_parameter(phases, knm, backend="python")


class TestDetectChimera:
    """Check population classification and the model-specific index."""

    def test_perfect_sync_all_coherent(self) -> None:
        """Classify coupled synchronized oscillators as coherent."""
        n = 10
        phases = np.full(n, 0.5)
        state = detect_chimera(phases, _all_to_all(n), backend="python")
        assert len(state.coherent_indices) == n
        assert len(state.incoherent_indices) == 0
        assert state.chimera_index == 0.0

    def test_antiphase_classified_incoherent(self) -> None:
        """Classify the balanced antiphase neighbourhood below the low threshold."""
        n = 6
        phases = np.array([0.0, math.pi, 0.0, math.pi, 0.0, math.pi])
        state = detect_chimera(phases, _all_to_all(n), backend="python")
        assert len(state.incoherent_indices) == n

    def test_chimera_index_is_fraction(self) -> None:
        """Keep the instantaneous boundary count normalized by population."""
        rng = np.random.default_rng(11)
        n = 32
        phases = rng.uniform(0, TWO_PI, n)
        state = detect_chimera(phases, _all_to_all(n), backend="python")
        assert 0.0 <= state.chimera_index <= 1.0

    def test_partition_totals_to_n(self) -> None:
        """|coherent| + |incoherent| + boundary = N."""
        rng = np.random.default_rng(7)
        n = 32
        phases = np.concatenate([np.zeros(n // 2), rng.uniform(0, TWO_PI, n // 2)])
        state = detect_chimera(phases, _all_to_all(n), backend="python")
        boundary = int(round(state.chimera_index * n))
        assert (
            len(state.coherent_indices) + len(state.incoherent_indices) + boundary == n
        )

    def test_detection_invariant_under_global_phase_shift(self) -> None:
        """Preserve coherent and incoherent classes under a common phase rotation."""
        rng = np.random.default_rng(19)
        n = 18
        phases = rng.uniform(-math.pi, math.pi, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        np.fill_diagonal(knm, 0.0)

        base = detect_chimera(phases, knm, backend="python")
        shifted = detect_chimera(phases - 41.0, knm, backend="python")

        assert shifted == base

    def test_detection_permutation_equivariance(self) -> None:
        """Relabel phases and directed graph rows consistently in classification."""
        rng = np.random.default_rng(23)
        n = 12
        phases = np.concatenate([np.zeros(n // 2), rng.uniform(0.0, TWO_PI, n // 2)])
        knm = _all_to_all(n)
        permutation = np.array([7, 0, 11, 2, 5, 9, 1, 10, 3, 8, 4, 6])
        inverse = {int(old): int(new) for new, old in enumerate(permutation)}

        base = detect_chimera(phases, knm, backend="python")
        relabelled = detect_chimera(
            phases[permutation], knm[np.ix_(permutation, permutation)], backend="python"
        )

        expected_coherent = sorted(inverse[idx] for idx in base.coherent_indices)
        expected_incoherent = sorted(inverse[idx] for idx in base.incoherent_indices)
        assert sorted(relabelled.coherent_indices) == expected_coherent
        assert sorted(relabelled.incoherent_indices) == expected_incoherent
        assert relabelled.chimera_index == pytest.approx(base.chimera_index)

    def test_empty_input_returns_empty_state(self) -> None:
        """Preserve empty coherent and incoherent lists."""
        state = detect_chimera(np.array([]), np.zeros((0, 0)), backend="python")
        assert isinstance(state, ChimeraState)
        assert state.coherent_indices == []
        assert state.incoherent_indices == []
        assert state.chimera_index == 0.0


class TestChimeraStateBoundaries:
    """Keep coherent, incoherent and boundary populations consistent."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"coherent_indices": [0, 0]}, "duplicate"),
            ({"coherent_indices": [0], "incoherent_indices": [0]}, "disjoint"),
            ({"coherent_indices": [True]}, "integer"),
            ({"coherent_indices": [-1]}, "non-negative"),
            ({"chimera_index": True}, "finite real"),
            ({"chimera_index": 1.5}, "lie in \\[0, 1\\]"),
        ],
    )
    def test_constructor_rejects_invalid_public_state(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Refuse invalid original result fields without coercing them."""
        with pytest.raises(ValueError, match=match):
            ChimeraState(
                coherent_indices=cast(list[int], kwargs.get("coherent_indices", [])),
                incoherent_indices=cast(
                    list[int], kwargs.get("incoherent_indices", [])
                ),
                chimera_index=cast(float, kwargs.get("chimera_index", 0.0)),
            )


class TestHypothesis:
    """Exercise sampled phase and topology invariants through the public API."""

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bounded_and_finite(self, n: int, seed: int) -> None:
        """Exercise genuine seeded Python owner graphs across sizes."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        knm = (knm > 0.4).astype(np.float64) * knm
        np.fill_diagonal(knm, 0.0)
        r = local_order_parameter(phases, knm, backend="python")
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0 - 1e-12)
        assert np.all(r <= 1.0 + 1e-12)


class TestDispatcherSurface:
    """Check public backend metadata and numerical dispatch."""

    def test_available_non_empty(self) -> None:
        """Keep Python in the actual public availability declaration."""
        assert ch_mod.AVAILABLE_BACKENDS
        assert "python" in ch_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self) -> None:
        """Keep actual default preference first in the availability order."""
        assert ch_mod.AVAILABLE_BACKENDS[0] == ch_mod.ACTIVE_BACKEND
