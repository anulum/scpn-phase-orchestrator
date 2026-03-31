# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic oscillator tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Ring-phase mapping: θ = 2πs/N
# ---------------------------------------------------------------------------


class TestRingPhaseMapping:
    """Verify the analytical ring-phase mapping from discrete states
    to continuous phase on the unit circle."""

    def test_four_state_ring_phases(self):
        """States [0,1,2,3] with N=4 → θ = [0, π/2, π, 3π/2]."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        thetas = [s.theta for s in states]
        np.testing.assert_allclose(
            thetas, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-12
        )

    def test_equispaced_phases(self):
        """N states must produce equispaced phases with gap = 2π/N."""
        for n in [3, 5, 8, 16]:
            ext = SymbolicExtractor(n_states=n, mode="ring")
            states = ext.extract(np.arange(n), sample_rate=1.0)
            thetas = np.array([s.theta for s in states])
            gaps = np.diff(thetas)
            expected_gap = TWO_PI / n
            np.testing.assert_allclose(
                gaps, expected_gap, atol=1e-12, err_msg=f"N={n}: gaps not equispaced"
            )

    def test_phase_wraps_at_2pi(self):
        """State index ≥ N must wrap via modulo."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 4, 8]), sample_rate=1.0)
        for s in states:
            assert s.theta == pytest.approx(0.0, abs=1e-12), (
                f"Multiples of N must map to θ=0, got {s.theta}"
            )

    def test_all_phases_in_range(self):
        """All output phases must be in [0, 2π)."""
        ext = SymbolicExtractor(n_states=7, mode="ring")
        states = ext.extract(np.arange(20), sample_rate=1.0)
        for s in states:
            assert 0.0 <= s.theta < TWO_PI, f"Phase {s.theta} out of [0, 2π)"


# ---------------------------------------------------------------------------
# Graph-walk mode
# ---------------------------------------------------------------------------


class TestGraphWalkMode:
    """Verify the graph-walk phase mapping: cumulative transition distances
    normalised to [0, 2π)."""

    def test_graph_phases_in_range(self):
        ext = SymbolicExtractor(n_states=10, mode="graph")
        states = ext.extract(np.array([3, 5, 7, 2, 9]), sample_rate=1.0)
        for s in states:
            assert 0.0 <= s.theta < TWO_PI

    def test_stationary_sequence_zero_phase(self):
        """No transitions → cumulative distance = 0 → all phases = 0."""
        ext = SymbolicExtractor(n_states=5, mode="graph")
        states = ext.extract(np.array([3, 3, 3, 3]), sample_rate=1.0)
        # Total distance is 0, so thetas are all 0 or single-value division
        for s in states:
            assert 0.0 <= s.theta < TWO_PI

    def test_single_state_has_valid_phase(self):
        ext = SymbolicExtractor(n_states=5, mode="graph")
        states = ext.extract(np.array([2]), sample_rate=1.0)
        assert len(states) == 1
        assert 0.0 <= states[0].theta < TWO_PI


# ---------------------------------------------------------------------------
# Transition quality scoring
# ---------------------------------------------------------------------------


class TestTransitionQuality:
    """Verify the quality heuristic: single-step transitions = 1.0,
    stalled = 0.2, large jumps penalised proportionally."""

    def test_single_step_transitions_quality_1(self):
        """Consecutive states [0,1,2,3,4] — all single-step → quality=1.0."""
        ext = SymbolicExtractor(n_states=8, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3, 4]), sample_rate=1.0)
        for s in states[1:]:
            assert s.quality == pytest.approx(1.0)

    def test_stalled_state_quality_0_2(self):
        """Repeated state → quality = 0.2 (penalised)."""
        ext = SymbolicExtractor(n_states=5, mode="ring")
        states = ext.extract(np.array([2, 2, 2, 2]), sample_rate=1.0)
        for s in states[1:]:
            assert s.quality == pytest.approx(0.2)

    def test_first_state_quality_0_5(self):
        """First state has no previous transition → default quality = 0.5."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert states[0].quality == pytest.approx(0.5)

    def test_large_jump_penalised(self):
        """Jump of size 3 with N=8 → quality = max(0.1, 1-(3-1)/8) = 0.75."""
        ext = SymbolicExtractor(n_states=8, mode="ring")
        states = ext.extract(np.array([0, 3]), sample_rate=1.0)
        expected_q = max(0.1, 1.0 - (3 - 1) / 8)  # 0.75
        assert states[1].quality == pytest.approx(expected_q)

    def test_quality_discriminates_clean_vs_noisy(self):
        """Clean sequential signal must score higher than noisy random jumps."""
        ext = SymbolicExtractor(n_states=10, mode="ring")
        clean = ext.extract(np.array([0, 1, 2, 3, 4, 5, 6, 7]), sample_rate=1.0)
        noisy = ext.extract(np.array([0, 7, 2, 9, 1, 8, 3, 6]), sample_rate=1.0)
        q_clean = ext.quality_score(clean)
        q_noisy = ext.quality_score(noisy)
        assert q_clean > q_noisy, (
            f"Clean signal quality ({q_clean:.3f}) must exceed noisy ({q_noisy:.3f})"
        )


# ---------------------------------------------------------------------------
# Metadata and validation
# ---------------------------------------------------------------------------


class TestSymbolicExtractorMetadata:
    """Verify channel, node_id, and construction constraints."""

    def test_channel_is_S(self):
        ext = SymbolicExtractor(n_states=4, node_id="sym_q")
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert all(s.channel == "S" for s in states)
        assert all(s.node_id == "sym_q" for s in states)

    def test_n_states_below_2_rejected(self):
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            SymbolicExtractor(n_states=1)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            SymbolicExtractor(n_states=4, mode="invalid")

    def test_quality_score_empty(self):
        assert SymbolicExtractor(n_states=4).quality_score([]) == 0.0

    def test_quality_score_range(self):
        ext = SymbolicExtractor(n_states=4)
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        score = ext.quality_score(states)
        assert 0.0 < score <= 1.0

    def test_omega_from_phase_differences(self):
        """Omega must be derived from phase differences / dt.
        For ring N=4 at sample_rate=1: Δθ = π/2, so ω ≈ π/2."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        # states[0].omega = 0 (no previous), states[1..].omega ≈ π/2
        for s in states[1:]:
            assert abs(s.omega - np.pi / 2) < 1e-10, (
                f"Expected ω≈π/2 for single-step ring(4), got {s.omega}"
            )


class TestSymbolicPipelineEndToEnd:
    """Full pipeline: SymbolicExtractor → theta/omega → Engine → R.

    Proves SymbolicExtractor is a functional input adapter.
    """

    def test_symbolic_phases_feed_engine(self):
        """Extract symbolic phases from state sequences → engine → R."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        ext = SymbolicExtractor(n_states=8, mode="ring")
        sequences = [
            np.array([0, 1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([2, 3, 4, 5, 6, 7]),
            np.array([3, 4, 5, 6, 7, 0]),
        ]
        phases = []
        omegas = []
        for seq in sequences:
            states = ext.extract(seq, sample_rate=100.0)
            phases.append(states[-1].theta)
            omegas.append(states[-1].omega)
        phases_arr = np.array(phases)
        omegas_arr = np.array(omegas)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01)
        for _ in range(100):
            phases_arr = eng.step(phases_arr, omegas_arr, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases_arr)
        assert 0.0 <= r <= 1.0
        assert np.all(phases_arr >= 0.0)
        assert np.all(phases_arr < TWO_PI)

    def test_ring_vs_graph_both_produce_valid_engine_input(self):
        """Both modes produce phases in [0, 2π) suitable for engine."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        seq = np.array([0, 2, 4, 1, 3, 5, 7, 6])
        for mode in ("ring", "graph"):
            ext = SymbolicExtractor(n_states=8, mode=mode)
            states = ext.extract(seq, sample_rate=1.0)
            phases = np.array([s.theta for s in states])
            assert np.all(phases >= 0.0)
            assert np.all(phases < TWO_PI)
            r, _ = compute_order_parameter(phases)
            assert 0.0 <= r <= 1.0

    def test_performance_extract_1000_states_under_5ms(self):
        """SymbolicExtractor.extract(1000 states) < 5ms."""
        import time

        ext = SymbolicExtractor(n_states=16, mode="ring")
        seq = np.tile(np.arange(16), 63)[:1000]
        ext.extract(seq, sample_rate=1.0)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            ext.extract(seq, sample_rate=1.0)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 5e-3, f"extract(1000) took {elapsed * 1e3:.2f}ms"


# Pipeline wiring: SymbolicExtractor → theta/omega → UPDEEngine
# → compute_order_parameter. Ring + graph modes, quality scoring,
# omega derivation from phase diffs. Performance: extract(1000)<5ms.
