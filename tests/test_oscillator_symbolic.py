# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic oscillator tests

"""Exercise symbolic extraction, cyclic quality and public engine integration."""

from __future__ import annotations

import importlib.util
import os
from typing import cast, get_type_hints

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


def test_large_vocabulary_retains_the_declared_float64_rounding_contract() -> None:
    """Real backends retain exact residues but may round ratios differently."""
    n_states = 2**53 + 1
    use_native = importlib.util.find_spec("spo_kernel") is not None and n_states <= int(
        np.iinfo(np.uintp).max
    )
    ratio = 1.0 / float(n_states) if use_native else 1 / n_states
    ring = SymbolicExtractor(n_states=n_states).extract(
        np.array([0, 1], dtype=np.int64), sample_rate=1.0
    )
    assert ring[1].theta == TWO_PI * ratio
    assert ring[1].theta in (TWO_PI / float(n_states), TWO_PI * (1 / n_states))

    distance = 3 * 2**51 + 1
    penalty = (
        float(distance - 1) / float(n_states)
        if use_native
        else (distance - 1) / n_states
    )
    graph = SymbolicExtractor(n_states=n_states, mode="graph").extract(
        np.array([0, distance], dtype=np.int64), sample_rate=1.0
    )
    assert graph[1].quality == 1.0 - penalty


# ---------------------------------------------------------------------------
# Ring-phase mapping: θ = 2πs/N
# ---------------------------------------------------------------------------


class TestRingPhaseMapping:
    """Verify the analytical ring-phase mapping.

    Map discrete states to continuous phase on the unit circle.
    """

    def test_four_state_ring_phases(self) -> None:
        """States [0,1,2,3] with N=4 → θ = [0, π/2, π, 3π/2]."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        thetas = [s.theta for s in states]
        np.testing.assert_allclose(
            thetas, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-12
        )

    def test_equispaced_phases(self) -> None:
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

    def test_phase_wraps_at_2pi(self) -> None:
        """State index ≥ N must wrap via modulo."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 4, 8]), sample_rate=1.0)
        for s in states:
            assert s.theta == pytest.approx(0.0, abs=1e-12), (
                f"Multiples of N must map to θ=0, got {s.theta}"
            )

    def test_all_phases_in_range(self) -> None:
        """All output phases must be in [0, 2π)."""
        ext = SymbolicExtractor(n_states=7, mode="ring")
        states = ext.extract(np.arange(20), sample_rate=1.0)
        for s in states:
            assert 0.0 <= s.theta < TWO_PI, f"Phase {s.theta} out of [0, 2π)"


# ---------------------------------------------------------------------------
# Graph-walk mode
# ---------------------------------------------------------------------------


class TestGraphWalkMode:
    """Verify graph-walk cumulative transition distances.

    Normalise the observed walk to [0, 2π).
    """

    def test_graph_phases_in_range(self) -> None:
        """Keep observed graph phases on the unit circle."""
        ext = SymbolicExtractor(n_states=10, mode="graph")
        states = ext.extract(np.array([3, 5, 7, 2, 9]), sample_rate=1.0)
        for s in states:
            assert 0.0 <= s.theta < TWO_PI

    def test_stationary_sequence_zero_phase(self) -> None:
        """No transitions → cumulative distance = 0 → all phases = 0."""
        ext = SymbolicExtractor(n_states=5, mode="graph")
        states = ext.extract(np.array([3, 3, 3, 3]), sample_rate=1.0)
        assert [state.theta for state in states] == pytest.approx([0.0] * 4)
        assert [state.omega for state in states] == pytest.approx([0.0] * 4)

    def test_single_state_has_valid_phase(self) -> None:
        """Map a singleton graph sequence through the ring mapping."""
        ext = SymbolicExtractor(n_states=5, mode="graph")
        states = ext.extract(np.array([2]), sample_rate=1.0)
        assert len(states) == 1
        assert states[0].theta == pytest.approx(4 * np.pi / 5)
        assert states[0].omega == 0.0
        assert states[0].quality == 0.5


# ---------------------------------------------------------------------------
# Transition quality scoring
# ---------------------------------------------------------------------------


class TestTransitionQuality:
    """Verify transition quality.

    Single steps score 1.0, stalls 0.2, and larger jumps are penalised.
    """

    def test_single_step_transitions_quality_1(self) -> None:
        """Consecutive states [0,1,2,3,4] — all single-step → quality=1.0."""
        ext = SymbolicExtractor(n_states=8, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3, 4]), sample_rate=1.0)
        for s in states[1:]:
            assert s.quality == pytest.approx(1.0)

    def test_stalled_state_quality_0_2(self) -> None:
        """Repeated state → quality = 0.2 (penalised)."""
        ext = SymbolicExtractor(n_states=5, mode="ring")
        states = ext.extract(np.array([2, 2, 2, 2]), sample_rate=1.0)
        for s in states[1:]:
            assert s.quality == pytest.approx(0.2)

    def test_first_state_quality_0_5(self) -> None:
        """First state has no previous transition → default quality = 0.5."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert states[0].quality == pytest.approx(0.5)

    def test_first_state_quality_respects_explicit_policy_override(self) -> None:
        """Respect the configured initial transition quality."""
        ext = SymbolicExtractor(
            n_states=4,
            mode="ring",
            initial_transition_quality=0.8,
        )
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert states[0].quality == pytest.approx(0.8)

    def test_large_jump_penalised(self) -> None:
        """Jump of size 3 with N=8 → quality = max(0.1, 1-(3-1)/8) = 0.75."""
        ext = SymbolicExtractor(n_states=8, mode="ring")
        states = ext.extract(np.array([0, 3]), sample_rate=1.0)
        expected_q = max(0.1, 1.0 - (3 - 1) / 8)  # 0.75
        assert states[1].quality == pytest.approx(expected_q)

    def test_quality_discriminates_clean_vs_noisy(self) -> None:
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

    def test_extract_signal_type_hint_includes_discrete_integer_contract(self) -> None:
        """Retain the public discrete ndarray typing contract."""
        hint = get_type_hints(SymbolicExtractor.extract)["signal"]
        assert_precise_ndarray_hint(hint)
        assert "int64" in str(hint)

    def test_channel_is_S(self) -> None:
        """Preserve symbolic channel and requested oscillator identifier."""
        ext = SymbolicExtractor(n_states=4, node_id="sym_q")
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert all(s.channel == "S" for s in states)
        assert all(s.node_id == "sym_q" for s in states)

    @pytest.mark.parametrize("node_id", ["", "   ", 42, True])
    def test_invalid_node_id_rejected(self, node_id: object) -> None:
        """Reject blank and non-string oscillator identifiers."""
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            SymbolicExtractor(n_states=4, node_id=cast(str, node_id))

    def test_n_states_below_2_rejected(self) -> None:
        """Reject a vocabulary without two distinct states."""
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            SymbolicExtractor(n_states=1)

    @pytest.mark.parametrize("n_states", [True, 4.0, "4"])
    def test_non_integer_n_states_rejected(self, n_states: object) -> None:
        """Reject boolean, fractional and textual vocabulary sizes."""
        with pytest.raises(ValueError, match="n_states must be an integer"):
            SymbolicExtractor(n_states=cast(int, n_states))

    def test_numpy_integer_n_states_normalised(self) -> None:
        """Accept a NumPy integral vocabulary size."""
        ext = SymbolicExtractor(n_states=cast(int, np.int64(4)))
        states = ext.extract(np.array([0, 1]), sample_rate=1.0)
        assert states[1].theta == pytest.approx(np.pi / 2)

    def test_invalid_mode_rejected(self) -> None:
        """Reject an unknown extraction mode."""
        with pytest.raises(ValueError, match="mode must be"):
            SymbolicExtractor(n_states=4, mode="invalid")

    @pytest.mark.parametrize(
        "initial_transition_quality",
        [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"],
    )
    def test_invalid_initial_transition_quality_rejected(
        self, initial_transition_quality: object
    ) -> None:
        """Reject non-real or out-of-range initial qualities."""
        with pytest.raises(ValueError, match="initial_transition_quality"):
            SymbolicExtractor(
                n_states=4,
                initial_transition_quality=cast(float, initial_transition_quality),
            )

    def test_quality_score_empty(self) -> None:
        """Score an empty observation list as zero."""
        assert SymbolicExtractor(n_states=4).quality_score([]) == 0.0

    def test_quality_score_range(self) -> None:
        """Keep the mean quality in the unit interval."""
        ext = SymbolicExtractor(n_states=4)
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        score = ext.quality_score(states)
        assert 0.0 < score <= 1.0

    def test_omega_from_phase_differences(self) -> None:
        """Omega must be derived from phase differences / dt.

        For ring N=4 at sample_rate=1: Δθ = π/2, so ω ≈ π/2.
        """
        ext = SymbolicExtractor(n_states=4, mode="ring")
        states = ext.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
        # states[0].omega = 0 (no previous), states[1..].omega ≈ π/2
        for s in states[1:]:
            assert abs(s.omega - np.pi / 2) < 1e-10, (
                f"Expected ω≈π/2 for single-step ring(4), got {s.omega}"
            )

    @pytest.mark.parametrize(
        "signal",
        [
            np.array([True, False]),
            np.array([0.0, 1.0]),
            np.array([1 + 0j]),
            np.array(["1"], dtype=object),
        ],
    )
    def test_extract_rejects_non_integer_signal(self, signal: object) -> None:
        """Reject boolean, floating, complex and object observations."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        with pytest.raises(ValueError, match="signal must be integer"):
            ext.extract(cast(NDArray[np.int64], signal), sample_rate=1.0)

    def test_extract_rejects_multidimensional_signal(self) -> None:
        """Reject state arrays with multiple dimensions."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        with pytest.raises(ValueError, match="signal must be 1-D"):
            ext.extract(np.array([[0, 1], [2, 3]]), sample_rate=1.0)

    @pytest.mark.parametrize(
        "sample_rate",
        [True, 0.0, -1.0, float("nan"), float("inf"), "1.0"],
    )
    def test_extract_rejects_invalid_sample_rate(self, sample_rate: object) -> None:
        """Reject non-positive, non-real or non-finite rates."""
        ext = SymbolicExtractor(n_states=4, mode="ring")
        with pytest.raises(ValueError, match="sample_rate must be finite and positive"):
            ext.extract(np.array([0, 1]), sample_rate=cast(float, sample_rate))


class TestSymbolicPipelineEndToEnd:
    """Full pipeline: SymbolicExtractor → theta/omega → Engine → R.

    Proves SymbolicExtractor is a functional input adapter.
    """

    def test_symbolic_phases_feed_engine(self) -> None:
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

    def test_ring_vs_graph_both_produce_valid_engine_input(self) -> None:
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

    def test_performance_extract_1000_states_under_5ms(self) -> None:
        """SymbolicExtractor.extract(1000 states) < 5ms."""
        import time

        ext = SymbolicExtractor(n_states=16, mode="ring")
        seq = np.tile(np.arange(16), 63)[:1000]
        ext.extract(seq, sample_rate=1.0)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            ext.extract(seq, sample_rate=1.0)
        elapsed = (time.perf_counter() - t0) / 100
        # CI shared runners and locally loaded workstations can be slower.
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        load_threshold = max(2.0, float(os.cpu_count() or 1) / 4.0)
        budget = 50e-3 if (os.getenv("CI") or load > load_threshold) else 5e-3
        assert elapsed < budget, f"extract(1000) took {elapsed * 1e3:.2f}ms"


def test_ring_mode_scores_the_circular_step() -> None:
    """Ring phases and qualities, with or without the kernel, no substitution.

    On a ring the wrap from N-1 to 0 is one step: omega already treats it so,
    and the quality now does too instead of scoring an (N-1)-sized jump.
    """
    ext = SymbolicExtractor(n_states=4, mode="ring")
    states = ext.extract(np.array([0, 1, 3]), sample_rate=1.0)
    assert [state.theta for state in states] == pytest.approx(
        [0.0, np.pi / 2, 3 * np.pi / 2]
    )
    assert [state.quality for state in states] == pytest.approx([0.5, 1.0, 0.75])

    wrap = SymbolicExtractor(n_states=6, mode="ring").extract(
        np.array([3, 4, 5, 0, 1]), sample_rate=1.0
    )
    assert [state.quality for state in wrap] == pytest.approx([0.5, 1.0, 1.0, 1.0, 1.0])
    assert {round(state.omega, 9) for state in wrap[1:]} == {round(np.pi / 3, 9)}


def test_graph_mode_keeps_the_linear_step() -> None:
    """Graph-walk distance is linear: 5 -> 0 on six states is a jump of five."""
    states = SymbolicExtractor(n_states=6, mode="graph").extract(
        np.array([4, 5, 0]), sample_rate=1.0
    )
    assert states[2].quality == pytest.approx(max(0.1, 1.0 - 4 / 6))


@pytest.mark.parametrize("indices", [[0, 2, 5, 1], [5, 3, 0, 4], [1, 3, 6, 2]])
@pytest.mark.parametrize("sample_rate", [1.0, 8.0])
def test_graph_walk_matches_observed_linear_distance(
    indices: list[int], sample_rate: float
) -> None:
    """Pin phases, signed velocity and quality through real graph extraction."""
    signal = np.array(indices, dtype=np.int64)
    original = signal.copy()
    extractor = SymbolicExtractor(n_states=8, node_id="workflow", mode="graph")
    states = extractor.extract(signal, sample_rate)
    np.testing.assert_array_equal(signal, original)
    assert [state.theta for state in states] == pytest.approx(
        [0.0, 4 * np.pi / 9, 10 * np.pi / 9, 0.0], abs=1e-12
    )
    assert [state.omega for state in states] == pytest.approx(
        np.array([0.0, 4 * np.pi / 9, 2 * np.pi / 3, 8 * np.pi / 9]) * sample_rate
    )
    assert [state.quality for state in states] == pytest.approx(
        [0.5, 0.875, 0.75, 0.625]
    )
    assert extractor.quality_score(states) == pytest.approx(0.6875)
    assert all(
        state.amplitude == 1.0 and state.channel == "S" and state.node_id == "workflow"
        for state in states
    )


@pytest.mark.parametrize(
    "indices",
    [
        [3, 0, 1, 3],
        [7, 0, 9, 7],
        [-5, 0, 9, -1],
        [2**53 + 3, -(2**62), 2**53 + 1, -(2**62) + 3],
    ],
)
@pytest.mark.parametrize("sample_rate", [1.0, 8.0])
def test_ring_aliases_preserve_cyclic_observation(
    indices: list[int], sample_rate: float
) -> None:
    """Signed reindexing preserves phase, half-turn direction and bounded quality."""
    signal = np.array(indices, dtype=np.int64)
    original = signal.copy()
    extractor = SymbolicExtractor(
        n_states=4, node_id="cycle", initial_transition_quality=0.8
    )
    states = extractor.extract(signal, sample_rate)
    np.testing.assert_array_equal(signal, original)
    assert [state.theta for state in states] == pytest.approx(
        [3 * np.pi / 2, 0.0, np.pi / 2, 3 * np.pi / 2], abs=1e-12
    )
    assert [state.omega for state in states] == pytest.approx(
        np.array([0.0, np.pi / 2, np.pi / 2, -np.pi]) * sample_rate
    )
    assert [state.quality for state in states] == pytest.approx([0.8, 1.0, 1.0, 0.75])
    assert extractor.quality_score(states) == pytest.approx(0.8875)
    assert all(
        state.amplitude == 1.0 and state.channel == "S" and state.node_id == "cycle"
        for state in states
    )


@pytest.mark.parametrize("mode", ["ring", "graph"])
@pytest.mark.parametrize("label", [2**53 + 1, -(2**62) + 3])
def test_singleton_large_labels_preserve_integer_residues(
    mode: str, label: int
) -> None:
    """Reduce int64 labels before floating-point mapping, including graph singletons."""
    signal = np.array([label], dtype=np.int64)
    states = SymbolicExtractor(n_states=4, mode=mode).extract(signal, 8.0)
    expected = np.pi / 2 if label > 0 else 3 * np.pi / 2
    assert len(states) == 1
    assert states[0].theta == pytest.approx(expected, abs=1e-12)
    assert states[0].omega == 0.0
    assert states[0].quality == 0.5
    assert signal[0] == label


def test_complete_ring_cycles_are_stalls() -> None:
    """Full positive and negative cycles retain the same observed state."""
    states = SymbolicExtractor(n_states=4).extract(np.array([0, 8, -4, 12]), 1.0)
    assert [state.theta for state in states] == pytest.approx([0.0] * 4)
    assert [state.omega for state in states] == pytest.approx([0.0] * 4)
    assert [state.quality for state in states] == pytest.approx([0.5, 0.2, 0.2, 0.2])


@pytest.mark.parametrize("mode", ["ring", "graph"])
def test_empty_symbolic_observation(mode: str) -> None:
    """Empty observations return no fabricated state or quality."""
    extractor = SymbolicExtractor(n_states=4, mode=mode)
    states = extractor.extract(np.array([], dtype=np.int64), 1.0)
    assert states == []
    assert extractor.quality_score(states) == 0.0


@pytest.mark.parametrize("mode", ["ring", "graph"])
def test_symbolic_states_drive_an_exact_uncoupled_engine(mode: str) -> None:
    """Exercise extracted signed frequencies through the public engine trajectory."""
    from scpn_phase_orchestrator.upde.engine import UPDEEngine
    from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

    if mode == "ring":
        states = SymbolicExtractor(n_states=4).extract(np.array([7, 0, 9, -1]), 8.0)
        expected = np.array([5 * np.pi / 8, 5 * np.pi / 4])
    else:
        states = SymbolicExtractor(n_states=8, mode="graph").extract(
            np.array([0, 2, 5, 1]), 8.0
        )
        expected = np.array([23 * np.pi / 18, 2 * np.pi / 9])
    phases = np.array([states[2].theta, states[3].theta])
    frequencies = np.array([states[2].omega, states[3].omega])
    engine = UPDEEngine(2, dt=0.03125)
    coupling = np.zeros((2, 2))
    lags = np.zeros((2, 2))
    actual = engine.step(phases, frequencies, coupling, 0.0, 0.0, lags)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    coherence, mean_phase = compute_order_parameter(actual)
    expected_vector = np.mean(np.exp(1j * expected))
    assert coherence == pytest.approx(abs(expected_vector))
    assert mean_phase == pytest.approx(float(np.angle(expected_vector)) % TWO_PI)


@pytest.mark.parametrize(
    ("labels", "expected_turns"),
    [
        ([-(2**62), 2**62, 0], [0.0, 2.0 / 3.0, 0.0]),
        ([-(2**63), 2**63 - 1, -(2**63)], [0.0, 0.5, 0.0]),
        ([-(2**63), 0, 2**63 - 1], [0.0, 0.5, 0.0]),
        ([-(2**63), 2**63 - 1, 0, -(2**63)], [0.0, 0.5, 0.75, 0.0]),
    ],
)
def test_graph_walk_full_signed_domain(
    labels: list[int], expected_turns: list[float]
) -> None:
    """Preserve full-span distances and totals beyond uint64 in public extraction."""
    signal = np.array(labels, dtype=np.int64)
    original = signal.copy()
    states = SymbolicExtractor(n_states=4, mode="graph").extract(signal, 8.0)
    expected = TWO_PI * np.array(expected_turns)
    expected_omega = np.concatenate(
        [[0.0], ((np.diff(expected) + np.pi) % TWO_PI - np.pi) * 8.0]
    )
    np.testing.assert_allclose([state.theta for state in states], expected, atol=1e-12)
    np.testing.assert_allclose(
        [state.omega for state in states], expected_omega, atol=1e-12
    )
    assert [state.quality for state in states] == pytest.approx(
        [0.5] + [0.1] * (len(labels) - 1)
    )
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("mode", ["ring", "graph"])
@pytest.mark.parametrize(
    "signal",
    [
        np.array([0, 9, 2, 9, 5, 9, 1, 9], dtype=np.int64)[::2],
        np.array([1, 5, 2, 0], dtype=np.int64)[::-1],
    ],
)
def test_symbolic_strided_views_preserve_logical_order(
    mode: str, signal: NDArray[np.int64]
) -> None:
    """Forward and reversed strides map the same logical labels without mutation."""
    original = signal.copy()
    states = SymbolicExtractor(n_states=8, mode=mode).extract(signal, 8.0)
    expected = (
        [0.0, np.pi / 2, 5 * np.pi / 4, np.pi / 4]
        if mode == "ring"
        else [0.0, 4 * np.pi / 9, 10 * np.pi / 9, 0.0]
    )
    np.testing.assert_allclose([state.theta for state in states], expected, atol=1e-12)
    assert [state.quality for state in states] == pytest.approx(
        [0.5, 0.875, 0.75, 0.625]
    )
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("mode", ["ring", "graph"])
def test_symbolic_unaligned_observations(mode: str) -> None:
    """Normalise an unaligned int64 buffer before extraction without rewriting it."""
    signal = np.ndarray((3,), dtype=np.int64, buffer=bytearray(25), offset=1)
    signal[:] = [0, 1, 0]
    original = signal.copy()
    states = SymbolicExtractor(n_states=4, mode=mode).extract(signal, 8.0)
    expected_middle = np.pi / 2 if mode == "ring" else np.pi
    assert [state.theta for state in states] == pytest.approx(
        [0.0, expected_middle, 0.0]
    )
    assert [state.quality for state in states] == pytest.approx([0.5, 1.0, 1.0])
    np.testing.assert_array_equal(signal, original)


def test_graph_walk_many_full_span_transitions() -> None:
    """Repeated full-span transitions retain every normalised prefix, not a clamp."""
    signal = np.tile(np.array([-(2**63), 2**63 - 1], dtype=np.int64), 129)[:-1]
    states = SymbolicExtractor(n_states=4, mode="graph").extract(signal, 8.0)
    expected = (TWO_PI * np.arange(257) / 256.0) % TWO_PI
    np.testing.assert_allclose([state.theta for state in states], expected, atol=1e-12)
    assert states[128].theta == pytest.approx(np.pi)
    assert [state.quality for state in states] == pytest.approx([0.5] + [0.1] * 256)


@pytest.mark.parametrize("mode", ["ring", "graph"])
def test_unsigned_full_span_labels_keep_their_values(mode: str) -> None:
    """A uint64 maximum is a full linear jump, not the signed label minus one."""
    signal = np.array([0, 2**64 - 1, 0], dtype=np.uint64)
    original = signal.copy()
    states = SymbolicExtractor(n_states=4, mode=mode).extract(signal, 8.0)
    middle = 3 * np.pi / 2 if mode == "ring" else np.pi
    quality = 1.0 if mode == "ring" else 0.1
    assert [state.theta for state in states] == pytest.approx([0.0, middle, 0.0])
    assert [state.quality for state in states] == pytest.approx([0.5, quality, quality])
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("mode", ["ring", "graph"])
@pytest.mark.parametrize("n_states", [2**31, 2**32, 2**63, 2**64 - 1, 2**64, 2**4096])
def test_large_vocabulary_preserves_public_integer_contract(
    mode: str, n_states: int
) -> None:
    """Vocabulary sizes beyond machine integers retain real Python phase mapping."""
    signal = np.array([0, 1, 0], dtype=np.int64)
    states = SymbolicExtractor(n_states=n_states, mode=mode).extract(signal, 8.0)
    middle = TWO_PI * (1 / n_states) if mode == "ring" else np.pi
    np.testing.assert_allclose(
        [state.theta for state in states], [0.0, middle, 0.0], rtol=1e-15, atol=0.0
    )
    assert [state.quality for state in states] == pytest.approx([0.5, 1.0, 1.0])


@pytest.mark.parametrize("mode", ["ring", "graph"])
@pytest.mark.parametrize("n_states", [2**64 - 1, 2**64, 2**4096])
def test_large_vocabulary_singleton_retains_signed_alias(
    mode: str, n_states: int
) -> None:
    """Graph singletons and ring observations share exact integer residue reduction."""
    signal = np.array([-1], dtype=np.int64)
    states = SymbolicExtractor(n_states=n_states, mode=mode).extract(signal, 8.0)
    expected = (TWO_PI * ((n_states - 1) / n_states)) % TWO_PI
    assert len(states) == 1
    assert states[0].theta == pytest.approx(expected, abs=0.0)
    assert states[0].omega == 0.0
    assert states[0].quality == 0.5


@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("mode", ["ring", "graph"])
def test_swapped_byte_order_preserves_public_labels(unsigned: bool, mode: str) -> None:
    """Public normalisation preserves labels with non-native byte order."""
    swapped: NDArray[np.int64] | NDArray[np.uint64]
    if unsigned:
        swapped = (
            np.array([0, 2, 5, 1], dtype=np.uint64)
            .byteswap()
            .view(np.dtype(np.uint64).newbyteorder("S"))
        )
    else:
        swapped = (
            np.array([0, 2, 5, 1], dtype=np.int64)
            .byteswap()
            .view(np.dtype(np.int64).newbyteorder("S"))
        )
    original = swapped.copy()
    states = SymbolicExtractor(n_states=8, mode=mode).extract(swapped, 8.0)
    expected = (
        [0.0, np.pi / 2, 5 * np.pi / 4, np.pi / 4]
        if mode == "ring"
        else [0.0, 4 * np.pi / 9, 10 * np.pi / 9, 0.0]
    )
    np.testing.assert_allclose([state.theta for state in states], expected, atol=1e-12)
    np.testing.assert_array_equal(swapped, original)


# Pipeline wiring: SymbolicExtractor → theta/omega → UPDEEngine
# → compute_order_parameter. Ring + graph modes, quality scoring,
# omega derivation from phase diffs. Performance: extract(1000)<5ms.
