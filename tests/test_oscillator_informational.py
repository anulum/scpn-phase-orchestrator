# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational oscillator tests

"""Exercise event-cadence extraction and its real phase-engine consumer."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators import informational as informational_module
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Phase extraction from event timestamps
# ---------------------------------------------------------------------------


class TestInformationalPhaseExtraction:
    """Verify event-derived phase, frequency and quality on the unit circle."""

    def test_omega_from_regular_10hz_events(self) -> None:
        """10 Hz events: median interval = 0.1s → ω = 2π·10 = 20π rad/s."""
        timestamps = np.arange(0.0, 1.0, 0.1)
        states = InformationalExtractor().extract(timestamps, sample_rate=0.0)
        expected_omega = TWO_PI * 10.0
        assert states[0].omega == pytest.approx(expected_omega, rel=0.01), (
            f"ω={states[0].omega:.1f}, expected ≈{expected_omega:.1f}"
        )

    def test_theta_from_cumulative_phase(self) -> None:
        """A regular train completes whole cycles, including the circular seam."""
        timestamps = np.arange(0.0, 1.0, 0.1)  # 0.0 to 0.9, duration=0.9
        states = InformationalExtractor().extract(timestamps, sample_rate=0.0)
        # f_median = 10 Hz, T = 0.9s → cumulative = 2π·10·0.9 = 18π → mod 2π = 0
        assert 0.0 <= states[0].theta < TWO_PI
        assert abs(np.exp(1j * states[0].theta) - 1.0) < 1e-12

    def test_theta_in_range_for_various_rates(self) -> None:
        """θ must be in [0, 2π) regardless of event rate."""
        for rate in [1.0, 5.0, 20.0, 100.0]:
            ts = np.arange(0.0, 2.0, 1.0 / rate)
            states = InformationalExtractor().extract(ts, sample_rate=0.0)
            assert 0.0 <= states[0].theta < TWO_PI, (
                f"rate={rate}: θ={states[0].theta} out of [0, 2π)"
            )

    def test_amplitude_is_mean_frequency(self) -> None:
        """Amplitude field stores mean instantaneous frequency."""
        ts = np.arange(0.0, 1.0, 0.1)  # intervals all = 0.1 → freq = 10 Hz
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        assert states[0].amplitude == pytest.approx(10.0, rel=0.01)


# ---------------------------------------------------------------------------
# Quality: interval regularity via inverse CV
# ---------------------------------------------------------------------------


class TestInformationalQuality:
    """Verify inverse interval variation distinguishes regular event trains."""

    def test_regular_events_high_quality(self) -> None:
        """Perfectly regular 10 Hz: CV=0 → quality = 1/(1+0) = 1.0."""
        ts = np.arange(0.0, 1.0, 0.1)
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        q = states[0].quality
        assert q > 0.99, f"Regular events → q≈1.0, got {q:.4f}"

    def test_irregular_events_lower_quality(self) -> None:
        """Random timestamps → higher CV → quality < 1."""
        rng = np.random.default_rng(123)
        ts = np.sort(rng.uniform(0, 10, size=50))
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        assert states[0].quality < 0.9, (
            f"Irregular events should have quality<0.9, got {states[0].quality:.4f}"
        )

    def test_quality_discriminates_regular_vs_irregular(self) -> None:
        """Regular events must score higher than irregular ones."""
        regular_ts = np.arange(0.0, 5.0, 0.1)
        rng = np.random.default_rng(0)
        irregular_ts = np.sort(rng.uniform(0, 5, size=50))

        ext = InformationalExtractor()
        q_regular = ext.extract(regular_ts, 0.0)[0].quality
        q_irregular = ext.extract(irregular_ts, 0.0)[0].quality
        assert q_regular > q_irregular, (
            f"Regular ({q_regular:.3f}) must exceed irregular ({q_irregular:.3f})"
        )

    def test_quality_in_unit_interval(self) -> None:
        """Quality must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            ts = np.sort(rng.uniform(0, 10, size=rng.integers(5, 100)))
            states = InformationalExtractor().extract(ts, sample_rate=0.0)
            assert 0.0 <= states[0].quality <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestInformationalEdgeCases:
    """Verify defined behaviour for degenerate inputs."""

    def test_single_timestamp_zero_everything(self) -> None:
        """Single event: no intervals → θ=0, ω=0, quality=0."""
        states = InformationalExtractor().extract(np.array([1.0]), sample_rate=0.0)
        assert states[0].theta == 0.0
        assert states[0].omega == 0.0
        assert states[0].quality == 0.0

    def test_identical_timestamps_zero_quality(self) -> None:
        """All-same timestamps: intervals all zero → quality=0."""
        states = InformationalExtractor().extract(np.array([1.0, 1.0, 1.0]), 0.0)
        assert states[0].quality == 0.0
        assert states[0].omega == 0.0

    def test_unsorted_timestamps_rejected(self) -> None:
        """Reject out-of-order timestamps rather than silently sorting them."""
        extractor = InformationalExtractor()
        with pytest.raises(ValueError, match="signal timestamps must be sorted"):
            extractor.extract(np.array([0.0, 0.3, 0.2, 0.5]), sample_rate=0.0)

    def test_two_timestamps_minimal(self) -> None:
        """Two timestamps = one interval → valid extraction."""
        states = InformationalExtractor().extract(np.array([0.0, 0.5]), 0.0)
        # interval=0.5 → freq=2 Hz → ω=4π, T=0.5, θ=(2π·2·0.5) mod 2π = 2π mod 2π = 0
        assert states[0].omega == pytest.approx(TWO_PI * 2.0, rel=0.01)
        assert states[0].quality > 0.0

    @pytest.mark.parametrize(
        "signal",
        [
            np.array([0.0, float("nan")]),
            np.array([0.0, float("inf")]),
            np.array([True, False]),
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            np.array(["0.0", "1.0"], dtype=object),
        ],
    )
    def test_extract_rejects_invalid_signal(self, signal: object) -> None:
        """Reject non-real or non-finite values at the public input boundary."""
        extractor = InformationalExtractor()
        with pytest.raises(ValueError, match="signal must be finite"):
            extractor.extract(cast(NDArray[np.float64], signal), sample_rate=0.0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestInformationalMetadata:
    """Verify channel assignment and quality_score aggregation."""

    def test_channel_is_I(self) -> None:
        """Carry the informational channel and caller's node identity."""
        ext = InformationalExtractor(node_id="info_x")
        states = ext.extract(np.arange(0.0, 1.0, 0.1), sample_rate=0.0)
        assert states[0].channel == "I"
        assert states[0].node_id == "info_x"

    @pytest.mark.parametrize("node_id", ["", "   ", 42, True])
    def test_invalid_node_id_rejected(self, node_id: object) -> None:
        """Reject blank or non-string node identities without coercion."""
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            InformationalExtractor(node_id=cast(str, node_id))

    def test_quality_score_empty(self) -> None:
        """An empty state collection has no usable quality."""
        assert InformationalExtractor().quality_score([]) == 0.0

    def test_quality_score_matches_single_state(self) -> None:
        """Aggregate one real extraction without changing its quality."""
        ext = InformationalExtractor()
        states = ext.extract(np.arange(0.0, 1.0, 0.1), sample_rate=0.0)
        score = ext.quality_score(states)
        assert score == states[0].quality


class TestInformationalPipelineEndToEnd:
    """InformationalExtractor → theta/omega → Engine → R.

    Proves InformationalExtractor is a functional input adapter.
    """

    def test_event_streams_feed_engine(self) -> None:
        """Multiple event streams → extract → engine → order parameter."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        ext = InformationalExtractor()
        rates = [5.0, 10.0, 15.0, 20.0]
        phases = []
        omegas = []
        for rate in rates:
            ts = np.arange(0.0, 2.0, 1.0 / rate)
            states = ext.extract(ts, sample_rate=0.0)
            phases.append(states[0].theta)
            omegas.append(states[0].omega)
        phases_arr = np.array(phases)
        omegas_arr = np.array(omegas)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.001)
        for _ in range(200):
            phases_arr = eng.step(phases_arr, omegas_arr, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases_arr)
        assert 0.0 <= r <= 1.0
        assert np.all(phases_arr >= 0.0)
        assert np.all(phases_arr < TWO_PI)

    def test_quality_gates_engine_input(self) -> None:
        """Low-quality extraction should still produce valid engine input."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        ext = InformationalExtractor()
        rng = np.random.default_rng(42)
        phases_list = []
        for _ in range(4):
            ts = np.sort(rng.uniform(0, 5, size=rng.integers(5, 50)))
            states = ext.extract(ts, sample_rate=0.0)
            assert 0.0 <= states[0].theta < TWO_PI
            phases_list.append(states[0].theta)
        r, _ = compute_order_parameter(np.array(phases_list))
        assert 0.0 <= r <= 1.0

    def test_performance_extract_100_timestamps_under_600us(self) -> None:
        """InformationalExtractor.extract(100 timestamps) < 600μs."""
        import time

        ts = np.arange(0.0, 10.0, 0.1)
        ext = InformationalExtractor()
        ext.extract(ts, sample_rate=0.0)  # warm-up
        t0 = time.perf_counter()
        for _ in range(1000):
            ext.extract(ts, sample_rate=0.0)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 6e-4, f"extract(100) took {elapsed * 1e6:.0f}μs"


# Pipeline wiring: InformationalExtractor → theta/omega → UPDEEngine
# → compute_order_parameter. Event timestamp input, quality-gated.
# Performance: extract(100)<600μs.


class TestInformationalCadenceContracts:
    """Pin analytical cadence results through the available real backend."""

    @pytest.mark.parametrize(
        ("timestamps", "frequency", "amplitude", "theta", "quality"),
        [
            (np.array([0.0, 0.125, 0.25, 0.375]), 8.0, 8.0, 0.0, 1.0),
            (np.array([0.0, 0.125, 0.375]), 6.0, 6.0, np.pi / 2, 0.75),
            (
                np.array([0.0, 0.125, 0.375, 0.875]),
                4.0,
                14.0 / 3.0,
                np.pi,
                7.0 / (7.0 + np.sqrt(14.0)),
            ),
            (
                np.array([0.0, 0.0, 0.125, 0.125, 0.375, 0.375]),
                6.0,
                6.0,
                np.pi / 2,
                0.75,
            ),
        ],
        ids=["regular", "even-intervals", "odd-intervals", "duplicates"],
    )
    def test_real_cadence_matches_analytical_values(
        self,
        timestamps: NDArray[np.float64],
        frequency: float,
        amplitude: float,
        theta: float,
        quality: float,
    ) -> None:
        """Use independent exact interval oracles, not a synthetic backend result."""
        original = timestamps.copy()
        extractor = InformationalExtractor(node_id="cadence")
        states = extractor.extract(timestamps, sample_rate=0.0)
        assert len(states) == 1
        state = states[0]
        assert state.omega == pytest.approx(TWO_PI * frequency, abs=1e-12)
        assert state.amplitude == pytest.approx(amplitude, abs=1e-12)
        assert abs(np.exp(1j * state.theta) - np.exp(1j * theta)) < 1e-12
        assert state.quality == pytest.approx(quality, abs=1e-12)
        assert 0.0 <= state.theta < TWO_PI
        assert state.channel == "I"
        assert state.node_id == "cadence"
        assert extractor.quality_score(states) == state.quality
        np.testing.assert_array_equal(timestamps, original)

    @pytest.mark.parametrize(("origin", "scale"), [(0.0, 2.0), (8.0, 1.0), (8.0, 0.5)])
    def test_time_translation_and_scaling(self, origin: float, scale: float) -> None:
        """Time origin preserves phase; time dilation rescales frequency only."""
        timestamps = origin + scale * np.array([0.0, 0.125, 0.375])
        state = InformationalExtractor().extract(timestamps, sample_rate=123.0)[0]
        assert state.omega == pytest.approx(TWO_PI * 6.0 / scale, abs=1e-12)
        assert state.amplitude == pytest.approx(6.0 / scale, abs=1e-12)
        assert state.theta == pytest.approx(np.pi / 2, abs=1e-12)
        assert state.quality == pytest.approx(0.75, abs=1e-12)

    def test_cadence_drives_exact_uncoupled_engine_trajectory(self) -> None:
        """Propagate real extracted states and verify their observable synchrony."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        extractor = InformationalExtractor()
        states = [
            extractor.extract(np.array([0.0, 0.125, 0.375]), 0.0)[0],
            extractor.extract(np.array([0.0, 0.125, 0.375, 0.875]), 0.0)[0],
        ]
        phases = np.array([state.theta for state in states])
        frequencies = np.array([state.omega for state in states])
        engine = UPDEEngine(2, dt=1.0 / 64.0)
        coupling = np.zeros((2, 2))
        lag = np.zeros((2, 2))
        for _ in range(16):
            phases = engine.step(phases, frequencies, coupling, 0.0, 0.0, lag)
        np.testing.assert_allclose(phases, [3.0 * np.pi / 2, np.pi], atol=1e-12, rtol=0)
        coherence, mean_phase = compute_order_parameter(phases)
        assert coherence == pytest.approx(np.sqrt(0.5), abs=1e-12)
        assert mean_phase == pytest.approx(5.0 * np.pi / 4, abs=1e-12)


class TestInformationalKernelFailure:
    """Retain only deterministic failure injection at the optional-kernel boundary."""

    def test_extract_falls_back_to_python_when_rust_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Continue extracting when a kernel fails, not test its numerical success.

        A healthy real kernel cannot deterministically raise on valid timestamps.
        This existing exception injection exercises the documented fallback only;
        successful cadence and engine contracts use unmodified real backends.
        """

        def _raising_event_phase(
            _signal: NDArray[np.float64],
        ) -> tuple[float, float, float]:
            """Represent an optional-kernel failure, never a numerical success."""
            raise RuntimeError("boom")

        monkeypatch.setattr(
            informational_module, "_rust_event_phase", _raising_event_phase
        )
        timestamps = np.array([0.0, 0.125, 0.375])
        original = timestamps.copy()
        extractor = InformationalExtractor(node_id="recovering")
        state = extractor.extract(timestamps, 0.0)[0]
        assert state.omega == pytest.approx(TWO_PI * 6.0, abs=1e-12)
        assert state.theta == pytest.approx(np.pi / 2, abs=1e-12)
        assert state.amplitude == pytest.approx(6.0, abs=1e-12)
        assert state.quality == pytest.approx(0.75, abs=1e-12)
        assert state.node_id == "recovering"
        assert state.channel == "I"
        np.testing.assert_array_equal(timestamps, original)
        subsequent = extractor.extract(np.array([0.0, 0.125, 0.25]), 0.0)[0]
        assert subsequent.omega == pytest.approx(TWO_PI * 8.0, abs=1e-12)
        assert subsequent.quality == pytest.approx(1.0, abs=1e-12)
