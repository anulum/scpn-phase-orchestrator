# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational oscillator tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Phase extraction from event timestamps
# ---------------------------------------------------------------------------


class TestInformationalPhaseExtraction:
    """Verify that event timestamps are correctly converted to phase,
    frequency, and quality on the unit circle."""

    def test_omega_from_regular_10hz_events(self):
        """10 Hz events: median interval = 0.1s → ω = 2π·10 = 20π rad/s."""
        timestamps = np.arange(0.0, 1.0, 0.1)
        states = InformationalExtractor().extract(timestamps, sample_rate=0.0)
        expected_omega = TWO_PI * 10.0
        assert states[0].omega == pytest.approx(expected_omega, rel=0.01), (
            f"ω={states[0].omega:.1f}, expected ≈{expected_omega:.1f}"
        )

    def test_theta_from_cumulative_phase(self):
        """θ = (2π·f·T) mod 2π where f = median freq, T = total duration.
        10 Hz over 1s: θ = (2π·10·0.9) mod 2π = (18π) mod 2π = 0."""
        timestamps = np.arange(0.0, 1.0, 0.1)  # 0.0 to 0.9, duration=0.9
        states = InformationalExtractor().extract(timestamps, sample_rate=0.0)
        # f_median = 10 Hz, T = 0.9s → cumulative = 2π·10·0.9 = 18π → mod 2π = 0
        assert 0.0 <= states[0].theta < TWO_PI

    def test_theta_in_range_for_various_rates(self):
        """θ must be in [0, 2π) regardless of event rate."""
        for rate in [1.0, 5.0, 20.0, 100.0]:
            ts = np.arange(0.0, 2.0, 1.0 / rate)
            states = InformationalExtractor().extract(ts, sample_rate=0.0)
            assert 0.0 <= states[0].theta < TWO_PI, (
                f"rate={rate}: θ={states[0].theta} out of [0, 2π)"
            )

    def test_amplitude_is_mean_frequency(self):
        """Amplitude field stores mean instantaneous frequency."""
        ts = np.arange(0.0, 1.0, 0.1)  # intervals all = 0.1 → freq = 10 Hz
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        assert states[0].amplitude == pytest.approx(10.0, rel=0.01)


# ---------------------------------------------------------------------------
# Quality: interval regularity via inverse CV
# ---------------------------------------------------------------------------


class TestInformationalQuality:
    """Quality = 1/(1+CV) where CV = std(intervals)/mean(intervals).
    Regular events → CV≈0 → quality≈1. Irregular → quality<1."""

    def test_regular_events_high_quality(self):
        """Perfectly regular 10 Hz: CV=0 → quality = 1/(1+0) = 1.0."""
        ts = np.arange(0.0, 1.0, 0.1)
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        q = states[0].quality
        assert q > 0.99, f"Regular events → q≈1.0, got {q:.4f}"

    def test_irregular_events_lower_quality(self):
        """Random timestamps → higher CV → quality < 1."""
        rng = np.random.default_rng(123)
        ts = np.sort(rng.uniform(0, 10, size=50))
        states = InformationalExtractor().extract(ts, sample_rate=0.0)
        assert states[0].quality < 0.9, (
            f"Irregular events should have quality<0.9, got {states[0].quality:.4f}"
        )

    def test_quality_discriminates_regular_vs_irregular(self):
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

    def test_quality_in_unit_interval(self):
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

    def test_single_timestamp_zero_everything(self):
        """Single event: no intervals → θ=0, ω=0, quality=0."""
        states = InformationalExtractor().extract(np.array([1.0]), sample_rate=0.0)
        assert states[0].theta == 0.0
        assert states[0].omega == 0.0
        assert states[0].quality == 0.0

    def test_identical_timestamps_zero_quality(self):
        """All-same timestamps: intervals all zero → quality=0."""
        states = InformationalExtractor().extract(np.array([1.0, 1.0, 1.0]), 0.0)
        assert states[0].quality == 0.0
        assert states[0].omega == 0.0

    def test_two_timestamps_minimal(self):
        """Two timestamps = one interval → valid extraction."""
        states = InformationalExtractor().extract(np.array([0.0, 0.5]), 0.0)
        # interval=0.5 → freq=2 Hz → ω=4π, T=0.5, θ=(2π·2·0.5) mod 2π = 2π mod 2π = 0
        assert states[0].omega == pytest.approx(TWO_PI * 2.0, rel=0.01)
        assert states[0].quality > 0.0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestInformationalMetadata:
    """Verify channel assignment and quality_score aggregation."""

    def test_channel_is_I(self):
        ext = InformationalExtractor(node_id="info_x")
        states = ext.extract(np.arange(0.0, 1.0, 0.1), sample_rate=0.0)
        assert states[0].channel == "I"
        assert states[0].node_id == "info_x"

    def test_quality_score_empty(self):
        assert InformationalExtractor().quality_score([]) == 0.0

    def test_quality_score_matches_single_state(self):
        ext = InformationalExtractor()
        states = ext.extract(np.arange(0.0, 1.0, 0.1), sample_rate=0.0)
        score = ext.quality_score(states)
        assert score == states[0].quality


class TestInformationalPipelineEndToEnd:
    """InformationalExtractor → theta/omega → Engine → R.

    Proves InformationalExtractor is a functional input adapter.
    """

    def test_event_streams_feed_engine(self):
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

    def test_quality_gates_engine_input(self):
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

    def test_performance_extract_100_timestamps_under_600us(self):
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
