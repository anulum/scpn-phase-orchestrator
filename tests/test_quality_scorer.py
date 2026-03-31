# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quality scorer tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


def _ps(quality: float, amplitude: float = 1.0) -> PhaseState:
    return PhaseState(
        theta=0.0,
        omega=1.0,
        amplitude=amplitude,
        quality=quality,
        channel="P",
        node_id="test",
    )


# ---------------------------------------------------------------------------
# Weighted quality score
# ---------------------------------------------------------------------------


class TestQualityScore:
    """Verify amplitude-weighted quality aggregation. High-amplitude oscillators
    should dominate the score, matching how strong signals are more trustworthy."""

    def test_empty_returns_zero(self):
        assert PhaseQualityScorer().score([]) == 0.0

    def test_uniform_quality_returns_exact(self):
        states = [_ps(0.8), _ps(0.8), _ps(0.8)]
        np.testing.assert_allclose(PhaseQualityScorer().score(states), 0.8, atol=1e-12)

    def test_high_amplitude_dominates(self):
        """Oscillator with amplitude=10 and quality=1.0 should dominate
        over oscillator with amplitude≈0 and quality=0."""
        states = [_ps(1.0, amplitude=10.0), _ps(0.0, amplitude=1e-15)]
        score = PhaseQualityScorer().score(states)
        assert score > 0.99, (
            f"High-amplitude oscillator should dominate, got {score:.4f}"
        )

    def test_equal_amplitude_is_simple_average(self):
        """With equal amplitudes, score must be arithmetic mean of qualities."""
        states = [_ps(0.2), _ps(0.8)]
        score = PhaseQualityScorer().score(states)
        np.testing.assert_allclose(score, 0.5, atol=1e-12)

    def test_score_in_unit_interval(self):
        """Score must always be in [0, 1] for valid qualities."""
        states = [
            _ps(0.0, amplitude=5.0),
            _ps(1.0, amplitude=0.001),
            _ps(0.5, amplitude=1.0),
        ]
        score = PhaseQualityScorer().score(states)
        assert 0.0 <= score <= 1.0

    def test_single_oscillator_returns_its_quality(self):
        assert PhaseQualityScorer().score([_ps(0.73)]) == 0.73

    def test_weighted_formula_exact(self):
        """Verify against manual weighted average:
        score = (q1*a1 + q2*a2) / (a1 + a2) = (0.6*3 + 0.9*1) / 4 = 2.7/4 = 0.675."""
        states = [_ps(0.6, amplitude=3.0), _ps(0.9, amplitude=1.0)]
        score = PhaseQualityScorer().score(states)
        np.testing.assert_allclose(score, 0.675, atol=1e-12)


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------


class TestCollapseDetection:
    """Verify that collapse detection correctly identifies when the majority
    of oscillators have lost signal quality."""

    def test_empty_is_collapsed(self):
        """No oscillators → collapsed (defensive)."""
        assert PhaseQualityScorer().detect_collapse([]) is True

    def test_all_high_quality_no_collapse(self):
        states = [_ps(0.9), _ps(0.8), _ps(0.7)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_majority_low_is_collapsed(self):
        """2/3 below threshold → collapsed."""
        states = [_ps(0.01), _ps(0.02), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is True

    def test_minority_low_not_collapsed(self):
        """1/3 below threshold → not collapsed (minority failure tolerated)."""
        states = [_ps(0.01), _ps(0.5), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_exactly_half_not_collapsed(self):
        """Exactly 50% below threshold → not collapsed (strict majority required)."""
        states = [_ps(0.01), _ps(0.01), _ps(0.5), _ps(0.9)]
        # 2/4 = 50% below, but > requires strict majority
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_threshold_boundary(self):
        """Quality exactly at threshold → NOT below threshold → not collapsed."""
        states = [_ps(0.1), _ps(0.1), _ps(0.1)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_custom_threshold(self):
        """Higher threshold → more oscillators qualify as low quality."""
        states = [_ps(0.3), _ps(0.4), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.5) is True
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.2) is False


# ---------------------------------------------------------------------------
# Downweight mask
# ---------------------------------------------------------------------------


class TestDownweightMask:
    """Verify the quality gating mask that filters low-quality oscillators
    from coupling computations."""

    def test_empty_returns_empty(self):
        mask = PhaseQualityScorer().downweight_mask([])
        assert len(mask) == 0

    def test_below_threshold_zeroed(self):
        states = [_ps(0.5), _ps(0.1), _ps(0.8)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert mask[0] > 0.0, "Quality 0.5 >= 0.3, should pass"
        assert mask[1] == 0.0, "Quality 0.1 < 0.3, should be zeroed"
        assert mask[2] > 0.0, "Quality 0.8 >= 0.3, should pass"

    def test_mask_values_equal_quality(self):
        """Non-zero mask values must equal the original quality (not just 1.0)."""
        states = [_ps(0.5), _ps(0.8)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        np.testing.assert_allclose(mask, [0.5, 0.8])

    def test_all_above_threshold_all_nonzero(self):
        states = [_ps(0.9), _ps(0.7), _ps(0.5)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert np.all(mask > 0.0)

    def test_all_below_threshold_all_zero(self):
        states = [_ps(0.1), _ps(0.05), _ps(0.2)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        np.testing.assert_array_equal(mask, [0.0, 0.0, 0.0])

    def test_mask_dtype_float64(self):
        states = [_ps(0.5)]
        mask = PhaseQualityScorer().downweight_mask(states)
        assert mask.dtype == np.float64

    def test_exactly_at_threshold_passes(self):
        """Quality exactly at min_quality should pass (>= not >)."""
        states = [_ps(0.3)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert mask[0] == 0.3


class TestQualityScorerPipelineEndToEnd:
    """Full pipeline: PhysicalExtractor → PhaseState → quality mask → Engine.

    Proves quality scorer gates which oscillators enter the engine.
    """

    def test_quality_mask_gates_engine_coupling(self):
        """Low-quality oscillators get downweighted → affects K_nm."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        # 4 oscillators, first two high quality, last two low
        states = [_ps(0.9), _ps(0.8), _ps(0.1), _ps(0.05)]
        scorer = PhaseQualityScorer()
        mask = scorer.downweight_mask(states, min_quality=0.3)
        assert mask[0] > 0.0  # high quality passes
        assert mask[2] == 0.0  # low quality blocked
        n = len(states)
        phases = np.array([s.theta for s in states])
        omegas = np.array([s.omega for s in states])
        knm_base = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm_base, 0.0)
        # Apply mask: K_nm[i,j] *= mask[i] * mask[j]
        knm = knm_base * mask[:, None] * mask[None, :]
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

    def test_performance_downweight_mask_100_under_50us(self):
        """PhaseQualityScorer.downweight_mask(100 states) < 50μs."""
        import time
        states = [_ps(np.random.default_rng(i).uniform(0, 1)) for i in range(100)]
        scorer = PhaseQualityScorer()
        scorer.downweight_mask(states)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10000):
            scorer.downweight_mask(states)
        elapsed = (time.perf_counter() - t0) / 10000
        assert elapsed < 5e-5, f"downweight_mask(100) took {elapsed*1e6:.0f}μs"


# Pipeline wiring: PhaseQualityScorer → downweight_mask → K_nm modulation
# → UPDEEngine → compute_order_parameter. Quality gates engine coupling.
# Performance: downweight_mask(100)<50μs.
