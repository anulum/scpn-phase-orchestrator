# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quality scorer tests

"""Exercise public aggregation, collapse, masking and real engine integration."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


def _ps(quality: float, amplitude: float = 1.0) -> PhaseState:
    """Build an admitted public measurement record for the stated quality."""
    return PhaseState(
        theta=0.0,
        omega=1.0,
        amplitude=amplitude,
        quality=quality,
        channel="P",
        node_id="test",
    )


class TestQualityScore:
    """Verify finite amplitude-weighted quality through the public scorer."""

    def test_empty_returns_zero(self) -> None:
        """An empty extraction returns zero confidence."""
        assert PhaseQualityScorer().score([]) == 0.0

    def test_uniform_quality_returns_exact(self) -> None:
        """Equal qualities retain their value under positive equal weights."""
        states = [_ps(0.8), _ps(0.8), _ps(0.8)]
        np.testing.assert_allclose(PhaseQualityScorer().score(states), 0.8, atol=1e-12)

    def test_high_amplitude_dominates(self) -> None:
        """A stronger signal dominates a weight below the amplitude floor."""
        states = [_ps(1.0, amplitude=10.0), _ps(0.0, amplitude=1e-15)]
        score = PhaseQualityScorer().score(states)
        assert score > 0.99, (
            f"High-amplitude oscillator should dominate, got {score:.4f}"
        )

    def test_equal_amplitude_is_simple_average(self) -> None:
        """With equal amplitudes, score must be arithmetic mean of qualities."""
        states = [_ps(0.2), _ps(0.8)]
        score = PhaseQualityScorer().score(states)
        np.testing.assert_allclose(score, 0.5, atol=1e-12)

    def test_score_in_unit_interval(self) -> None:
        """Score must always be in [0, 1] for valid qualities."""
        states = [
            _ps(0.0, amplitude=5.0),
            _ps(1.0, amplitude=0.001),
            _ps(0.5, amplitude=1.0),
        ]
        score = PhaseQualityScorer().score(states)
        assert 0.0 <= score <= 1.0

    def test_single_oscillator_returns_its_quality(self) -> None:
        """One usable oscillator retains its extraction quality."""
        assert PhaseQualityScorer().score([_ps(0.73)]) == 0.73

    def test_weighted_formula_exact(self) -> None:
        """Weights 3:1 give the analytical mean 0.675 for qualities 0.6:0.9."""
        states = [_ps(0.6, amplitude=3.0), _ps(0.9, amplitude=1.0)]
        score = PhaseQualityScorer().score(states)
        np.testing.assert_allclose(score, 0.675, atol=1e-12)


class TestCollapseDetection:
    """Verify strict-majority collapse at configured and override thresholds."""

    def test_empty_is_collapsed(self) -> None:
        """No oscillators → collapsed (defensive)."""
        assert PhaseQualityScorer().detect_collapse([]) is True

    def test_all_high_quality_no_collapse(self) -> None:
        """Qualities above the threshold prevent collapse."""
        states = [_ps(0.9), _ps(0.8), _ps(0.7)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_majority_low_is_collapsed(self) -> None:
        """2/3 below threshold → collapsed."""
        states = [_ps(0.01), _ps(0.02), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is True

    def test_minority_low_not_collapsed(self) -> None:
        """1/3 below threshold → not collapsed (minority failure tolerated)."""
        states = [_ps(0.01), _ps(0.5), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_exactly_half_not_collapsed(self) -> None:
        """Exactly 50% below threshold → not collapsed (strict majority required)."""
        states = [_ps(0.01), _ps(0.01), _ps(0.5), _ps(0.9)]
        # 2/4 = 50% below, but > requires strict majority
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_threshold_boundary(self) -> None:
        """Quality exactly at threshold → NOT below threshold → not collapsed."""
        states = [_ps(0.1), _ps(0.1), _ps(0.1)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False

    def test_custom_threshold(self) -> None:
        """Higher threshold → more oscillators qualify as low quality."""
        states = [_ps(0.3), _ps(0.4), _ps(0.9)]
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.5) is True
        assert PhaseQualityScorer().detect_collapse(states, threshold=0.2) is False


class TestDownweightMask:
    """Verify the public quality mask used in coupling computations."""

    def test_empty_returns_empty(self) -> None:
        """Empty extraction returns an empty mask."""
        mask = PhaseQualityScorer().downweight_mask([])
        assert len(mask) == 0

    def test_below_threshold_zeroed(self) -> None:
        """Only qualities below the selected boundary lose their weight."""
        states = [_ps(0.5), _ps(0.1), _ps(0.8)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert mask[0] > 0.0, "Quality 0.5 >= 0.3, should pass"
        assert mask[1] == 0.0, "Quality 0.1 < 0.3, should be zeroed"
        assert mask[2] > 0.0, "Quality 0.8 >= 0.3, should pass"

    def test_mask_values_equal_quality(self) -> None:
        """Non-zero mask values must equal the original quality (not just 1.0)."""
        states = [_ps(0.5), _ps(0.8)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        np.testing.assert_allclose(mask, [0.5, 0.8])

    def test_all_above_threshold_all_nonzero(self) -> None:
        """Every passing quality contributes its own positive weight."""
        states = [_ps(0.9), _ps(0.7), _ps(0.5)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert np.all(mask > 0.0)

    def test_all_below_threshold_all_zero(self) -> None:
        """A rejected cohort contributes no coupling weight."""
        states = [_ps(0.1), _ps(0.05), _ps(0.2)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        np.testing.assert_array_equal(mask, [0.0, 0.0, 0.0])

    def test_mask_dtype_float64(self) -> None:
        """Public masking returns the documented float64 array."""
        states = [_ps(0.5)]
        mask = PhaseQualityScorer().downweight_mask(states)
        assert mask.dtype == np.float64

    def test_exactly_at_threshold_passes(self) -> None:
        """Quality exactly at min_quality should pass (>= not >)."""
        states = [_ps(0.3)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert mask[0] == 0.3


class TestQualityScorerPipelineEndToEnd:
    """Full pipeline: PhysicalExtractor → PhaseState → quality mask → Engine.

    Proves quality scorer gates which oscillators enter the engine.
    """

    def test_quality_mask_gates_engine_coupling(self) -> None:
        """Extract waveforms, gate silence and verify the coupled phase trajectory."""
        from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        sample_rate = 128.0
        time = np.arange(128, dtype=np.float64) / sample_rate
        states = [
            PhysicalExtractor(node_id="high-a").extract(
                np.cos(8.0 * np.pi * time), sample_rate
            )[0],
            PhysicalExtractor(node_id="high-b").extract(
                np.cos(8.0 * np.pi * time + 1.0), sample_rate
            )[0],
            PhysicalExtractor(node_id="silent").extract(
                np.zeros(128, dtype=np.float64), sample_rate
            )[0],
        ]
        scorer = PhaseQualityScorer()
        mask = scorer.downweight_mask(states, min_quality=0.3)
        np.testing.assert_allclose(mask, [1.0, 1.0, 0.0], atol=1e-12)
        n = len(states)
        initial = np.array([s.theta for s in states])
        phases = initial.copy()
        ungated = initial.copy()
        omegas = np.array([s.omega for s in states])
        knm_base = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm_base, 0.0)
        knm = knm_base * mask[:, None] * mask[None, :]
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            ungated = eng.step(ungated, omegas, knm_base, 0.0, 0.0, alpha)
        delta = float(
            np.arctan2(np.sin(phases[1] - phases[0]), np.cos(phases[1] - phases[0]))
        )
        expected = 2.0 * np.arctan(np.tan(0.5) * np.exp(-1.0))
        assert delta == pytest.approx(expected, abs=1e-8)
        assert phases[2] == initial[2]
        assert np.linalg.norm(np.sin(phases - ungated)) > 0.01

    def test_performance_downweight_mask_100_under_50us(self) -> None:
        """PhaseQualityScorer.downweight_mask(100 states) < 50μs."""
        import time

        states = [_ps(np.random.default_rng(i).uniform(0, 1)) for i in range(100)]
        scorer = PhaseQualityScorer()
        scorer.downweight_mask(states)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10000):
            scorer.downweight_mask(states)
        elapsed = (time.perf_counter() - t0) / 10000
        assert elapsed < 5e-5, f"downweight_mask(100) took {elapsed * 1e6:.0f}μs"


class TestRustQualityDispatch:
    """Verify configured and override results in each actual runtime."""

    def test_configured_thresholds_match_real_measurements(self) -> None:
        """Matching thresholds retain the real weighted mean and strict majority."""
        scorer = PhaseQualityScorer(collapse_threshold=0.7, min_quality=0.4)
        states = [_ps(0.1, 2.0), _ps(0.6, 1.0), _ps(0.9, 3.0)]

        assert scorer.score(states) == pytest.approx(3.5 / 6.0)
        assert scorer.detect_collapse(states, threshold=0.7) is True
        np.testing.assert_allclose(
            scorer.downweight_mask(states, min_quality=0.4), [0.0, 0.6, 0.9]
        )

    def test_override_thresholds_stay_in_python(self) -> None:
        """Overrides change both decisions relative to the configured thresholds."""
        scorer = PhaseQualityScorer(collapse_threshold=0.7, min_quality=0.4)
        states = [_ps(0.1), _ps(0.6), _ps(0.9)]

        assert scorer.detect_collapse(states, threshold=0.5) is False
        np.testing.assert_allclose(
            scorer.downweight_mask(states, min_quality=0.8), [0.0, 0.0, 0.9], atol=1e-12
        )


class TestQualityValidation:
    """Invalid thresholds and inputs fail closed with explicit errors."""

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_init_rejects_non_finite_collapse_threshold(self, bad: float) -> None:
        """A nonfinite collapse configuration is refused before scoring.

        Parameters
        ----------
        bad : float
            Nonfinite or outside-unit-interval configuration under test.
        """
        with pytest.raises(ValueError, match="collapse_threshold must be finite"):
            PhaseQualityScorer(collapse_threshold=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_init_rejects_non_finite_min_quality(self, bad: float) -> None:
        """A nonfinite mask configuration is refused before scoring.

        Parameters
        ----------
        bad : float
            Nonfinite or outside-unit-interval configuration under test.
        """
        with pytest.raises(ValueError, match="min_quality must be finite"):
            PhaseQualityScorer(min_quality=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_init_rejects_collapse_threshold_out_of_range(self, bad: float) -> None:
        """Collapse configuration must remain in the unit interval.

        Parameters
        ----------
        bad : float
            Nonfinite or outside-unit-interval configuration under test.
        """
        with pytest.raises(ValueError, match=r"collapse_threshold must be in \[0, 1\]"):
            PhaseQualityScorer(collapse_threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_init_rejects_min_quality_out_of_range(self, bad: float) -> None:
        """Mask configuration must remain in the unit interval.

        Parameters
        ----------
        bad : float
            Nonfinite or outside-unit-interval configuration under test.
        """
        with pytest.raises(ValueError, match=r"min_quality must be in \[0, 1\]"):
            PhaseQualityScorer(min_quality=bad)

    def test_detect_collapse_rejects_non_finite_threshold(self) -> None:
        """A nonfinite per-call collapse threshold is refused."""
        with pytest.raises(ValueError, match="threshold must be finite"):
            PhaseQualityScorer().detect_collapse([_ps(0.5)], threshold=float("inf"))

    def test_detect_collapse_rejects_out_of_range_threshold(self) -> None:
        """An out-of-domain collapse override is refused."""
        with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
            PhaseQualityScorer().detect_collapse([_ps(0.5)], threshold=1.5)

    def test_downweight_mask_rejects_non_finite_min_quality(self) -> None:
        """A nonfinite mask override is refused."""
        with pytest.raises(ValueError, match="min_quality must be finite"):
            PhaseQualityScorer().downweight_mask([_ps(0.5)], min_quality=float("nan"))

    def test_downweight_mask_rejects_out_of_range_min_quality(self) -> None:
        """An out-of-domain mask override is refused."""
        with pytest.raises(ValueError, match=r"min_quality must be in \[0, 1\]"):
            PhaseQualityScorer().downweight_mask([_ps(0.5)], min_quality=-0.5)
