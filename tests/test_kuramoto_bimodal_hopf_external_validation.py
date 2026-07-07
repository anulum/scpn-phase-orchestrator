# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bimodal Kuramoto Hopf external-validation unit tests

"""Unit tests for the bimodal Kuramoto Hopf eigenvalue external validation.

These exercise the pure pieces — the analytic eigenvalue, critical coupling and Hopf
frequency, the correlation-with-slope, the bimodal frequency draw, the ringdown,
the envelope/autocorrelation family reads, the decay-window and step helpers, and the
record/verdict/payload assembly — on small deterministic inputs. The sealed-artefact
numbers are pinned separately by the evidence integrity test.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.kuramoto_bimodal_hopf_external_validation import (
    BENCHMARK,
    FAMILIES,
    autocorrelation_family_rate,
    bimodal_frequencies,
    correlation,
    critical_coupling,
    decaying_window_end,
    eigenvalue_real,
    envelope_family_rate,
    hopf_frequency,
    kuramoto_bimodal_payload,
    kuramoto_bimodal_record,
    kuramoto_bimodal_verdict,
    measured_frequency,
    ringdown_steps,
    simulate_bimodal_ringdown,
    sweep_bimodal_coupling,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


class TestAnalyticGroundTruth:
    def test_critical_coupling_is_four_delta(self) -> None:
        assert critical_coupling(0.5) == pytest.approx(2.0)

    def test_eigenvalue_real_crosses_zero_at_critical_coupling(self) -> None:
        delta = 0.5
        k_c = critical_coupling(delta)
        assert eigenvalue_real(k_c, delta) == pytest.approx(0.0)
        assert eigenvalue_real(0.5 * k_c, delta) < 0.0

    def test_hopf_frequency_real_in_oscillatory_regime(self) -> None:
        # omega0 > K/4: the eigenvalue is complex, Omega = sqrt(omega0^2 - (K/4)^2)
        assert hopf_frequency(2.0, 1.5) == pytest.approx(np.sqrt(1.5**2 - 0.5**2))

    def test_hopf_frequency_zero_outside_oscillatory_regime(self) -> None:
        # omega0 < K/4: radicand negative, clipped to a real (non-oscillatory) zero
        assert hopf_frequency(20.0, 0.5) == 0.0


class TestCorrelation:
    def test_perfect_line_has_unit_slope(self) -> None:
        lam = [-0.35, -0.25, -0.15, -0.05]
        result = correlation(lam, lam)
        assert result["slope"] == pytest.approx(1.0)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["mean_abs_gap"] == pytest.approx(0.0, abs=1e-9)
        assert result["n"] == 4

    def test_confounded_channel_has_steep_slope(self) -> None:
        lam = [-0.35, -0.25, -0.15, -0.05]
        steep = [6.0 * v for v in lam]
        result = correlation(lam, steep)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["slope"] == pytest.approx(6.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            correlation([-0.3, -0.2], [-0.3, -0.2, -0.1])

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            correlation([-0.3, -0.2], [-0.3, -0.2])


class TestBimodalFrequencies:
    def test_two_populations_split_by_sign(self) -> None:
        omega = bimodal_frequencies(200, 1.5, 0.5, seed=1)
        assert omega.shape == (200,)
        # the first half centres near +omega0, the second near -omega0
        assert np.median(omega[:100]) > 0.5
        assert np.median(omega[100:]) < -0.5

    @pytest.mark.parametrize("n", [0, 1, 3, 7])
    def test_odd_or_nonpositive_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match="even number"):
            bimodal_frequencies(n, 1.5, 0.5, seed=1)


class TestRingdown:
    def _omega(self) -> np.ndarray:
        return bimodal_frequencies(40, 1.5, 0.5, seed=2)

    def test_returns_global_and_subpopulation_fields(self) -> None:
        omega = self._omega()
        theta0 = 0.6 * np.random.default_rng(0).standard_normal(40)
        global_field, subpop = simulate_bimodal_ringdown(
            omega, 1.0, theta0, dt=0.02, n_steps=100, sample_every=5
        )
        assert global_field.shape == (20,)
        assert subpop.shape == (20,)
        assert global_field.dtype == np.complex128
        assert np.all(np.abs(subpop) <= 1.0 + 1e-9)

    def test_non_positive_sample_every_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            simulate_bimodal_ringdown(
                self._omega(), 1.0, np.zeros(40), dt=0.02, n_steps=100, sample_every=0
            )

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="too few"):
            simulate_bimodal_ringdown(
                self._omega(), 1.0, np.zeros(40), dt=0.02, n_steps=4, sample_every=5
            )


class TestRingdownSteps:
    def test_scales_with_decay_time(self) -> None:
        # weaker coupling decays faster (shorter run) than a near-threshold one
        fast = ringdown_steps(0.6, 0.5, dt=0.02, cap_time=60.0)
        slow = ringdown_steps(1.8, 0.5, dt=0.02, cap_time=60.0)
        assert 0 < fast < slow

    def test_zero_rate_at_threshold_is_capped(self) -> None:
        # at K = K_c the decay rate is zero (infinite decay time): the cap binds
        capped = ringdown_steps(2.0, 0.5, dt=0.02, cap_time=60.0)
        assert capped == int(60.0 / 0.02)

    def test_min_time_floors_a_fast_decay(self) -> None:
        # a fast-decaying point (six decay times < min_time) is floored to min_time
        rate = abs(0.6 / 4 - 0.5)  # |Re(λ)| at K = 0.6, Δ = 0.5
        assert 6.0 / rate < 36.0  # six decay times is below the floor
        floored = ringdown_steps(0.6, 0.5, dt=0.02, cap_time=60.0, min_time=36.0)
        assert floored == int(36.0 / 0.02)


class TestDecayingWindowEnd:
    def test_returns_length_when_never_reaches_floor(self) -> None:
        amplitude = np.array([0.5, 0.4, 0.3, 0.25])
        assert decaying_window_end(amplitude, 0.1) == 4

    def test_returns_first_index_below_floor(self) -> None:
        amplitude = np.array([0.5, 0.3, 0.08, 0.05])
        assert decaying_window_end(amplitude, 0.1) == 2


class TestFamilyReads:
    def test_envelope_recovers_a_known_decay_rate(self) -> None:
        # a clean complex decay exp((rate + i*Omega) t): |Z| = exp(rate t)
        rate_true, omega, dt_s = -0.2, 1.4, 0.1
        t = np.arange(120) * dt_s
        field = (0.6 * np.exp((rate_true + 1j * omega) * t)).astype(np.complex128)
        estimate = envelope_family_rate(field, rate=1.0 / dt_s, oscillators=4000)
        assert estimate == pytest.approx(rate_true, abs=0.05)

    def test_autocorrelation_returns_finite_rate(self) -> None:
        rng = np.random.default_rng(3)
        # an oscillating tail, as the confounded family sees it
        t = np.arange(600) * 0.1
        field = (np.exp(1j * 1.4 * t) + 0.3 * rng.standard_normal(600)).astype(
            np.complex128
        )
        rate = autocorrelation_family_rate(field, dt=0.1, window=64, step=8)
        assert np.isfinite(rate)

    def test_measured_frequency_recovers_a_pure_tone(self) -> None:
        dt_s, omega = 0.1, 1.4
        t = np.arange(1024) * dt_s
        field = np.exp(1j * omega * t).astype(np.complex128)
        got = measured_frequency(field, dt_sample=dt_s, max_angular=2.25)
        assert got == pytest.approx(omega, abs=0.1)

    def test_measured_frequency_rejects_out_of_band_peak(self) -> None:
        # a strong fast tone plus a weak in-band one: the band limit keeps the slow peak
        dt_s = 0.1
        t = np.arange(1024) * dt_s
        fast = np.exp(1j * 8.0 * t)
        slow = 0.3 * np.exp(1j * 1.4 * t)
        field = (fast + slow).astype(np.complex128)
        got = measured_frequency(field, dt_sample=dt_s, max_angular=2.25)
        assert got == pytest.approx(1.4, abs=0.15)

    def test_measured_frequency_falls_back_when_band_empty(self) -> None:
        # max_angular below the lowest positive bin leaves the band empty: full spectrum
        dt_s = 0.1
        t = np.arange(256) * dt_s
        field = np.exp(1j * 1.4 * t).astype(np.complex128)
        got = measured_frequency(field, dt_sample=dt_s, max_angular=1e-6)
        assert np.isfinite(got)


class TestRecord:
    def _kwargs(self, autocorr_factor: float) -> dict[str, object]:
        lam = [-0.35, -0.25, -0.15, -0.05]
        return {
            "coupling": [0.6, 1.0, 1.4, 1.8],
            "critical_coupling_value": 2.0,
            "true_rate": lam,
            "detector_value": {
                "envelope_growth": list(lam),
                "autocorrelation": [autocorr_factor * v for v in lam],
            },
            "analytic_frequency": [1.49, 1.46, 1.42, 1.39],
            "measured_frequency_value": [1.48, 1.45, 1.43, 1.40],
        }

    def test_assembles_families_and_frequency(self) -> None:
        record = kuramoto_bimodal_record(**self._kwargs(6.0))  # type: ignore[arg-type]
        assert record["n"] == 4
        assert record["critical_coupling"] == 2.0
        envelope = record["families"]["envelope_growth"]["correlation"]
        assert envelope["slope"] == pytest.approx(1.0)
        assert record["frequency"]["spearman"] == pytest.approx(1.0)

    def test_missing_family_raises(self) -> None:
        kwargs = self._kwargs(6.0)
        kwargs["detector_value"] = {"envelope_growth": [-0.35, -0.25, -0.15, -0.05]}
        with pytest.raises(ValueError, match="missing family"):
            kuramoto_bimodal_record(**kwargs)  # type: ignore[arg-type]


class TestVerdict:
    def _record(self, autocorr_factor: float) -> dict[str, object]:
        lam = [-0.35, -0.25, -0.15, -0.05]
        return kuramoto_bimodal_record(
            coupling=[0.6, 1.0, 1.4, 1.8],
            critical_coupling_value=2.0,
            true_rate=lam,
            detector_value={
                "envelope_growth": list(lam),
                "autocorrelation": [autocorr_factor * v for v in lam],
            },
            analytic_frequency=[1.49, 1.46, 1.42, 1.39],
            measured_frequency_value=[1.48, 1.45, 1.43, 1.40],
        )

    def test_reports_magnitude_recovery_when_only_envelope_has_unit_slope(self) -> None:
        verdict = kuramoto_bimodal_verdict(self._record(6.0))
        assert "recovers Re(λ) in magnitude" in verdict
        assert "first-principles" in verdict

    def test_inconclusive_when_autocorrelation_also_has_unit_slope(self) -> None:
        verdict = kuramoto_bimodal_verdict(self._record(1.0))
        assert "inconclusive" in verdict


class TestPayload:
    def _record(self) -> dict[str, object]:
        lam = [-0.35, -0.25, -0.15, -0.05]
        return kuramoto_bimodal_record(
            coupling=[0.6, 1.0, 1.4, 1.8],
            critical_coupling_value=2.0,
            true_rate=lam,
            detector_value={
                "envelope_growth": list(lam),
                "autocorrelation": [6.0 * v for v in lam],
            },
            analytic_frequency=[1.49, 1.46, 1.42, 1.39],
            measured_frequency_value=[1.48, 1.45, 1.43, 1.40],
        )

    def test_payload_seals_with_recomputable_hash(self) -> None:
        payload = kuramoto_bimodal_payload(
            record=self._record(),
            oscillators=4000,
            omega0=1.5,
            delta=0.5,
            sampling_dt=0.1,
            window=128,
            step=16,
        )
        assert payload["benchmark"] == BENCHMARK
        stored = dict(payload)
        content_hash = stored.pop("content_hash")
        assert canonical_record_hash(stored) == content_hash

    def test_payload_records_provenance(self) -> None:
        payload = kuramoto_bimodal_payload(
            record=self._record(),
            oscillators=4000,
            omega0=1.5,
            delta=0.5,
            sampling_dt=0.1,
            window=128,
            step=16,
        )
        assert payload["omega0"] == 1.5
        assert payload["delta"] == 0.5
        assert set(FAMILIES) == {"envelope_growth", "autocorrelation"}


class TestSweep:
    def test_tiny_sweep_runs_end_to_end(self) -> None:
        k_c = critical_coupling(0.5)
        record = sweep_bimodal_coupling(
            oscillators=80,
            omega0=1.5,
            delta=0.5,
            couplings=[0.3 * k_c, 0.6 * k_c, 0.9 * k_c],
            dt=0.02,
            sample_every=5,
            cap_time=25.0,
            min_time=20.0,
            window=32,
            step=4,
            seeds=2,
            seed=5,
        )
        assert record["n"] == 3
        assert record["critical_coupling"] == pytest.approx(k_c)
        assert record["true_rate"][0] < record["true_rate"][-1] < 0.0
        for label in FAMILIES:
            assert len(record["families"][label]["detector_value"]) == 3
        assert len(record["frequency"]["measured"]) == 3

    def test_non_positive_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="seeds"):
            sweep_bimodal_coupling(
                oscillators=80,
                omega0=1.5,
                delta=0.5,
                couplings=[0.6, 1.0, 1.4],
                dt=0.02,
                sample_every=5,
                cap_time=20.0,
                min_time=15.0,
                window=32,
                step=4,
                seeds=0,
                seed=1,
            )
