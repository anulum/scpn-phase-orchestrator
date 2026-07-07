# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kuramoto synchronisation external-validation unit tests

"""Unit tests for the Kuramoto synchronisation eigenvalue external validation.

These exercise the pure pieces — the analytic eigenvalue and critical coupling, the
correlation-with-slope, the noisy Kuramoto integrator, the observable split, the shipped
detector read, and the record/verdict/payload assembly — on small deterministic inputs.
The sealed-artefact numbers are pinned separately by the evidence integrity test.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.kuramoto_synchronization_external_validation import (
    BENCHMARK,
    OBSERVABLES,
    correlation,
    critical_coupling,
    detector_rate,
    eigenvalue,
    kuramoto_record,
    kuramoto_sync_external_validation_payload,
    kuramoto_sync_external_validation_verdict,
    observable_series,
    simulate_kuramoto,
    sweep_coupling,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


class TestAnalyticGroundTruth:
    def test_critical_coupling_generalises_lorentzian(self) -> None:
        # D = 0 recovers the classic K_c = 2γ; noise raises it to 2(γ + D)
        assert critical_coupling(0.5, 0.0) == pytest.approx(1.0)
        assert critical_coupling(0.5, 0.5) == pytest.approx(2.0)

    def test_eigenvalue_crosses_zero_at_critical_coupling(self) -> None:
        gamma, diffusion = 0.5, 0.5
        k_c = critical_coupling(gamma, diffusion)
        assert eigenvalue(k_c, gamma, diffusion) == pytest.approx(0.0)
        # below K_c the incoherent state is stable (λ < 0), above it unstable
        assert eigenvalue(0.5 * k_c, gamma, diffusion) < 0.0
        assert eigenvalue(1.5 * k_c, gamma, diffusion) > 0.0

    def test_eigenvalue_is_half_the_distance_to_threshold(self) -> None:
        # λ = (K − K_c)/2 exactly
        gamma, diffusion = 0.4, 0.3
        k_c = critical_coupling(gamma, diffusion)
        for coupling in (0.2, 0.6, 1.0):
            assert eigenvalue(coupling, gamma, diffusion) == pytest.approx(
                0.5 * (coupling - k_c)
            )


class TestCorrelation:
    def test_perfect_line_has_unit_slope(self) -> None:
        lam = [-0.7, -0.5, -0.3, -0.1]
        result = correlation(lam, lam)
        assert result["pearson"] == pytest.approx(1.0)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["slope"] == pytest.approx(1.0)
        assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
        assert result["mean_abs_gap"] == pytest.approx(0.0, abs=1e-9)
        assert result["n"] == 4

    def test_doubled_channel_has_slope_two(self) -> None:
        lam = [-0.7, -0.5, -0.3, -0.1]
        doubled = [2.0 * v for v in lam]
        result = correlation(lam, doubled)
        # a folded observable ranks perfectly but sizes at twice the slope
        assert result["spearman"] == pytest.approx(1.0)
        assert result["slope"] == pytest.approx(2.0)
        assert result["mean_abs_gap"] > 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            correlation([-0.5, -0.3], [-0.5, -0.3, -0.1])

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            correlation([-0.5, -0.3], [-0.5, -0.3])


class TestSimulateKuramoto:
    def _omega(self, n: int) -> np.ndarray:
        return np.linspace(-0.5, 0.5, n)

    def test_returns_sampled_mean_field(self) -> None:
        omega = self._omega(16)
        theta0 = np.linspace(-np.pi, np.pi, 16)
        field = simulate_kuramoto(
            omega,
            1.0,
            theta0,
            dt=0.05,
            n_samples=20,
            sample_every=3,
            diffusion=0.2,
            seed=1,
        )
        assert field.shape == (20,)
        assert field.dtype == np.complex128
        # the mean field is bounded by one in modulus
        assert np.all(np.abs(field) <= 1.0 + 1e-9)

    def test_reproducible_under_seed(self) -> None:
        omega = self._omega(16)
        theta0 = np.linspace(-np.pi, np.pi, 16)
        kwargs = {
            "dt": 0.05,
            "n_samples": 12,
            "sample_every": 2,
            "diffusion": 0.2,
            "seed": 7,
        }
        a = simulate_kuramoto(omega, 1.0, theta0, **kwargs)
        b = simulate_kuramoto(omega, 1.0, theta0, **kwargs)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        omega = self._omega(16)
        theta0 = np.linspace(-np.pi, np.pi, 16)
        a = simulate_kuramoto(
            omega,
            1.0,
            theta0,
            dt=0.05,
            n_samples=12,
            sample_every=2,
            diffusion=0.2,
            seed=7,
        )
        b = simulate_kuramoto(
            omega,
            1.0,
            theta0,
            dt=0.05,
            n_samples=12,
            sample_every=2,
            diffusion=0.2,
            seed=8,
        )
        assert not np.array_equal(a, b)

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            simulate_kuramoto(
                self._omega(8),
                1.0,
                np.zeros(8),
                dt=0.05,
                n_samples=1,
                sample_every=2,
                diffusion=0.1,
                seed=1,
            )

    def test_non_positive_sample_every_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            simulate_kuramoto(
                self._omega(8),
                1.0,
                np.zeros(8),
                dt=0.05,
                n_samples=4,
                sample_every=0,
                diffusion=0.1,
                seed=1,
            )


class TestObservableSeries:
    def test_splits_into_two_zero_mean_observables(self) -> None:
        field = np.array([0.2 + 0.1j, -0.1 + 0.3j, 0.4 - 0.2j, 0.0 + 0.0j])
        series = observable_series(field)
        assert set(series) == set(OBSERVABLES)
        for values in series.values():
            assert values.shape == (4,)
            assert float(np.mean(values)) == pytest.approx(0.0, abs=1e-12)

    def test_amplitude_is_folded_nonnegative_before_centering(self) -> None:
        field = np.array([0.3 + 0.4j, -0.6 + 0.0j, 0.0 - 0.5j])
        amplitude = np.abs(field)
        assert np.allclose(amplitude, [0.5, 0.6, 0.5])


class TestDetectorRate:
    def test_slowly_relaxing_series_gives_small_negative_rate(self) -> None:
        # an AR(1) with a known coefficient close to one: ln(AR1)/dt ≈ ln(0.9)/dt
        rng = np.random.default_rng(0)
        n = 3000
        x = np.zeros(n)
        phi = 0.9
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.standard_normal()
        rate = detector_rate(x - x.mean(), dt=0.25, window=256, step=32)
        assert rate < 0.0
        # the implied rate is in the neighbourhood of ln(0.9)/0.25 ≈ −0.42
        assert rate == pytest.approx(np.log(phi) / 0.25, abs=0.25)

    def test_returns_finite_float(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        rate = detector_rate(x, dt=0.25, window=256, step=32)
        assert np.isfinite(rate)


class TestKuramotoRecord:
    def _detector(self) -> dict[str, list[float]]:
        lam = [-0.7, -0.5, -0.3, -0.1]
        return {
            "mean_field_real": list(lam),
            "order_parameter_amplitude": [2.0 * v for v in lam],
        }

    def test_assembles_per_observable_correlation(self) -> None:
        lam = [-0.7, -0.5, -0.3, -0.1]
        record = kuramoto_record(
            coupling=[0.6, 1.0, 1.4, 1.8],
            critical_coupling_value=2.0,
            true_rate=lam,
            detector_value=self._detector(),
        )
        assert record["n"] == 4
        assert record["critical_coupling"] == 2.0
        signed = record["observables"]["mean_field_real"]["correlation"]
        assert signed["slope"] == pytest.approx(1.0)

    def test_missing_observable_raises(self) -> None:
        with pytest.raises(ValueError, match="missing observable"):
            kuramoto_record(
                coupling=[0.6, 1.0, 1.4, 1.8],
                critical_coupling_value=2.0,
                true_rate=[-0.7, -0.5, -0.3, -0.1],
                detector_value={"mean_field_real": [-0.7, -0.5, -0.3, -0.1]},
            )


class TestVerdict:
    def _record(self, amplitude_factor: float) -> dict[str, object]:
        lam = [-0.7, -0.5, -0.3, -0.1]
        return kuramoto_record(
            coupling=[0.6, 1.0, 1.4, 1.8],
            critical_coupling_value=2.0,
            true_rate=lam,
            detector_value={
                "mean_field_real": list(lam),
                "order_parameter_amplitude": [amplitude_factor * v for v in lam],
            },
        )

    def test_reports_magnitude_recovery_when_only_signed_has_unit_slope(self) -> None:
        verdict = kuramoto_sync_external_validation_verdict(self._record(2.0))
        assert "recovers λ in magnitude" in verdict
        assert "first-principles" in verdict

    def test_inconclusive_when_amplitude_also_has_unit_slope(self) -> None:
        # if the amplitude channel also fits with slope ≈ 1, the split is not clean
        verdict = kuramoto_sync_external_validation_verdict(self._record(1.0))
        assert "inconclusive" in verdict


class TestPayload:
    def _record(self) -> dict[str, object]:
        lam = [-0.7, -0.5, -0.3, -0.1]
        return kuramoto_record(
            coupling=[0.6, 1.0, 1.4, 1.8],
            critical_coupling_value=2.0,
            true_rate=lam,
            detector_value={
                "mean_field_real": list(lam),
                "order_parameter_amplitude": [2.0 * v for v in lam],
            },
        )

    def test_payload_seals_with_recomputable_hash(self) -> None:
        payload = kuramoto_sync_external_validation_payload(
            record=self._record(),
            oscillators=512,
            gamma=0.5,
            diffusion=0.5,
            sampling_dt=0.25,
            window=512,
            step=64,
        )
        assert payload["benchmark"] == BENCHMARK
        stored = dict(payload)
        content_hash = stored.pop("content_hash")
        assert canonical_record_hash(stored) == content_hash

    def test_payload_records_provenance(self) -> None:
        payload = kuramoto_sync_external_validation_payload(
            record=self._record(),
            oscillators=512,
            gamma=0.5,
            diffusion=0.5,
            sampling_dt=0.25,
            window=512,
            step=64,
        )
        assert payload["oscillators"] == 512
        assert payload["gamma"] == 0.5
        assert payload["diffusion"] == 0.5
        assert payload["window"] == 512


class TestSweepCoupling:
    def test_tiny_sweep_runs_end_to_end(self) -> None:
        gamma, diffusion = 0.5, 0.5
        k_c = critical_coupling(gamma, diffusion)
        record = sweep_coupling(
            oscillators=32,
            gamma=gamma,
            diffusion=diffusion,
            couplings=[0.3 * k_c, 0.6 * k_c, 0.9 * k_c],
            dt=0.05,
            n_samples=200,
            sample_every=2,
            window=64,
            step=8,
            burn_in_fraction=0.25,
            seed=3,
        )
        assert record["n"] == 3
        assert record["critical_coupling"] == pytest.approx(k_c)
        # the eigenvalue is more negative at weaker coupling
        assert record["true_rate"][0] < record["true_rate"][-1] < 0.0
        for label in OBSERVABLES:
            assert len(record["observables"][label]["detector_value"]) == 3

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_burn_in_out_of_range_raises(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="burn_in_fraction"):
            sweep_coupling(
                oscillators=16,
                gamma=0.5,
                diffusion=0.5,
                couplings=[0.6, 1.0, 1.4],
                dt=0.05,
                n_samples=100,
                sample_every=2,
                window=32,
                step=4,
                burn_in_fraction=fraction,
                seed=1,
            )
