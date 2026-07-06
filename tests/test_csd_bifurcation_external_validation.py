# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CSD bifurcation external-validation pure-core tests

"""Branch-complete tests for the critical-slowing-down external-validation core.

Every pure function is exercised on synthetic data — correlations, the Euler–Maruyama
integrator, the shipped-detector indicator read, record assembly, the verdict's honest
branches, the sealing payload, and the sweep orchestration — with no dependence on the
committed artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.csd_bifurcation_external_validation import (
    BENCHMARK,
    INDICATORS,
    bifurcation_record,
    correlation,
    csd_external_validation_payload,
    csd_external_validation_verdict,
    detector_indicators,
    simulate_normal_form,
    sweep_bifurcation,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

# --- correlation


def test_correlation_perfect_rank() -> None:
    result = correlation([-0.8, -0.5, -0.2], [-0.82, -0.49, -0.19])
    assert result["n"] == 3
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] > 0.99


def test_correlation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_correlation_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        correlation([1.0, 2.0], [1.0, 2.0])


# ----------------------------------------------------------------- simulate_normal_form


def test_simulate_normal_form_length_and_start() -> None:
    series = simulate_normal_form(
        lambda x, mu: mu * x, 0.0, -0.3, dt=0.1, n=64, sigma=0.01, seed=7
    )
    assert series.shape == (64,)
    assert series[0] == 0.0


def test_simulate_normal_form_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        simulate_normal_form(
            lambda x, mu: mu * x, 0.0, -0.3, dt=0.1, n=1, sigma=0.01, seed=1
        )


def test_simulate_normal_form_recovers_ou_autocorrelation() -> None:
    # a linear (Ornstein–Uhlenbeck) drift has lag-one autocorrelation exp(λ dt); the
    # integrator must reproduce it, which validates the scheme mathematically
    lam, dt = -0.4, 0.2
    series = simulate_normal_form(
        lambda x, mu: mu * x, 0.0, lam, dt=dt, n=40_000, sigma=0.05, seed=3
    )
    tail = series[2000:]
    centred = tail - tail.mean()
    ar1 = float(np.mean(centred[:-1] * centred[1:]) / np.mean(centred * centred))
    assert ar1 == pytest.approx(np.exp(lam * dt), abs=0.02)


# --- detector_indicators


def test_detector_indicators_keys_and_rate_sign() -> None:
    rng = np.random.default_rng(11)
    # a mildly correlated series: positive but sub-unity AR1 -> a negative implied rate
    noise = rng.standard_normal(512)
    series = np.convolve(noise, np.ones(4) / 4.0, mode="same")
    indicators = detector_indicators(series, dt=0.2, window=64, step=16)
    assert set(indicators) == set(INDICATORS)
    assert indicators["autocorrelation"] < 0.0
    assert indicators["variance"] > 0.0


def test_detector_indicators_clips_autocorrelation_floor() -> None:
    # a sign-alternating series has lag-one autocorrelation near -1, below the floor,
    # so the implied rate is finite (the clip prevents ln of a non-positive value)
    series = np.array([1.0, -1.0] * 64, dtype=np.float64)
    indicators = detector_indicators(series, dt=0.2, window=8, step=4)
    assert np.isfinite(indicators["autocorrelation"])


def test_detector_indicators_clips_autocorrelation_ceiling() -> None:
    # a smooth ramp has lag-one autocorrelation near 1; the ceiling clip keeps the
    # implied rate finite rather than ln(1)/dt collapsing to exactly zero from above
    series = np.linspace(0.0, 1.0, 512, dtype=np.float64)
    indicators = detector_indicators(series, dt=0.2, window=64, step=16)
    assert np.isfinite(indicators["autocorrelation"])
    assert indicators["autocorrelation"] < 0.0


# ------------------------------------------------------------------- bifurcation_record


def _good_record(name: str = "fold") -> dict[str, object]:
    return bifurcation_record(
        name=name,
        normal_form="dx = (μ - x²) dt",
        control=[0.5, 0.3, 0.1],
        true_rate=[-1.4, -1.1, -0.6],
        detector_value={
            "autocorrelation": [-1.5, -1.0, -0.55],
            "variance": [1.0, 1.5, 2.4],
        },
    )


def test_bifurcation_record_shape() -> None:
    record = _good_record()
    assert record["name"] == "fold"
    assert record["n"] == 3
    indicators = record["indicators"]
    assert isinstance(indicators, dict)
    assert indicators["autocorrelation"]["correlation"]["spearman"] == pytest.approx(
        1.0
    )


def test_bifurcation_record_missing_indicator_raises() -> None:
    with pytest.raises(ValueError, match="missing indicator 'variance'"):
        bifurcation_record(
            name="fold",
            normal_form="dx = (μ - x²) dt",
            control=[0.5, 0.3, 0.1],
            true_rate=[-1.4, -1.1, -0.6],
            detector_value={"autocorrelation": [-1.5, -1.0, -0.55]},
        )


# ----------------------------------------------------------------------------- verdict


def test_verdict_recovers_branch() -> None:
    verdict = csd_external_validation_verdict([_good_record("fold")])
    assert "recovers the true recovery rate" in verdict
    assert "rises in step" in verdict


def test_verdict_does_not_recover_branch() -> None:
    # an autocorrelation channel that is anti-correlated with λ trips the failure clause
    bad = bifurcation_record(
        name="fold",
        normal_form="dx = (μ - x²) dt",
        control=[0.5, 0.3, 0.1],
        true_rate=[-1.4, -1.1, -0.6],
        detector_value={
            "autocorrelation": [-0.55, -1.0, -1.5],  # decreasing while λ increases
            "variance": [2.4, 1.5, 1.0],  # also anti-correlated
        },
    )
    verdict = csd_external_validation_verdict([bad])
    assert "does not recover" in verdict
    assert "inconsistent" in verdict


# ----------------------------------------------------------------------------- payload


def test_payload_seals_and_recomputes() -> None:
    payload = csd_external_validation_payload(
        bifurcations=[_good_record("fold"), _good_record("pitchfork")],
        sigma=0.02,
        sampling_dt=0.2,
        window=256,
        step=32,
    )
    assert payload["benchmark"] == BENCHMARK
    stored = dict(payload)
    content_hash = stored.pop("content_hash")
    assert canonical_record_hash(stored) == content_hash


# --- sweep_bifurcation


def test_sweep_bifurcation_end_to_end_positive_recovery() -> None:
    record = sweep_bifurcation(
        name="fold",
        normal_form="dx = (μ - x²) dt",
        drift=lambda x, mu: mu - x * x,
        equilibrium=lambda mu: float(np.sqrt(mu)),
        true_lambda=lambda mu: -2.0 * float(np.sqrt(mu)),
        controls=[0.5, 0.3, 0.15, 0.05],
        dt=0.2,
        n=1600,
        sigma=0.02,
        window=128,
        step=16,
        burn_in_fraction=0.2,
        seed=1000,
    )
    rho = record["indicators"]["autocorrelation"]["correlation"]["spearman"]
    assert rho > 0.5  # the shipped detector recovers the true-λ ordering end to end


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_sweep_bifurcation_burn_in_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="burn_in_fraction"):
        sweep_bifurcation(
            name="fold",
            normal_form="dx = (μ - x²) dt",
            drift=lambda x, mu: mu - x * x,
            equilibrium=lambda mu: float(np.sqrt(mu)),
            true_lambda=lambda mu: -2.0 * float(np.sqrt(mu)),
            controls=[0.5, 0.3, 0.1],
            dt=0.2,
            n=800,
            sigma=0.02,
            window=128,
            step=16,
            burn_in_fraction=bad,
            seed=1000,
        )
