# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hopf-bridge external-validation pure-core tests

"""Branch-complete tests for the Hopf-bridge external-validation core.

Every pure function is exercised on synthetic data — the correlation with its magnitude
gap, the Hopf integrator, both family readers, record assembly, the verdict's
frequency-invariant and frequency-sensitive branches, the robustness helper, the sweep,
and the sealing payload — with no dependence on the committed artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.hopf_bridge_external_validation import (
    BENCHMARK,
    FAMILIES,
    _envelope_pearson,
    autocorrelation_family_rate,
    correlation,
    envelope_family_sigma,
    hopf_bridge_payload,
    hopf_bridge_verdict,
    hopf_record,
    simulate_hopf,
    sweep_hopf,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

# --- correlation


def test_correlation_reports_magnitude_gap() -> None:
    result = correlation([-0.5, -0.3, -0.1], [-0.5, -0.3, -0.1])
    assert result["n"] == 3
    assert result["spearman"] == pytest.approx(1.0)
    assert result["mean_abs_error"] == pytest.approx(0.0)


def test_correlation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_correlation_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        correlation([1.0, 2.0], [1.0, 2.0])


# --- simulate_hopf


def test_simulate_hopf_shape_and_start() -> None:
    x = simulate_hopf(-0.3, 2.0, 1.0, dt=0.2, n=64, sigma=0.0, seed=1)
    assert x.shape == (64,)
    assert x[0] == pytest.approx(1.0)  # r0 * cos(0)


def test_simulate_hopf_ringdown_decays() -> None:
    x = simulate_hopf(-0.4, 2.0, 1.0, dt=0.2, n=120, sigma=0.0, seed=1)
    # a damped Hopf ringdown shrinks: the late envelope is below the early one
    assert np.max(np.abs(x[-20:])) < np.max(np.abs(x[:20]))


def test_simulate_hopf_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        simulate_hopf(-0.3, 2.0, 1.0, dt=0.2, n=1, sigma=0.0, seed=1)


# --- family readers


def test_envelope_family_sigma_negative_on_ringdown() -> None:
    ring = simulate_hopf(-0.4, 2.0, 1.0, dt=0.2, n=90, sigma=0.001, seed=2)
    assert envelope_family_sigma(ring, rate=5.0) < 0.0


def test_autocorrelation_family_rate_finite() -> None:
    stat = simulate_hopf(-0.2, 2.0, 0.5, dt=0.2, n=800, sigma=0.05, seed=3)
    rate = autocorrelation_family_rate(stat - stat.mean(), dt=0.2, window=64, step=16)
    assert np.isfinite(rate)


def test_autocorrelation_family_rate_clips_floor() -> None:
    # a sign-alternating series has lag-one autocorrelation near -1, below the floor
    series = np.array([1.0, -1.0] * 64, dtype=np.float64)
    assert np.isfinite(autocorrelation_family_rate(series, dt=0.2, window=8, step=4))


def test_autocorrelation_family_rate_clips_ceiling() -> None:
    # a smooth ramp has lag-one autocorrelation near 1, at the ceiling
    series = np.linspace(0.0, 1.0, 512, dtype=np.float64)
    assert autocorrelation_family_rate(series, dt=0.2, window=64, step=16) < 0.0


# --- hopf_record


def _record() -> dict[str, object]:
    return hopf_record(
        omega_hz=0.4,
        ring_sigma=0.002,
        alphas=[-0.5, -0.3, -0.1],
        envelope_sigma=[-0.48, -0.29, -0.11],  # near 1:1 with alpha
        autocorrelation_rate=[-1.0, -0.95, -0.9],  # confounded magnitude
    )


def test_hopf_record_shape_and_families() -> None:
    record = _record()
    assert record["n"] == 3
    families = record["families"]
    assert isinstance(families, dict)
    assert set(families) == set(FAMILIES)
    env = families["envelope_growth"]["correlation"]
    ac = families["autocorrelation"]["correlation"]
    # the envelope estimate is close to alpha; the autocorrelation is far in magnitude
    assert env["mean_abs_error"] < ac["mean_abs_error"]


# --- verdict


def test_verdict_frequency_invariant_branch() -> None:
    verdict = hopf_bridge_verdict(_record(), {"pearson_spread": 0.0})
    assert "frequency-invariant" in verdict
    assert "regime-dependent" in verdict


def test_verdict_frequency_sensitive_branch() -> None:
    verdict = hopf_bridge_verdict(_record(), {"pearson_spread": 0.2})
    assert "frequency-sensitive" in verdict


# --- robustness helper


def test_envelope_pearson_recovers_alpha() -> None:
    alphas = list(np.linspace(-0.5, -0.03, 8))
    pearson = _envelope_pearson(
        alphas, omega_hz=0.4, rate=5.0, ring_sigma=0.002, ring_n=90, seed=5000
    )
    assert pearson > 0.8  # a clean ringdown recovers alpha


# --- payload


def test_payload_seals_and_recomputes() -> None:
    payload = hopf_bridge_payload(
        record=_record(),
        snr_robustness=[{"ring_sigma": 0.002, "pearson": 0.97}],
        frequency_invariance={"pearson_spread": 0.0},
        window=128,
        step=16,
    )
    assert payload["benchmark"] == BENCHMARK
    stored = dict(payload)
    content_hash = stored.pop("content_hash")
    assert canonical_record_hash(stored) == content_hash


# --- sweep_hopf


def test_sweep_hopf_end_to_end() -> None:
    record = sweep_hopf(
        alphas=list(np.linspace(-0.5, -0.05, 6)),
        omega_hz=0.4,
        rate=5.0,
        ring_sigma=0.002,
        ring_n=90,
        stationary_sigma=0.05,
        stationary_n=1200,
        stationary_burn=200,
        window=128,
        step=16,
        seed=5000,
    )
    env = record["families"]["envelope_growth"]["correlation"]
    assert env["spearman"] > 0.5  # the envelope family recovers the alpha ordering


@pytest.mark.parametrize("bad", [-1, 1200, 5000])
def test_sweep_hopf_burn_out_of_range_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="stationary_burn"):
        sweep_hopf(
            alphas=[-0.5, -0.3, -0.1],
            omega_hz=0.4,
            rate=5.0,
            ring_sigma=0.002,
            ring_n=90,
            stationary_sigma=0.05,
            stationary_n=1200,
            stationary_burn=bad,
            window=128,
            step=16,
            seed=5000,
        )
