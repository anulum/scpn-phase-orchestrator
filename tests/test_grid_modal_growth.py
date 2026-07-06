# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth detector primitive tests

"""Tests for the product-side grid modal-growth detector primitives.

The cross-bus and per-bus deviation envelopes, the exponential growth rate with and
without recency weighting, and the per-segment score under both aggregations are
exercised on synthetic multi-bus oscillations with a planted growth rate, alongside
every guard. The detector recovers a known ``σ`` and separates growing (unstable)
segments from damped ones — the behaviour the offline benchmark then certifies.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    cross_bus_deviation,
    envelope_growth_rate,
    modal_growth_score,
    per_bus_deviation,
)

_RATE = 238.0
_SAMPLES = 476  # a 2 s segment at the PSML rate


def _oscillation(
    rng: np.random.Generator, *, sigma: float, buses: int = 4
) -> np.ndarray:
    """Return a multi-bus oscillation whose amplitude envelope grows at rate sigma.

    The shared envelope ``exp(sigma·t)`` modulates a 1 Hz wave on every bus, so the
    cross-bus deviation grows (sigma > 0) or decays (sigma < 0) at the planted rate.
    """
    time = np.arange(_SAMPLES) / _RATE
    envelope = np.exp(sigma * time)
    wave = np.sin(2.0 * np.pi * 1.0 * time)
    return np.stack(
        [
            1.0
            + envelope * wave * rng.uniform(0.8, 1.2)
            + 1e-3 * rng.standard_normal(_SAMPLES)
            for _ in range(buses)
        ]
    )


# --------------------------------------------------------------------------- #
# cross_bus_deviation                                                         #
# --------------------------------------------------------------------------- #


def test_cross_bus_deviation_is_a_per_sample_envelope() -> None:
    rng = np.random.default_rng(0)
    deviation = cross_bus_deviation(_oscillation(rng, sigma=0.5))
    assert deviation.shape == (_SAMPLES,)
    assert np.all(deviation >= 0.0)
    assert deviation[-1] > deviation[0]  # the envelope grows


def test_cross_bus_deviation_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        cross_bus_deviation(np.zeros(8))


def test_cross_bus_deviation_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError, match="at least one bus and one sample"):
        cross_bus_deviation(np.zeros((0, 8)))


# --------------------------------------------------------------------------- #
# per_bus_deviation                                                           #
# --------------------------------------------------------------------------- #


def test_per_bus_deviation_keeps_one_envelope_per_bus() -> None:
    rng = np.random.default_rng(5)
    envelopes = per_bus_deviation(_oscillation(rng, sigma=0.6, buses=4))
    assert envelopes.shape == (4, _SAMPLES)
    assert np.all(envelopes >= 0.0)
    # the growing envelope makes the later peaks larger than the earliest ones
    assert envelopes[:, -100:].max() > envelopes[:, :100].max()


def test_per_bus_deviation_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        per_bus_deviation(np.zeros(8))


def test_per_bus_deviation_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError, match="at least one bus and one sample"):
        per_bus_deviation(np.zeros((8, 0)))


# --------------------------------------------------------------------------- #
# envelope_growth_rate                                                        #
# --------------------------------------------------------------------------- #


def test_envelope_growth_rate_recovers_a_planted_sigma() -> None:
    rate = 100.0
    times = np.arange(500) / rate
    envelope = np.exp(0.7 * times)  # a known growth rate
    assert envelope_growth_rate(envelope, rate=rate) == pytest.approx(0.7, abs=1e-6)


def test_envelope_growth_rate_recovers_a_planted_sigma_with_recency() -> None:
    rate = 100.0
    times = np.arange(500) / rate
    envelope = np.exp(0.7 * times)  # perfectly log-linear: any weighting recovers 0.7
    assert envelope_growth_rate(envelope, rate=rate, recency_top=3.0) == pytest.approx(
        0.7, abs=1e-6
    )


def test_envelope_growth_rate_is_negative_for_a_damped_envelope() -> None:
    rate = 100.0
    times = np.arange(500) / rate
    assert envelope_growth_rate(np.exp(-0.4 * times), rate=rate) < 0.0


def test_envelope_growth_rate_is_zero_for_a_non_finite_envelope() -> None:
    assert envelope_growth_rate(np.array([float("nan"), 1.0, 2.0]), rate=_RATE) == 0.0


def test_envelope_growth_rate_recency_is_zero_for_a_non_finite_envelope() -> None:
    # the weighted path also floors an undefined fit to zero
    assert (
        envelope_growth_rate(
            np.array([float("nan"), 1.0, 2.0]), rate=_RATE, recency_top=3.0
        )
        == 0.0
    )


def test_envelope_growth_rate_rejects_a_non_vector() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        envelope_growth_rate(np.zeros((2, 4)), rate=_RATE)


def test_envelope_growth_rate_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        envelope_growth_rate(np.zeros(1), rate=_RATE)


def test_envelope_growth_rate_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        envelope_growth_rate(np.ones(8), rate=0.0)


def test_envelope_growth_rate_rejects_a_recency_below_one() -> None:
    with pytest.raises(ValueError, match="recency_top"):
        envelope_growth_rate(np.ones(8), rate=_RATE, recency_top=0.5)


def test_envelope_growth_rate_rejects_a_non_finite_recency() -> None:
    with pytest.raises(ValueError, match="recency_top"):
        envelope_growth_rate(np.ones(8), rate=_RATE, recency_top=float("nan"))


# --------------------------------------------------------------------------- #
# modal_growth_score                                                          #
# --------------------------------------------------------------------------- #


def test_modal_growth_score_recovers_the_segment_growth() -> None:
    rng = np.random.default_rng(1)
    growing = modal_growth_score(_oscillation(rng, sigma=0.5), rate=_RATE)
    damped = modal_growth_score(_oscillation(rng, sigma=-0.5), rate=_RATE)
    assert growing > 0.0
    assert damped < 0.0


def test_modal_growth_score_mean_aggregation_recovers_growth() -> None:
    rng = np.random.default_rng(6)
    growing = modal_growth_score(
        _oscillation(rng, sigma=0.5), rate=_RATE, aggregation="mean"
    )
    damped = modal_growth_score(
        _oscillation(rng, sigma=-0.5), rate=_RATE, aggregation="mean"
    )
    assert growing > 0.0
    assert damped < 0.0


def test_modal_growth_score_rejects_an_unknown_aggregation() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="aggregation must be 'mean' or 'focal'"):
        modal_growth_score(
            _oscillation(rng, sigma=0.5), rate=_RATE, aggregation="median"
        )
