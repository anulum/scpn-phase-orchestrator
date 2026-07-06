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
    fit_gated_growth_rate,
    growth_rate_and_fit,
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


def _monotone_growth(sigma: float, *, buses: int = 4) -> np.ndarray:
    """Return buses whose per-bus deviation is a smooth monotone ``exp(sigma·t)``.

    Each bus offsets the (constant) cross-bus mean by ``offset·exp(sigma·t)`` with
    offsets summing to zero, so every per-bus deviation is a clean exponential — a fit
    the ``R²`` gate keeps (unlike a rectified sine, whose zero crossings fit poorly).
    """
    time = np.arange(_SAMPLES) / _RATE
    offsets = np.linspace(-0.3, 0.3, buses)
    return 1.0 + offsets[:, None] * np.exp(sigma * time)[None, :]


def _step_transient(*, buses: int = 4) -> np.ndarray:
    """Return buses whose per-bus deviation jumps late — a fault's step, not a mode.

    The deviation trends upward (a positive slope) but fits an exponential poorly (a low
    ``R²``), so the fit-quality gate rejects it.
    """
    voltages = np.ones((buses, _SAMPLES))
    voltages[1, -40:] = 3.0
    voltages[2, -40:] = 2.5
    return voltages


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


def test_modal_growth_score_gate_keeps_smooth_focal_growth() -> None:
    # a genuine instability fits an exponential well, so the gate is a no-op on it
    segment = _monotone_growth(0.6)
    ungated = modal_growth_score(segment, rate=_RATE, aggregation="focal")
    gated = modal_growth_score(segment, rate=_RATE, aggregation="focal", r2_gate=0.5)
    assert gated == pytest.approx(ungated, abs=1e-9)
    assert gated == pytest.approx(0.6, abs=1e-6)


def test_modal_growth_score_gate_keeps_smooth_mean_growth() -> None:
    segment = _monotone_growth(0.6)
    ungated = modal_growth_score(segment, rate=_RATE, aggregation="mean")
    gated = modal_growth_score(segment, rate=_RATE, aggregation="mean", r2_gate=0.5)
    assert gated == pytest.approx(ungated, abs=1e-9)


def test_modal_growth_score_gate_clamps_a_step_segment() -> None:
    segment = _step_transient()
    ungated = modal_growth_score(segment, rate=_RATE, aggregation="focal")
    gated = modal_growth_score(segment, rate=_RATE, aggregation="focal", r2_gate=0.5)
    assert ungated > 0.0  # the step's upward trend reads as growth ungated
    assert gated == 0.0  # the fit gate rejects it


def test_modal_growth_score_rejects_a_gate_out_of_range() -> None:
    with pytest.raises(ValueError, match="r2_gate must be a finite number"):
        modal_growth_score(_monotone_growth(0.5), rate=_RATE, r2_gate=1.5)


# --------------------------------------------------------------------------- #
# growth_rate_and_fit                                                         #
# --------------------------------------------------------------------------- #


def test_growth_rate_and_fit_recovers_sigma_with_a_high_r2() -> None:
    times = np.arange(_SAMPLES) / _RATE
    sigma, r2 = growth_rate_and_fit(np.exp(0.8 * times), rate=_RATE, recency_top=3.0)
    assert sigma == pytest.approx(0.8, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_growth_rate_and_fit_sigma_matches_envelope_growth_rate() -> None:
    times = np.arange(_SAMPLES) / _RATE
    envelope = np.exp(0.6 * times) * (1.0 + 0.05 * np.sin(3.0 * times))
    for recency_top in (1.0, 3.0):
        sigma, _ = growth_rate_and_fit(envelope, rate=_RATE, recency_top=recency_top)
        # the gate must never perturb the certified rate, on either fit path
        assert sigma == envelope_growth_rate(
            envelope, rate=_RATE, recency_top=recency_top
        )


def test_growth_rate_and_fit_low_r2_for_a_step_transient() -> None:
    envelope = np.full(_SAMPLES, 0.05)
    envelope[-40:] = 1.5  # a fault's step-like jump, not a smooth exponential
    sigma, r2 = growth_rate_and_fit(envelope, rate=_RATE, recency_top=3.0)
    assert sigma > 0.0  # the step still trends upward
    assert r2 < 0.5  # but fits an exponential poorly


def test_growth_rate_and_fit_zero_r2_for_a_flat_envelope() -> None:
    # a constant log envelope has no variance to explain, so R^2 is defined as zero
    sigma, r2 = growth_rate_and_fit(np.full(_SAMPLES, 0.7), rate=_RATE, recency_top=3.0)
    assert sigma == pytest.approx(0.0, abs=1e-9)
    assert r2 == 0.0


def test_growth_rate_and_fit_propagates_input_guards() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        growth_rate_and_fit(np.zeros(1), rate=_RATE)


# --------------------------------------------------------------------------- #
# fit_gated_growth_rate                                                       #
# --------------------------------------------------------------------------- #


def test_fit_gated_growth_rate_off_equals_the_plain_rate() -> None:
    times = np.arange(_SAMPLES) / _RATE
    envelope = np.exp(0.7 * times)
    assert fit_gated_growth_rate(
        envelope, rate=_RATE, recency_top=3.0, r2_gate=0.0
    ) == envelope_growth_rate(envelope, rate=_RATE, recency_top=3.0)


def test_fit_gated_growth_rate_keeps_a_well_fit_growing_rate() -> None:
    times = np.arange(_SAMPLES) / _RATE
    gated = fit_gated_growth_rate(
        np.exp(0.7 * times), rate=_RATE, recency_top=3.0, r2_gate=0.5
    )
    assert gated == pytest.approx(0.7, abs=1e-6)


def test_fit_gated_growth_rate_keeps_a_well_fit_damped_rate() -> None:
    times = np.arange(_SAMPLES) / _RATE
    # a smoothly damped envelope fits well (high R^2), so its negative rate is kept
    gated = fit_gated_growth_rate(
        np.exp(-0.4 * times), rate=_RATE, recency_top=3.0, r2_gate=0.5
    )
    assert gated < 0.0


def test_fit_gated_growth_rate_clamps_a_poorly_fit_transient() -> None:
    envelope = np.full(_SAMPLES, 0.05)
    envelope[-40:] = 1.5  # positive slope but a poor exponential fit (R^2 < 0.5)
    gated = fit_gated_growth_rate(envelope, rate=_RATE, recency_top=3.0, r2_gate=0.5)
    assert gated == 0.0  # min(sigma, 0) once the fit gate rejects it


@pytest.mark.parametrize("gate", [1.5, -0.1, float("nan")])
def test_fit_gated_growth_rate_rejects_a_gate_out_of_range(gate: float) -> None:
    with pytest.raises(ValueError, match="r2_gate must be a finite number"):
        fit_gated_growth_rate(np.ones(8), rate=_RATE, r2_gate=gate)
