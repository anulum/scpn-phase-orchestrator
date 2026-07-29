# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — tests for adaptive multi-channel Kuramoto detector

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.adaptive_kuramoto import (
    _kurtosis_excess,
    compute_adaptive_kuramoto_scores,
    compute_channel_quality_weights,
    compute_phase_locking_weights,
    compute_weighted_kuramoto_r,
)


@pytest.fixture
def coherent_channels() -> np.ndarray:
    """Return a 4-channel signal with strong coherent delta oscillations."""
    fs = 100.0
    t = np.arange(int(60 * fs)) / fs
    base = np.sin(2 * np.pi * 2.0 * t)
    noise = 0.05 * np.random.default_rng(0).standard_normal((4, t.size))
    return np.asarray(base + noise, dtype=np.float64)


def test_quality_weights_reward_high_snr() -> None:
    """Channels with stronger target-band power receive higher weights."""
    fs = 100.0
    n_samples = int(60 * fs)
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(1)

    # Channel 0: strong 2 Hz oscillation.
    ch0 = np.sin(2 * np.pi * 2.0 * t) + 0.1 * rng.standard_normal(n_samples)
    # Channel 1: weak 2 Hz oscillation + broadband noise.
    ch1 = 0.2 * np.sin(2 * np.pi * 2.0 * t) + rng.standard_normal(n_samples)

    data = np.vstack([ch0, ch1]).astype(np.float64)
    weights = compute_channel_quality_weights(data, fs, epoch_seconds=30.0)

    assert weights.shape == (2, 2)
    assert np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=0), 1.0)
    # The strong channel should dominate in both epochs.
    assert weights[0, 0] > weights[1, 0]
    assert weights[0, 1] > weights[1, 1]


def test_adaptive_scores_detect_coherence_difference() -> None:
    """Higher phase coherence yields higher adaptive Kuramoto scores."""
    fs = 100.0
    n_samples = int(60 * fs)
    epoch_samples = int(30 * fs)
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(2)

    # First epoch: four perfectly coherent channels.
    base = np.sin(2 * np.pi * 2.0 * t)
    epoch1 = np.tile(base[:epoch_samples], (4, 1)) + 0.01 * rng.standard_normal(
        (4, epoch_samples)
    )
    # Second epoch: same channels with independent phase noise (low coherence).
    epoch2 = np.zeros((4, epoch_samples))
    for c in range(4):
        phase_shift = rng.uniform(0, 2 * np.pi)
        epoch2[c, :] = np.sin(2 * np.pi * 2.0 * t[:epoch_samples] + phase_shift)
    epoch2 += 0.01 * rng.standard_normal((4, epoch_samples))

    data = np.hstack([epoch1, epoch2]).astype(np.float64)
    scores, _ = compute_adaptive_kuramoto_scores(data, fs, epoch_seconds=30.0)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0


def test_weighted_kuramoto_requires_two_channels() -> None:
    """Single-channel input is rejected."""
    fs = 100.0
    data = np.random.default_rng(3).standard_normal((1, int(30 * fs)))
    with pytest.raises(ValueError, match="at least 2 channels"):
        compute_adaptive_kuramoto_scores(data, fs)


def test_weighted_kuramoto_short_signal_rejected() -> None:
    """Signals shorter than one epoch are rejected."""
    fs = 100.0
    data = np.random.default_rng(4).standard_normal((2, int(10 * fs)))
    with pytest.raises(ValueError, match="shorter than one epoch"):
        compute_adaptive_kuramoto_scores(data, fs)


def test_compute_weighted_kuramoto_r_shape() -> None:
    """Output shape matches the number of epochs."""
    n_channels, n_epochs, epoch_len = 4, 3, 100
    phases = np.random.default_rng(5).uniform(
        -np.pi, np.pi, (n_channels, n_epochs * epoch_len)
    )
    weights = np.ones((n_channels, n_epochs)) / n_channels
    r = compute_weighted_kuramoto_r(phases, weights, epoch_seconds=1.0, fs=100.0)
    assert r.shape == (n_epochs,)
    assert np.all((r >= 0) & (r <= 1))


def test_phase_locking_weights_shape_and_normalisation() -> None:
    """PLV weights have the expected shape and sum to one per epoch."""
    fs = 100.0
    n_samples = int(60 * fs)
    rng = np.random.default_rng(6)
    phases = rng.uniform(-np.pi, np.pi, (4, n_samples))
    weights = compute_phase_locking_weights(phases, fs, epoch_seconds=30.0)

    assert weights.shape == (4, 2)
    assert np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_plv_mode_scores_detect_coherence_difference() -> None:
    """PLV-weighted adaptive scores separate coherent from incoherent epochs."""
    fs = 100.0
    n_samples = int(60 * fs)
    epoch_samples = int(30 * fs)
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(7)

    # First epoch: four perfectly coherent channels.
    base = np.sin(2 * np.pi * 2.0 * t)
    epoch1 = np.tile(base[:epoch_samples], (4, 1)) + 0.01 * rng.standard_normal(
        (4, epoch_samples)
    )
    # Second epoch: independent phase noise (low coherence).
    epoch2 = np.zeros((4, epoch_samples))
    for c in range(4):
        phase_shift = rng.uniform(0, 2 * np.pi)
        epoch2[c, :] = np.sin(2 * np.pi * 2.0 * t[:epoch_samples] + phase_shift)
    epoch2 += 0.01 * rng.standard_normal((4, epoch_samples))

    data = np.hstack([epoch1, epoch2]).astype(np.float64)
    scores, _ = compute_adaptive_kuramoto_scores(
        data, fs, epoch_seconds=30.0, weight_mode="plv_mean_field"
    )

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0


def test_phase_locking_weights_top_k_shape_and_selection() -> None:
    """Top-k PLV weights are binary and select exactly k channels per epoch."""
    fs = 100.0
    n_samples = int(60 * fs)
    rng = np.random.default_rng(8)
    phases = rng.uniform(-np.pi, np.pi, (6, n_samples))
    weights = compute_phase_locking_weights(phases, fs, epoch_seconds=30.0, top_k=3)

    assert weights.shape == (6, 2)
    assert np.all((weights == 0) | (weights == 1))
    assert np.allclose(weights.sum(axis=0), 3.0)
    # Selected channels are identical across epochs (global selection).
    assert np.array_equal(weights[:, 0], weights[:, 1])


def test_top_k_mode_detects_coherence_difference() -> None:
    """Global top-k PLV adaptive scores separate coherent from incoherent epochs."""
    fs = 100.0
    n_samples = int(60 * fs)
    epoch_samples = int(30 * fs)
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(9)

    # First epoch: four perfectly coherent channels.
    base = np.sin(2 * np.pi * 2.0 * t)
    epoch1 = np.tile(base[:epoch_samples], (4, 1)) + 0.01 * rng.standard_normal(
        (4, epoch_samples)
    )
    # Second epoch: independent phase noise (low coherence).
    epoch2 = np.zeros((4, epoch_samples))
    for c in range(4):
        phase_shift = rng.uniform(0, 2 * np.pi)
        epoch2[c, :] = np.sin(2 * np.pi * 2.0 * t[:epoch_samples] + phase_shift)
    epoch2 += 0.01 * rng.standard_normal((4, epoch_samples))

    data = np.hstack([epoch1, epoch2]).astype(np.float64)
    scores, weights = compute_adaptive_kuramoto_scores(
        data,
        fs,
        epoch_seconds=30.0,
        weight_mode="plv_mean_field",
        top_k=2,
    )

    assert scores.shape == (2,)
    assert weights.shape == (4, 2)
    assert np.allclose(weights.sum(axis=0), 2.0)
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0


def test_top_k_invalid_value_raises() -> None:
    """top_k outside [1, n_channels] is rejected."""
    fs = 100.0
    n_samples = int(60 * fs)
    rng = np.random.default_rng(10)
    data = rng.standard_normal((4, n_samples))

    with pytest.raises(ValueError, match="top_k must be between"):
        compute_adaptive_kuramoto_scores(
            data,
            fs,
            epoch_seconds=30.0,
            weight_mode="plv_mean_field",
            top_k=0,
        )
    with pytest.raises(ValueError, match="top_k must be between"):
        compute_adaptive_kuramoto_scores(
            data,
            fs,
            epoch_seconds=30.0,
            weight_mode="plv_mean_field",
            top_k=5,
        )


def test_top_k_only_with_plv_mode() -> None:
    """top_k is rejected for snr_kurtosis weight mode."""
    fs = 100.0
    n_samples = int(60 * fs)
    rng = np.random.default_rng(11)
    data = rng.standard_normal((4, n_samples))

    with pytest.raises(ValueError, match="top_k is only supported"):
        compute_adaptive_kuramoto_scores(
            data,
            fs,
            epoch_seconds=30.0,
            weight_mode="snr_kurtosis",
            top_k=2,
        )


def test_kurtosis_excess_returns_neutral_for_short_axis() -> None:
    """Fewer than four samples along the axis yields a neutral (zero) kurtosis."""
    # Three samples along the reduction axis: too few for the k2 estimator.
    excess = _kurtosis_excess(np.ones((2, 3)), axis=1)
    assert excess.shape == (2,)
    assert np.array_equal(excess, np.zeros(2))


def test_quality_weights_short_signal_rejected() -> None:
    """``compute_channel_quality_weights`` rejects a sub-epoch signal directly."""
    with pytest.raises(ValueError, match="shorter than one epoch"):
        compute_channel_quality_weights(
            np.zeros((2, 5), dtype=np.float64), fs=100.0, epoch_seconds=30.0
        )


def test_phase_locking_weights_short_signal_rejected() -> None:
    """``compute_phase_locking_weights`` rejects a sub-epoch signal directly."""
    with pytest.raises(ValueError, match="shorter than one epoch"):
        compute_phase_locking_weights(
            np.zeros((2, 5), dtype=np.float64), fs=100.0, epoch_seconds=30.0
        )


def test_unknown_weight_mode_rejected() -> None:
    """An unrecognised ``weight_mode`` is rejected."""
    fs = 100.0
    data = np.random.default_rng(12).standard_normal((4, int(60 * fs)))
    with pytest.raises(ValueError, match="unknown weight_mode"):
        compute_adaptive_kuramoto_scores(
            data, fs, epoch_seconds=30.0, weight_mode="bogus"
        )


@pytest.mark.parametrize(
    "payload",
    [
        np.ones((2, 40), dtype=bool),
        np.ones((2, 40), dtype=np.complex128),
        np.full((2, 40), "0.5"),
    ],
)
def test_public_signal_ingress_rejects_coercive_aliases(payload: Any) -> None:
    with pytest.raises(ValueError, match="data must be a finite real matrix"):
        compute_channel_quality_weights(
            payload,
            fs=10.0,
            band_hz=(1.0, 4.0),
            epoch_seconds=2.0,
        )
    with pytest.raises(ValueError, match="phases must be a finite real matrix"):
        compute_phase_locking_weights(payload, fs=10.0, epoch_seconds=2.0)
    with pytest.raises(ValueError, match="phases must be a finite real matrix"):
        compute_weighted_kuramoto_r(
            payload,
            np.ones((2, 2), dtype=np.float64),
            epoch_seconds=2.0,
            fs=10.0,
        )
    with pytest.raises(ValueError, match="data must be a finite real matrix"):
        compute_adaptive_kuramoto_scores(
            payload,
            fs=10.0,
            band_hz=(1.0, 4.0),
            epoch_seconds=2.0,
        )


@pytest.mark.parametrize(
    "payload",
    [
        np.zeros(4),
        np.empty((0, 4)),
        np.array([[object(), object()]], dtype=object),
        np.array([[0.0, float("nan")]], dtype=np.float64),
    ],
)
def test_matrix_contract_rejects_shape_object_and_non_finite(payload: Any) -> None:
    with pytest.raises(ValueError, match="phases must be a finite real matrix"):
        compute_phase_locking_weights(payload, fs=10.0, epoch_seconds=1.0)


def test_matrix_contract_wraps_array_protocol_failures() -> None:
    class BrokenArray:
        def __array__(self) -> np.ndarray:
            raise TypeError("broken array protocol")

    with pytest.raises(ValueError, match="phases must be a finite real matrix"):
        compute_phase_locking_weights(BrokenArray(), fs=10.0, epoch_seconds=1.0)  # type: ignore[arg-type]


def test_real_numeric_object_matrix_remains_compatible() -> None:
    phases = np.array([[0, 0.5] * 20, [0.1, 0.6] * 20], dtype=object)
    weights = compute_phase_locking_weights(phases, fs=10.0, epoch_seconds=2.0)  # type: ignore[arg-type]

    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=0), 1.0)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan")])
def test_sampling_rate_must_be_positive_finite_real(value: Any) -> None:
    with pytest.raises(ValueError, match="fs must be a positive finite real"):
        compute_phase_locking_weights(
            np.zeros((2, 40), dtype=np.float64),
            fs=value,
            epoch_seconds=2.0,
        )


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan"), 0.01])
def test_epoch_duration_must_define_at_least_one_sample(value: Any) -> None:
    with pytest.raises(ValueError, match="epoch_seconds"):
        compute_phase_locking_weights(
            np.zeros((2, 40), dtype=np.float64),
            fs=10.0,
            epoch_seconds=value,
        )


@pytest.mark.parametrize(
    "band_hz",
    [
        [1.0, 4.0],
        (0.0, 4.0),
        (4.0, 1.0),
        (1.0, 5.0),
        (True, 4.0),
        (1.0, float("nan")),
    ],
)
def test_band_must_be_ordered_inside_nyquist(band_hz: Any) -> None:
    with pytest.raises(ValueError, match="band_hz"):
        compute_channel_quality_weights(
            np.zeros((2, 40), dtype=np.float64),
            fs=10.0,
            band_hz=band_hz,
            epoch_seconds=2.0,
        )


def test_weight_matrix_contract_rejects_broadcast_and_zero_mass() -> None:
    phases = np.zeros((2, 40), dtype=np.float64)

    with pytest.raises(ValueError, match="weights shape"):
        compute_weighted_kuramoto_r(
            phases,
            np.ones((2, 1), dtype=np.float64),
            epoch_seconds=2.0,
            fs=10.0,
        )
    with pytest.raises(ValueError, match="positive mass"):
        compute_weighted_kuramoto_r(
            phases,
            np.zeros((2, 2), dtype=np.float64),
            epoch_seconds=2.0,
            fs=10.0,
        )
    with pytest.raises(ValueError, match="weights must be a finite real matrix"):
        compute_weighted_kuramoto_r(
            phases,
            np.ones((2, 2), dtype=bool),
            epoch_seconds=2.0,
            fs=10.0,
        )
    negative = np.ones((2, 2), dtype=np.float64)
    negative[0, 0] = -0.1
    with pytest.raises(ValueError, match="weights must be non-negative"):
        compute_weighted_kuramoto_r(
            phases,
            negative,
            epoch_seconds=2.0,
            fs=10.0,
        )


def test_weighted_score_rejects_sub_epoch_phase_signal() -> None:
    with pytest.raises(ValueError, match="shorter than one epoch"):
        compute_weighted_kuramoto_r(
            np.zeros((2, 10), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
            epoch_seconds=2.0,
            fs=10.0,
        )


def test_filter_rejects_signal_at_or_below_padding_length() -> None:
    with pytest.raises(ValueError, match="signal must contain more than"):
        compute_channel_quality_weights(
            np.zeros((2, 20), dtype=np.float64),
            fs=10.0,
            band_hz=(1.0, 4.0),
            epoch_seconds=2.0,
        )


def test_constant_signal_quality_weights_fall_back_to_uniform() -> None:
    weights = compute_channel_quality_weights(
        np.zeros((2, 40), dtype=np.float64),
        fs=10.0,
        band_hz=(1.0, 4.0),
        epoch_seconds=2.0,
    )

    assert np.array_equal(weights, np.full((2, 2), 0.5))


def test_boolean_top_k_and_negative_penalty_rejected() -> None:
    phases = np.zeros((2, 40), dtype=np.float64)
    with pytest.raises(ValueError, match="top_k must be an integer"):
        compute_phase_locking_weights(
            phases,
            fs=10.0,
            epoch_seconds=2.0,
            top_k=True,
        )
    with pytest.raises(ValueError, match="kurtosis_penalty_scale"):
        compute_channel_quality_weights(
            phases,
            fs=10.0,
            band_hz=(1.0, 4.0),
            epoch_seconds=2.0,
            kurtosis_penalty_scale=-1.0,
        )


@pytest.mark.parametrize("score_precision", [True, -1, 1.5])
def test_score_precision_is_non_negative_integer(score_precision: Any) -> None:
    with pytest.raises(ValueError, match="score_precision"):
        compute_adaptive_kuramoto_scores(
            np.zeros((2, 40), dtype=np.float64),
            fs=10.0,
            band_hz=(1.0, 4.0),
            epoch_seconds=2.0,
            score_precision=score_precision,
        )
