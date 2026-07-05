# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared analytic-phase pipeline tests

"""Tests for the shared raw-signal to analytic-phase pipeline.

The band-pass / Hilbert-phase / phase-consistent-decimation primitives are the
domain-neutral half every capstone adapter reuses, so they are pinned here on
synthetic tones and arrays: the band-pass keeps the in-band tone and rejects the
out-of-band one, the analytic phase advances linearly for a pure tone, the
decimation shortens the field while preserving a constant phase, and the input
guard rejects every malformed block and band.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.analytic_phase_pipeline import (
    analytic_phase,
    bandpass,
    decimate_analytic_phase,
    validate_signals,
)

_TWO_PI = 2.0 * np.pi
_BAND = (4.0, 30.0)


# --------------------------------------------------------------------------- #
# bandpass                                                                     #
# --------------------------------------------------------------------------- #


def test_bandpass_keeps_in_band_and_rejects_out_of_band() -> None:
    fs = 256.0
    n = 2048
    times = np.arange(n) / fs
    in_band = np.sin(_TWO_PI * 10.0 * times)
    out_band = np.sin(_TWO_PI * 60.0 * times)
    signal = np.vstack([in_band + out_band])
    filtered = bandpass(signal, sampling_rate_hz=fs, band_hz=_BAND)
    residual_out = float(np.std(filtered[0] - in_band))
    assert residual_out < 0.2
    assert float(np.std(filtered[0])) > 0.5


def test_bandpass_promotes_a_single_channel() -> None:
    fs = 256.0
    tone = np.sin(_TWO_PI * 12.0 * np.arange(1024) / fs)
    filtered = bandpass(tone, sampling_rate_hz=fs, band_hz=_BAND)
    assert filtered.shape == (1, 1024)


def test_bandpass_rejects_a_band_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=64.0, band_hz=(4.0, 40.0))


def test_bandpass_rejects_a_low_edge_at_or_above_the_high_edge() -> None:
    with pytest.raises(ValueError, match="must be below high"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=256.0, band_hz=(30.0, 30.0))


def test_bandpass_rejects_a_malformed_band() -> None:
    with pytest.raises(ValueError, match="low, high"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=256.0, band_hz=(4.0, 20.0, 30.0))


def test_bandpass_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=0.0, band_hz=_BAND)


def test_bandpass_rejects_a_boolean_rate() -> None:
    with pytest.raises(ValueError, match="positive real"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=True, band_hz=_BAND)


def test_bandpass_rejects_a_non_positive_order() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=256.0, band_hz=_BAND, order=0)


# --------------------------------------------------------------------------- #
# analytic_phase                                                              #
# --------------------------------------------------------------------------- #


def test_analytic_phase_advances_linearly_for_a_pure_tone() -> None:
    fs = 256.0
    hz = 10.0
    tone = np.cos(_TWO_PI * hz * np.arange(1024) / fs)
    phase = analytic_phase(tone)
    increments = np.diff(np.unwrap(phase[0]))
    expected = _TWO_PI * hz / fs
    assert np.allclose(increments[50:-50], expected, atol=1.0e-3)


# --------------------------------------------------------------------------- #
# decimate_analytic_phase                                                     #
# --------------------------------------------------------------------------- #


def test_decimate_reduces_length_and_preserves_a_constant_phase() -> None:
    phases = np.zeros((3, 2400), dtype=np.float64)
    decimated = decimate_analytic_phase(phases, factor=8)
    assert decimated.shape == (3, 300)
    assert np.allclose(decimated, 0.0, atol=1.0e-6)


def test_decimate_with_unit_factor_is_the_identity() -> None:
    phases = np.linspace(-np.pi, np.pi, 200).reshape(2, 100)
    decimated = decimate_analytic_phase(phases, factor=1)
    assert np.array_equal(decimated, phases)


def test_decimate_rejects_a_non_positive_factor() -> None:
    with pytest.raises(ValueError, match="factor"):
        decimate_analytic_phase(np.zeros((2, 80)), factor=0)


def test_decimate_rejects_a_non_integer_factor() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        decimate_analytic_phase(np.zeros((2, 80)), factor=1.5)


def test_decimate_rejects_a_boolean_factor() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        decimate_analytic_phase(np.zeros((2, 80)), factor=True)


# --------------------------------------------------------------------------- #
# validate_signals guard surface                                              #
# --------------------------------------------------------------------------- #


def test_validate_signals_promotes_one_dimensional_input() -> None:
    promoted = validate_signals(np.arange(8.0), "signals")
    assert promoted.shape == (1, 8)
    assert promoted.flags["C_CONTIGUOUS"]


def test_validate_signals_rejects_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        validate_signals(np.ones((2, 4), dtype=bool), "signals")


def test_validate_signals_rejects_complex() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        validate_signals(np.ones((2, 4), dtype=np.complex128), "signals")


def test_validate_signals_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="real float array"):
        validate_signals(np.array([["a", "b"], ["c", "d"]]), "signals")


def test_validate_signals_rejects_a_three_dimensional_block() -> None:
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        validate_signals(np.zeros((2, 2, 2)), "signals")


def test_validate_signals_rejects_an_empty_block() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        validate_signals(np.zeros((2, 0)), "signals")


def test_validate_signals_rejects_a_non_finite_block() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_signals(np.array([[0.0, np.inf], [1.0, 2.0]]), "signals")
