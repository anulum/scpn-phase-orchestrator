# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune phase-extraction tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.phase_extract import extract_phases


def test_extract_phases_recovers_sinusoid_dominant_frequency() -> None:
    fs = 128.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    signal = np.sin(2.0 * np.pi * 6.0 * t)

    result = extract_phases(signal, fs)

    assert result.phases.shape == signal.shape
    assert result.amplitudes.shape == signal.shape
    assert result.instantaneous_freq.shape == signal.shape
    assert np.all((result.phases >= 0.0) & (result.phases < 2.0 * np.pi))
    assert np.isclose(result.dominant_freq, 6.0)
    assert np.isclose(np.median(result.instantaneous_freq[8:]), 6.0, atol=0.1)


def test_extract_phases_reports_zero_dominant_frequency_for_dc_signal() -> None:
    result = extract_phases(np.ones(32), 64.0)

    assert result.dominant_freq == 0.0
    assert np.allclose(result.instantaneous_freq, 0.0)


@pytest.mark.parametrize(
    ("signal", "match"),
    [
        ([False, True, False, True], "boolean"),
        ([0.0, 1.0 + 0.0j, 0.0, -1.0], "real-valued"),
        ([0.0, np.inf, 0.0, -1.0], "finite"),
    ],
)
def test_extract_phases_rejects_non_physical_signal_payloads(
    signal: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        extract_phases(cast(np.ndarray, signal), 64.0)


@pytest.mark.parametrize("fs", [True, 0.0, -1.0, np.inf])
def test_extract_phases_rejects_invalid_sample_rates(fs: object) -> None:
    with pytest.raises(ValueError, match="fs"):
        extract_phases(np.arange(4.0), cast(float, fs))


@pytest.mark.parametrize(
    "bandpass",
    [
        (True, 20.0),
        (-1.0, 20.0),
        (30.0, 20.0),
        (2.0, 40.0),
    ],
)
def test_extract_phases_rejects_invalid_bandpass_bounds(
    bandpass: tuple[object, object],
) -> None:
    with pytest.raises(ValueError, match="bandpass"):
        extract_phases(
            np.sin(np.linspace(0.0, 2.0 * np.pi, 64)),
            64.0,
            cast(tuple[float, float], bandpass),
        )


def test_extract_phases_bandpass_keeps_in_band_component() -> None:
    fs = 128.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    signal = np.sin(2.0 * np.pi * 6.0 * t)
    signal += 0.5 * np.sin(2.0 * np.pi * 22.0 * t)

    result = extract_phases(signal, fs, (4.0, 8.0))

    assert np.isclose(result.dominant_freq, 6.0)
