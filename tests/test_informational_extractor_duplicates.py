# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — informational extractor repeated-timestamp parity

"""Prove a repeated event timestamp gives the same phase state on every path.

The Python path drops zero inter-event intervals. The Rust kernel received the
raw train, treated the zero interval as degenerate and returned omega 0 and
quality 0, while the amplitude beside them was computed from the de-duplicated
intervals. Expected values here follow the documented rule, so the test holds
with or without the kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor


def _rule(timestamps: list[float]) -> tuple[float, float, float, float]:
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    frequencies = 1.0 / intervals
    median_hz = float(np.median(frequencies))
    theta = (2.0 * math.pi * median_hz * (timestamps[-1] - timestamps[0])) % (
        2.0 * math.pi
    )
    quality = 1.0 / (1.0 + float(np.std(intervals) / np.mean(intervals)))
    return theta, 2.0 * math.pi * median_hz, float(np.mean(frequencies)), quality


@pytest.mark.parametrize(
    "timestamps",
    [
        [0.0, 1.0, 1.0, 2.0, 3.0],
        [0.0, 0.5, 0.5, 0.5, 2.0, 2.5],
        [0.0, 0.3, 1.1, 1.1, 1.9],
    ],
)
def test_repeated_timestamps_are_ignored(timestamps: list[float]) -> None:
    state = InformationalExtractor().extract(np.array(timestamps), 1.0)[0]
    theta, omega, amplitude, quality = _rule(timestamps)

    assert state.omega == pytest.approx(omega)
    assert state.quality == pytest.approx(quality)
    assert state.amplitude == pytest.approx(amplitude)
    assert math.cos(state.theta - theta) == pytest.approx(1.0)
