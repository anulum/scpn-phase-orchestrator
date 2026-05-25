# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.pac import modulation_index, pac_gate, pac_matrix


def test_modulation_index_is_bounded_for_locked_amplitude_envelope() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    amplitude = 1.0 + np.cos(theta)

    mi = modulation_index(theta, amplitude, n_bins=18)

    assert 0.0 < mi < 1.0


def test_modulation_index_is_zero_for_uniform_amplitude() -> None:
    theta = np.linspace(0.0, 4.0 * np.pi, 720, endpoint=False)
    amplitude = np.ones_like(theta)

    assert modulation_index(theta, amplitude, n_bins=18) == pytest.approx(0.0)


def test_modulation_index_rejects_negative_amplitudes() -> None:
    theta = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    amplitude = np.array([1.0, -0.1, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="non-negative amplitudes"):
        modulation_index(theta, amplitude)


def test_pac_matrix_rejects_negative_amplitude_history() -> None:
    phases = np.zeros((3, 2), dtype=np.float64)
    amplitudes = np.ones((3, 2), dtype=np.float64)
    amplitudes[1, 0] = -0.2

    with pytest.raises(ValueError, match="amplitudes_history"):
        pac_matrix(phases, amplitudes)


def test_pac_matrix_shape_and_range_for_cross_channel_history() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    phases = np.column_stack([theta, theta + np.pi / 2.0])
    amplitudes = np.column_stack([1.0 + np.cos(theta), np.ones_like(theta)])

    matrix = pac_matrix(phases, amplitudes, n_bins=18)

    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    assert matrix[0, 0] > matrix[0, 1]


@pytest.mark.parametrize("n_bins", [True, 1, 0, -3])
def test_pac_rejects_invalid_bin_counts(n_bins: object) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    amplitude = np.ones_like(theta)

    with pytest.raises(ValueError, match="n_bins"):
        modulation_index(theta, amplitude, n_bins=n_bins)


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [(0.31, 0.3, True), (0.29, 0.3, False), (0.3, 0.3, True)],
)
def test_pac_gate_uses_closed_threshold(
    value: float,
    threshold: float,
    expected: bool,
) -> None:
    assert pac_gate(value, threshold) is expected
