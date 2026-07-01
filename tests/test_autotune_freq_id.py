# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune frequency-identification tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.freq_id as freq_mod
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies


def test_identify_frequencies_assigns_channels_to_nearest_dmd_modes() -> None:
    fs = 128.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.cos(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 17.0 * t),
            np.cos(2.0 * np.pi * 17.0 * t),
        ]
    )

    result = identify_frequencies(data, fs, n_modes=4, rank_threshold=0.0)

    assert result.frequencies.shape == (4,)
    assert result.amplitudes.shape == (4,)
    assert len(result.layer_assignment) == 4
    assert all(
        0 <= index < result.frequencies.size for index in result.layer_assignment
    )
    assigned = result.frequencies[result.layer_assignment]
    assert np.allclose(np.sort(assigned), [5.0, 5.0, 17.0, 17.0], atol=0.25)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([[False, True, False, True]], "boolean"),
        ([[0.0, 1.0 + 0.0j, 0.0, -1.0]], "real-valued"),
        ([[0.0, np.nan, 0.0, -1.0]], "finite"),
    ],
)
def test_identify_frequencies_rejects_non_physical_data_payloads(
    data: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        identify_frequencies(cast(np.ndarray, data), 64.0)


@pytest.mark.parametrize("fs", [True, 0.0, -1.0, np.inf])
def test_identify_frequencies_rejects_invalid_sample_rates(fs: object) -> None:
    with pytest.raises(ValueError, match="fs"):
        identify_frequencies(np.arange(8.0).reshape(1, -1), cast(float, fs))


@pytest.mark.parametrize("n_modes", [True, 0, -1, 1.25])
def test_identify_frequencies_rejects_invalid_mode_counts(n_modes: object) -> None:
    with pytest.raises(ValueError, match="n_modes"):
        identify_frequencies(
            np.sin(np.linspace(0.0, 2.0 * np.pi, 16)).reshape(1, -1),
            64.0,
            n_modes=cast(int, n_modes),
        )


@pytest.mark.parametrize("rank_threshold", [True, -0.1, np.inf])
def test_identify_frequencies_rejects_invalid_rank_thresholds(
    rank_threshold: object,
) -> None:
    with pytest.raises(ValueError, match="rank_threshold"):
        identify_frequencies(
            np.sin(np.linspace(0.0, 2.0 * np.pi, 16)).reshape(1, -1),
            64.0,
            rank_threshold=cast(float, rank_threshold),
        )


def test_identify_frequencies_rejects_no_temporal_dynamics() -> None:
    with pytest.raises(ValueError, match="non-zero temporal dynamics"):
        identify_frequencies(np.ones((2, 16)), 64.0)


def test_identify_frequencies_rejects_short_time_series() -> None:
    with pytest.raises(ValueError, match="Need >= 3 time samples"):
        identify_frequencies(np.asarray([[0.0, 1.0]], dtype=np.float64), 64.0)


def test_identify_frequencies_rejects_non_numeric_data_payloads() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        identify_frequencies(cast(np.ndarray, np.asarray([["x", "y", "z"]])), 64.0)


def test_identify_frequencies_uses_rank_threshold_when_mode_count_is_default() -> None:
    fs = 64.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    data = np.asarray(
        [[0.0, 1.0, 0.0, -1.0] * 16, np.sin(2.0 * np.pi * 4.0 * t)],
        dtype=object,
    )

    result = identify_frequencies(cast(np.ndarray, data), fs)

    assert result.frequencies.size >= 1


def test_identify_frequencies_rejects_degenerate_svd_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_svd(
        _matrix: np.ndarray,
        *,
        full_matrices: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert full_matrices is False
        return (
            np.empty((1, 0), dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    monkeypatch.setattr(freq_mod.np.linalg, "svd", empty_svd)

    with pytest.raises(ValueError, match="non-zero temporal dynamics"):
        identify_frequencies(
            np.asarray([[0.0, 1.0, 0.0, -1.0]], dtype=np.float64),
            64.0,
        )
