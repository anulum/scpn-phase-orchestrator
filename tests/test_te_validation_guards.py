# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend transfer-entropy validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._te_validation import (
    validate_phase_te_backend_inputs,
    validate_te_backend_output,
    validate_te_matrix_backend_inputs,
)

_SERIES = np.array([0.1, 0.2, 0.3], dtype=np.float64)


class TestPhaseTeInputs:
    def test_valid_round_trips(self) -> None:
        s, t, bins = validate_phase_te_backend_inputs(_SERIES, _SERIES, 4)
        assert bins == 4

    @pytest.mark.parametrize(
        ("source", "n_bins", "match"),
        [
            (np.array([True, False]), 4, "source must not contain boolean"),
            (np.array([0.1 + 1j, 0.2]), 4, "source must"),
            (np.zeros((2, 2)), 4, "source must"),
            (np.array([0.1, np.inf]), 4, "source must"),
            (_SERIES, 1, "n_bins must"),
        ],
    )
    def test_rejects_corrupt_input(self, source: Any, n_bins: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_phase_te_backend_inputs(source, _SERIES, n_bins)


class TestTeMatrixInputs:
    def test_valid_round_trips(self) -> None:
        series = np.zeros(6, dtype=np.float64)
        result = validate_te_matrix_backend_inputs(series, 2, 3, 4)
        assert result[1:] == (2, 3, 4)

    @pytest.mark.parametrize(
        ("n_osc", "n_time", "n_bins", "match"),
        [
            (0, 3, 4, "n_osc must"),
            (2, 0, 4, "n_time must"),
            (2, 3, 1, "n_bins must"),
        ],
    )
    def test_rejects_invalid_dimensions(
        self, n_osc: Any, n_time: Any, n_bins: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_te_matrix_backend_inputs(np.zeros(6), n_osc, n_time, n_bins)


class TestTeOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_te_backend_output(0.5, n_bins=4) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array(True), "must not be boolean"),
            (np.array(0.5 + 1j), "must be real"),
            (np.array([0.5, 0.6]), "must be scalar"),
            (-0.5, "must be non-negative"),
            (99.0, r"must not exceed log\(n_bins\)"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_te_backend_output(value, n_bins=4)

    def test_rejects_divergence_from_expected(self) -> None:
        with pytest.raises(ValueError, match="diverged from exact"):
            validate_te_backend_output(0.5, n_bins=4, expected=0.9)
