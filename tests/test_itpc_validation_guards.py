# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend ITPC validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_validation import (
    validate_compute_itpc_backend_inputs,
    validate_compute_itpc_backend_output,
    validate_itpc_persistence_backend_inputs,
    validate_itpc_persistence_backend_output,
)

_PHASES = np.zeros(6, dtype=np.float64)  # n_trials=2 * n_tp=3
_ITPC = np.array([0.5, 0.5, 0.5], dtype=np.float64)


class TestComputeItpcInputs:
    def test_valid_round_trips(self) -> None:
        phases, trials, tp = validate_compute_itpc_backend_inputs(_PHASES, 2, 3)
        assert (trials, tp) == (2, 3)

    @pytest.mark.parametrize(
        ("phases", "n_trials", "n_tp", "match"),
        [
            (_PHASES, True, 3, "n_trials must"),
            (_PHASES, 2, -1, "n_tp must"),
            (np.zeros(5), 2, 3, "phases"),
        ],
    )
    def test_rejects_corrupt_input(
        self, phases: Any, n_trials: Any, n_tp: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_compute_itpc_backend_inputs(phases, n_trials, n_tp)


class TestComputeItpcOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_compute_itpc_backend_output(_ITPC, 3)
        assert out.shape == (3,)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False, False]), "must not contain boolean"),
            (np.array([0.5 + 1j, 0.5, 0.5]), "must be real-valued"),
            (np.zeros(2), "does not match"),
            (np.array([np.inf, 0.5, 0.5]), "only finite values"),
            (np.array([2.0, 0.5, 0.5]), r"must lie in \[0, 1\]"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_compute_itpc_backend_output(value, 3)

    def test_rejects_divergence_from_expected(self) -> None:
        with pytest.raises(ValueError, match="diverged from exact reference"):
            validate_compute_itpc_backend_output(
                _ITPC, 3, expected=np.array([0.9, 0.9, 0.9])
            )


class TestPersistenceOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_itpc_persistence_backend_output(0.5) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True]), "must not contain booleans"),
            (np.array(0.5 + 1j), "must be real-valued"),
            (np.array([0.5, 0.6]), "must be scalar"),
            (np.inf, "must be finite"),
            (2.0, r"must lie in \[0, 1\]"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_itpc_persistence_backend_output(value)


class TestPersistenceInputs:
    def test_valid_round_trips(self) -> None:
        result = validate_itpc_persistence_backend_inputs(
            _PHASES, 2, 3, np.array([0, 1])
        )
        assert result[3].tolist() == [0, 1]

    def test_rejects_two_dimensional_pause_indices(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional integer array"):
            validate_itpc_persistence_backend_inputs(
                _PHASES, 2, 3, np.zeros((2, 2), dtype=np.int64)
            )

    def test_rejects_non_integer_pause_indices(self) -> None:
        with pytest.raises(ValueError, match="only integer indices"):
            validate_itpc_persistence_backend_inputs(
                _PHASES, 2, 3, np.array([0.5, 1.5])
            )
