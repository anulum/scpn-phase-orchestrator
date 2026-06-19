# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend PID input/output validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._pid_validation import (
    validate_pid_backend_inputs,
    validate_pid_scalar_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase_history_flat": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "t": 2,
        "n": 2,
        "group_a": np.array([0], dtype=np.int64),
        "group_b": np.array([1], dtype=np.int64),
        "n_bins": 4,
    }
    payload.update(overrides)
    return payload


def _call(**overrides: Any) -> Any:
    return validate_pid_backend_inputs(**_inputs(**overrides))


class TestPidBackendInputs:
    def test_valid_round_trips(self) -> None:
        history, t, n, ga, gb, bins = _call()
        assert (t, n, bins) == (2, 2, 4)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"t": True}, "t must be an integer"),
            ({"t": -1}, "t must be >= 0"),
            ({"n": 0}, "n must be >= 1"),
            ({"n_bins": 1}, "n_bins must be >= 2"),
            (
                {"phase_history_flat": np.array([True, False, False, False])},
                "must not contain boolean",
            ),
            (
                {"phase_history_flat": np.array([0.0 + 1j, 1.0, 2.0, 3.0])},
                "phase_history must be real-valued",
            ),
            (
                {"phase_history_flat": np.zeros((2, 2))},
                "phase_history must be one-dimensional",
            ),
            (
                {"phase_history_flat": np.array([0.0, np.inf, 2.0, 3.0])},
                "phase_history must contain only finite",
            ),
            ({"phase_history_flat": np.array([0.0, 1.0])}, "does not match t.n"),
            ({"group_a": np.array([True])}, "group_a must not contain boolean"),
            ({"group_a": np.array([0.0 + 1j])}, "group_a must be integer-valued"),
            (
                {"group_a": np.zeros((1, 1), dtype=np.int64)},
                "group_a must be one-dimensional",
            ),
            ({"group_a": np.array([5])}, r"group_a indices must lie in \[0, 2\)"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _call(**overrides)


class TestPidScalarOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_pid_scalar_output(0.5, name="synergy") == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (True, "must not be a boolean"),
            (np.array(0.0 + 1j), "must be a real scalar"),
            (np.array([0.5, 0.6]), "must be a scalar"),
            (-0.5, "finite and non-negative"),
            (np.inf, "finite and non-negative"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_pid_scalar_output(value, name="synergy")
