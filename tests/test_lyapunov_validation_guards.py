# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend Lyapunov validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _lyapunov_validation as lyapunov_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_validation import (  # noqa: E501
    validate_lyapunov_backend_inputs,
    validate_lyapunov_backend_output,
)

_KNM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


class _ArrayFails:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise TypeError("array conversion rejected")


class _ObjectDtypeFails:
    def __array__(self, dtype: object = None) -> np.ndarray:
        if dtype is object or dtype == np.dtype(object):
            raise TypeError("object coercion rejected")
        return np.array([0.1, 0.2, 0.3], dtype=np.float64)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phases_init": np.array([0.1, 0.2], dtype=np.float64),
        "omegas": np.array([0.0, 0.0], dtype=np.float64),
        "knm": _KNM,
        "alpha": np.zeros((2, 2), dtype=np.float64),
        "dt": 0.01,
        "n_steps": 10,
        "qr_interval": 5,
        "zeta": 0.0,
        "psi": 0.0,
    }
    payload.update(overrides)
    return payload


def _call(**overrides: Any) -> Any:
    return validate_lyapunov_backend_inputs(**_inputs(**overrides))


def test_complex_alias_helper_handles_failure_and_complex_dtype() -> None:
    assert lyapunov_validation._contains_complex_alias(_ArrayFails()) is False
    assert (
        lyapunov_validation._contains_complex_alias(
            np.array([1.0 + 0.0j], dtype=np.complex128)
        )
        is True
    )


class TestLyapunovInputs:
    def test_valid_round_trips(self) -> None:
        result = _call()
        assert result[5] == 10  # n_steps

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"phases_init": np.array([True, False])}, "phases_init must not contain"),
            ({"phases_init": _ObjectDtypeFails()}, "omegas shape .* does not match"),
            ({"phases_init": np.array(["bad", "payload"])}, "phases_init must"),
            (
                {"phases_init": np.array(["0.1", "0.2"])},
                "phases_init must contain only real numbers",
            ),
            ({"phases_init": np.zeros((2, 2))}, "phases_init.*one-dimensional"),
            ({"omegas": np.array([0.0, 0.0, 0.0])}, "omegas shape .* does not match"),
            ({"knm": np.array([[0.0, 1j], [1j, 0.0]])}, "knm must"),
            ({"knm": np.array([["bad", "payload"], ["data", "0.0"]])}, "knm must"),
            (
                {"knm": np.array([["0.0", "1.0"], ["1.0", "0.0"]])},
                "knm must contain only real numbers",
            ),
            ({"alpha": np.zeros((2, 3), dtype=np.float64)}, "alpha shape"),
            ({"knm": np.array([[True, False], [False, True]])}, "knm must"),
            ({"knm": np.array([[0.0, np.inf], [np.inf, 0.0]])}, "knm must"),
            ({"knm": np.array([[1.0, 1.0], [1.0, 0.0]])}, "diagonal"),
            ({"phases_init": np.array([0.1 + 1j, 0.2])}, "phases_init must"),
            ({"phases_init": np.array([0.1, np.inf])}, "phases_init must"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"dt": True}, "dt must be a finite real"),
            ({"dt": float("inf")}, "dt must be finite"),
            ({"n_steps": -1}, "n_steps must be >="),
            ({"n_steps": True}, "n_steps must be an integer"),
            ({"qr_interval": 0}, "qr_interval must be >="),
            ({"zeta": -0.5}, "zeta must be non-negative"),
            ({"psi": float("inf")}, "psi must be finite"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _call(**overrides)


class TestLyapunovOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_lyapunov_backend_output(np.array([1.0, -2.0]), 2)
        assert out.tolist() == [1.0, -2.0]

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False]), "must not contain boolean"),
            (np.array([1.0 + 1j, 2.0]), "must be real-valued"),
            (np.array(["not-a-number", "still-bad"]), "must be numeric"),
            (np.array(["1.0", "0.0"]), "must contain only real numbers"),
            (np.array([1.0, 2.0, 3.0]), "does not match"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_lyapunov_backend_output(value, 2)
