# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend spatial-modulator validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_validation import (  # noqa: E501
    validate_spatial_modulator_inputs,
    validate_spatial_modulator_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "k_nm_flat": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        "positions_flat": np.array([0.0, 1.0], dtype=np.float64),
        "n": 2,
        "dim": 1,
        "k_base": 1.0,
        "decay_form_code": 0,
        "decay_exponent": 1.0,
        "decay_length_scale": 1.0,
        "epsilon": 1.0e-12,
    }
    payload.update(overrides)
    return payload


def _call(**overrides: Any) -> Any:
    return validate_spatial_modulator_inputs(**_inputs(**overrides))


class TestSpatialModulatorInputs:
    def test_valid_round_trips(self) -> None:
        k, p, n, dim, k_base, form, exp, length, eps = _call()
        assert (n, dim, form) == (2, 1, 0)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": True}, "n must be a positive integer"),
            ({"n": 0}, "n must be positive"),
            ({"k_base": "x"}, "k_base must be a finite real scalar"),
            ({"k_base": float("inf")}, "k_base must be finite"),
            ({"decay_exponent": 0.0}, "decay_exponent must be positive"),
            ({"decay_length_scale": 0.0}, "decay_length_scale must be positive"),
            ({"epsilon": 0.0}, "epsilon must be positive"),
            ({"decay_form_code": True}, "decay_form_code must be 0, 1, 2, or 3"),
            ({"decay_form_code": 5}, "decay_form_code must be 0, 1, 2, or 3"),
            (
                {"k_nm_flat": np.array([True, False, False, False])},
                "k_nm_flat must be finite real-valued",
            ),
            (
                {"k_nm_flat": np.array([0.0, 1.0, 1.0])},
                "k_nm_flat length 3 does not match 4",
            ),
            (
                {"k_nm_flat": np.array([0.0, np.inf, 1.0, 0.0])},
                "k_nm_flat must contain only finite",
            ),
            (
                {"k_nm_flat": np.array([1.0, 1.0, 1.0, 0.0])},
                "k_nm_flat diagonal must be zero",
            ),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _call(**overrides)


class TestSpatialModulatorOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_spatial_modulator_output(
            np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64), n=2
        )
        assert out.shape == (4,)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False, False, False]), "finite real-valued"),
            (np.array([0.0 + 1j, 0.0, 0.0, 0.0]), "finite real-valued"),
            (np.array([0.0, 0.5]), "does not match"),
            (np.array([0.0, np.inf, 0.5, 0.0]), "only finite values"),
            (np.array([1.0, 0.5, 0.5, 0.0]), "diagonal must be zero"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_spatial_modulator_output(value, n=2)
