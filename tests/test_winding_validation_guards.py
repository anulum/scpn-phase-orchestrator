# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend winding validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._winding_validation import (  # noqa: E501
    validate_winding_backend_inputs,
    validate_winding_backend_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phases_flat": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "t": 2,
        "n": 2,
    }
    payload.update(overrides)
    return payload


class TestWindingInputs:
    def test_valid_round_trips(self) -> None:
        phases, t, n = validate_winding_backend_inputs(**_inputs())
        assert (t, n) == (2, 2)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"t": True}, "t must"),
            ({"t": 1}, "t must"),
            ({"n": 0}, "n must"),
            (
                {"phases_flat": np.array([True, False, False, False])},
                "must not contain boolean",
            ),
            (
                {"phases_flat": np.array([0.0 + 1j, 1.0, 2.0, 3.0])},
                "must be real-valued",
            ),
            ({"phases_flat": np.zeros((2, 2))}, "must be one-dimensional"),
            ({"phases_flat": np.array([0.0, np.inf, 2.0, 3.0])}, "only finite values"),
            ({"phases_flat": np.array([0.0, 1.0])}, "does not match t.n"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_winding_backend_inputs(**_inputs(**overrides))


class TestWindingOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_winding_backend_output(np.array([0, 1]), t=2, n=2)
        assert out.tolist() == [0, 1]

    def test_valid_against_matching_reference(self) -> None:
        out = validate_winding_backend_output(
            np.array([0, 1]), t=2, n=2, expected=np.array([0, 1], dtype=np.int64)
        )
        assert out.tolist() == [0, 1]

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False]), "must not contain boolean"),
            (np.array([0.0 + 1j, 1.0]), "must be real-valued"),
            (np.array([0, 1, 2]), r"shape .* must be \(2,\)"),
            (np.array([0.5, 1.0]), "must contain integer"),
            (np.array([5, 0]), "exceeds wrapped-increment bound"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_winding_backend_output(value, t=2, n=2)

    def test_rejects_reference_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="reference shape must match"):
            validate_winding_backend_output(
                np.array([0, 1]), t=2, n=2, expected=np.array([0, 1, 0], dtype=np.int64)
            )

    def test_rejects_reference_divergence(self) -> None:
        with pytest.raises(ValueError, match="diverged from exact"):
            validate_winding_backend_output(
                np.array([0, 1]), t=2, n=2, expected=np.array([1, 0], dtype=np.int64)
            )
