# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend NPE/phase-distance validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._npe_validation import (
    validate_npe_backend_inputs,
    validate_npe_backend_output,
    validate_phase_distance_backend_input,
    validate_phase_distance_backend_output,
)

_DIST = np.array([0.0, 0.1, 0.1, 0.0], dtype=np.float64)


class TestPhaseDistanceInput:
    def test_valid_round_trips(self) -> None:
        out = validate_phase_distance_backend_input(np.array([0.1, 0.2]))
        assert out.shape == (2,)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False]), "must not contain boolean"),
            (np.array([0.1 + 1j, 0.2]), "must contain real-valued"),
            (np.zeros((2, 2)), "must be one-dimensional"),
            (np.array([0.1, np.inf]), "only finite values"),
        ],
    )
    def test_rejects_corrupt_input(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_phase_distance_backend_input(value)


class TestPhaseDistanceOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_phase_distance_backend_output(_DIST, n_phases=2)
        assert out.shape == (2, 2)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True, False, False, False]), "must not contain booleans"),
            (np.array([0.0 + 1j, 0.1, 0.1, 0.0]), "must contain real values"),
            (np.zeros(3), "does not match"),
            (np.array([0.0, np.inf, np.inf, 0.0]), "must be finite"),
            (np.array([0.0, 4.0, 4.0, 0.0]), r"must lie in \[0, pi\]"),
            (np.array([0.0, 0.1, 0.2, 0.0]), "must be symmetric"),
            (np.array([1.0, 0.1, 0.1, 0.0]), "diagonal must be zero"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_phase_distance_backend_output(value, n_phases=2)

    def test_rejects_divergence_from_expected(self) -> None:
        with pytest.raises(ValueError, match="must match exact circular"):
            validate_phase_distance_backend_output(
                _DIST,
                n_phases=2,
                expected=np.array([0.0, 0.9, 0.9, 0.0], dtype=np.float64),
            )


class TestNpeInputs:
    def test_valid_round_trips(self) -> None:
        phases, radius = validate_npe_backend_inputs(np.array([0.1, 0.2]), 1.0)
        assert radius == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("max_radius", "match"),
        [
            (True, "max_radius must be a finite"),
            (-0.5, "must be finite and non-negative"),
            (10.0, "must not exceed pi"),
        ],
    )
    def test_rejects_invalid_max_radius(self, max_radius: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_npe_backend_inputs(np.array([0.1, 0.2]), max_radius)


class TestNpeOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_npe_backend_output(0.5) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (True, "must be a real scalar"),
            (np.inf, "must be finite"),
            (1.5, r"must lie in \[0, 1\]"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_npe_backend_output(value)
