# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend psychedelic-entropy validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._psychedelic_validation import (  # noqa: E501
    validate_psychedelic_backend_inputs,
    validate_psychedelic_entropy_backend_output,
)

_PHASES = np.array([0.1, 0.2, 0.3], dtype=np.float64)


class _ArrayProtocolFailure:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise TypeError("array protocol unavailable")


class TestPsychedelicInputs:
    def test_valid_round_trips(self) -> None:
        phases, n_bins = validate_psychedelic_backend_inputs(_PHASES, 4)
        assert n_bins == 4

    @pytest.mark.parametrize(
        ("phases", "n_bins", "exc", "match"),
        [
            (np.array([True, False]), 4, ValueError, "must not contain boolean"),
            (np.array([0.1 + 1j, 0.2]), 4, ValueError, "must be real-valued"),
            (np.zeros((2, 2)), 4, ValueError, "must be one-dimensional"),
            (np.array([0.1, np.inf]), 4, ValueError, "only finite values"),
            (_PHASES, True, TypeError, "n_bins must be an integer"),
            (_PHASES, 1, ValueError, "n_bins must be greater"),
        ],
    )
    def test_rejects_corrupt_input(
        self, phases: Any, n_bins: Any, exc: type, match: str
    ) -> None:
        with pytest.raises(exc, match=match):
            validate_psychedelic_backend_inputs(phases, n_bins)

    def test_rejects_array_protocol_failure(self) -> None:
        with pytest.raises(ValueError, match="finite one-dimensional"):
            validate_psychedelic_backend_inputs(_ArrayProtocolFailure(), 4)


class TestPsychedelicEntropyOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_psychedelic_entropy_backend_output(0.5, 4) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([True]), "must not contain boolean"),
            (np.array(0.5 + 1j), "must be real-valued"),
            (np.array([0.5, 0.6]), "must be scalar"),
            (np.inf, "must be finite"),
            (99.0, r"must lie in \[0, log\(n_bins\)\]"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_psychedelic_entropy_backend_output(value, 4)

    def test_rejects_array_protocol_failure(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            validate_psychedelic_entropy_backend_output(_ArrayProtocolFailure(), 4)
