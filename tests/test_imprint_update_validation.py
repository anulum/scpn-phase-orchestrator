# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint update input-validation tests

"""Input-validation tests for the ImprintModel update and modulation methods.

ImprintState is deliberately validation-free; the ImprintModel consumer performs
the checks. These tests exercise that consumer-side surface: boolean and
non-numeric vectors/matrices, a non-ImprintState argument, a negative
``last_update`` timestamp, and a non-dictionary attribution map.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel


def _model() -> ImprintModel:
    """Return a standard imprint model."""
    return ImprintModel(decay_rate=0.5, saturation=10.0)


def _state(last_update: float = 0.0) -> ImprintState:
    """Return a valid two-oscillator imprint state."""
    return ImprintState(
        m_k=np.array([0.5, 1.5], dtype=np.float64), last_update=last_update
    )


def test_update_rejects_a_boolean_exposure_vector() -> None:
    with pytest.raises(ValueError, match="exposure must not contain boolean values"):
        _model().update(_state(), np.array([True, False]), dt=0.1)


def test_update_rejects_a_non_numeric_exposure_vector() -> None:
    exposure = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="exposure must be numeric"):
        _model().update(_state(), cast("np.ndarray", exposure), dt=0.1)


def test_modulate_coupling_rejects_a_boolean_matrix() -> None:
    knm = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="knm must not contain boolean values"):
        _model().modulate_coupling(knm, _state())


def test_modulate_coupling_rejects_a_non_numeric_matrix() -> None:
    knm = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="knm must be numeric"):
        _model().modulate_coupling(cast("np.ndarray", knm), _state())


def test_modulate_lag_rejects_a_non_numeric_matrix() -> None:
    alpha = np.array([["x", "y"], ["z", "w"]], dtype=object)
    with pytest.raises(ValueError, match="alpha must be numeric"):
        _model().modulate_lag(cast("np.ndarray", alpha), _state())


def test_validated_state_rejects_a_non_imprint_state() -> None:
    with pytest.raises(ValueError, match="imprint state must be an ImprintState"):
        _model().modulate_mu(np.array([1.0, 2.0]), cast("ImprintState", "not a state"))


def test_validated_state_rejects_a_negative_last_update() -> None:
    bad_state = _state(last_update=-1.0)
    with pytest.raises(ValueError, match="last_update must be non-negative"):
        _model().modulate_mu(np.array([1.0, 2.0]), bad_state)


def test_validated_state_rejects_a_non_dictionary_attribution() -> None:
    bad_state = ImprintState(
        m_k=np.array([0.5, 1.5], dtype=np.float64),
        last_update=0.0,
        attribution=cast("dict[str, float]", ["not", "a", "dict"]),
    )
    with pytest.raises(ValueError, match="imprint attribution must be a dictionary"):
        _model().modulate_mu(np.array([1.0, 2.0]), bad_state)


def test_modulate_mu_accepts_valid_inputs() -> None:
    result = _model().modulate_mu(np.array([2.0, 4.0]), _state())
    # mu * (1 + m_k) with m_k = [0.5, 1.5] -> [2*1.5, 4*2.5] = [3.0, 10.0].
    np.testing.assert_allclose(result, [3.0, 10.0], atol=1.0e-12)
