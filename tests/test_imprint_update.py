# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel


def _state(vals):
    return ImprintState(m_k=np.array(vals, dtype=np.float64), last_update=0.0)


def test_decay_reduces_m_k():
    model = ImprintModel(decay_rate=1.0, saturation=10.0)
    state = _state([1.0, 2.0, 3.0])
    exposure = np.zeros(3)
    updated = model.update(state, exposure, dt=1.0)
    # exp(-1.0) * original
    expected = np.array([1.0, 2.0, 3.0]) * np.exp(-1.0)
    np.testing.assert_allclose(updated.m_k, expected, atol=1e-10)


def test_exposure_accumulation():
    model = ImprintModel(decay_rate=0.0, saturation=10.0)
    state = _state([0.0, 0.0])
    exposure = np.array([1.0, 2.0])
    updated = model.update(state, exposure, dt=0.5)
    np.testing.assert_allclose(updated.m_k, [0.5, 1.0], atol=1e-12)


def test_saturation_caps_m_k():
    model = ImprintModel(decay_rate=0.0, saturation=1.0)
    state = _state([0.8, 0.9])
    exposure = np.array([5.0, 5.0])
    updated = model.update(state, exposure, dt=1.0)
    np.testing.assert_allclose(updated.m_k, [1.0, 1.0], atol=1e-12)


def test_modulate_coupling_scales_rows():
    model = ImprintModel(decay_rate=0.1, saturation=5.0)
    knm = np.ones((3, 3)) * 0.5
    np.fill_diagonal(knm, 0.0)
    imprint = _state([1.0, 0.0, 0.5])
    result = model.modulate_coupling(knm, imprint)
    # Row 0 scaled by 2.0, row 1 by 1.0, row 2 by 1.5
    np.testing.assert_allclose(result[0, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(result[1, 0], 0.5, atol=1e-12)
    np.testing.assert_allclose(result[2, 0], 0.75, atol=1e-12)


def test_modulate_lag_shifts_alpha():
    model = ImprintModel(decay_rate=0.1, saturation=5.0)
    alpha = np.zeros((3, 3))
    imprint = _state([0.1, 0.2, 0.3])
    result = model.modulate_lag(alpha, imprint)
    # Antisymmetric: result[i,j] = m_k[i] - m_k[j]
    np.testing.assert_allclose(result[0, 1], -0.1, atol=1e-12)
    np.testing.assert_allclose(result[1, 0], 0.1, atol=1e-12)
    np.testing.assert_allclose(result[0, 2], -0.2, atol=1e-12)


def test_modulate_lag_preserves_antisymmetry():
    model = ImprintModel(decay_rate=0.1, saturation=5.0)
    n = 4
    # Start with antisymmetric alpha
    alpha = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            alpha[i, j] = 0.5 * (i - j)
            alpha[j, i] = -alpha[i, j]
    imprint = _state([0.1, 0.3, 0.0, 0.2])
    result = model.modulate_lag(alpha, imprint)
    # result + result^T should be zero (antisymmetry preserved)
    np.testing.assert_allclose(result + result.T, 0.0, atol=1e-12)


def test_last_update_advances():
    model = ImprintModel(decay_rate=0.1, saturation=5.0)
    state = _state([0.0])
    updated = model.update(state, np.zeros(1), dt=0.5)
    assert updated.last_update == pytest.approx(0.5)


def test_negative_decay_rate_raises():
    with pytest.raises(ValueError, match="decay_rate"):
        ImprintModel(decay_rate=-0.1, saturation=1.0)


def test_zero_saturation_raises():
    with pytest.raises(ValueError, match="saturation"):
        ImprintModel(decay_rate=0.1, saturation=0.0)
