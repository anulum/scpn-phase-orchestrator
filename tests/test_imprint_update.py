# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint update tests

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


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"decay_rate": np.nan, "saturation": 1.0}, "decay_rate"),
        ({"decay_rate": True, "saturation": 1.0}, "decay_rate"),
        ({"decay_rate": 0.1, "saturation": np.inf}, "saturation"),
        ({"decay_rate": 0.1, "saturation": True}, "saturation"),
    ],
)
def test_constructor_rejects_nonfinite_or_bool_config(kwargs, field):
    with pytest.raises(ValueError, match=field):
        ImprintModel(**kwargs)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, True])
def test_update_rejects_invalid_dt(dt):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="dt"):
        model.update(_state([0.0]), np.zeros(1), dt=dt)


@pytest.mark.parametrize(
    "exposure",
    [np.ones((1, 1)), np.ones(2), np.array([np.nan]), np.array([-0.1])],
)
def test_update_rejects_invalid_exposure(exposure):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="exposure"):
        model.update(_state([0.0]), exposure, dt=0.1)


@pytest.mark.parametrize(
    "state",
    [
        ImprintState(m_k=np.ones((1, 1)), last_update=0.0),
        ImprintState(m_k=np.array([np.nan]), last_update=0.0),
        ImprintState(m_k=np.array([-0.1]), last_update=0.0),
        ImprintState(m_k=np.array([0.0]), last_update=np.inf),
    ],
)
def test_update_rejects_invalid_state(state):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="imprint|last_update|m_k"):
        model.update(state, np.zeros(1), dt=0.1)


@pytest.mark.parametrize(
    "knm",
    [np.ones((2, 3)), np.ones(2), np.array([[0.0, np.nan], [0.0, 0.0]])],
)
def test_modulate_coupling_rejects_invalid_knm(knm):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="knm"):
        model.modulate_coupling(knm, _state([0.0, 0.1]))


@pytest.mark.parametrize(
    "alpha",
    [np.ones((2, 3)), np.ones(2), np.array([[0.0, np.nan], [0.0, 0.0]])],
)
def test_modulate_lag_rejects_invalid_alpha(alpha):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="alpha"):
        model.modulate_lag(alpha, _state([0.0, 0.1]))


@pytest.mark.parametrize(
    "mu",
    [np.ones((2, 1)), np.ones(3), np.array([0.0, np.nan])],
)
def test_modulate_mu_rejects_invalid_mu(mu):
    model = ImprintModel(decay_rate=0.1, saturation=1.0)
    with pytest.raises(ValueError, match="mu"):
        model.modulate_mu(mu, _state([0.0, 0.1]))


class TestImprintUpdatePipelineWiring:
    """Pipeline: engine → exposure → imprint update → modulate K → engine."""

    def test_imprint_feedback_loop(self):
        """Engine → R as exposure → ImprintModel.update → modulate_coupling
        → engine with boosted K. Complete imprint feedback loop."""
        import numpy as np

        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 6
        cs = CouplingBuilder().build(n, 0.3, 0.3)
        model = ImprintModel(decay_rate=0.01, saturation=5.0)
        imprint = _state(np.zeros(n).tolist())
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)

        for _ in range(100):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha)
            r, _ = compute_order_parameter(phases)
            exposure = np.full(n, r)
            imprint = model.update(imprint, exposure, dt=0.01)

        knm_boosted = model.modulate_coupling(cs.knm, imprint)
        assert np.sum(knm_boosted) > np.sum(cs.knm), (
            "Imprint feedback must boost coupling"
        )
        assert np.all(imprint.m_k >= 0.0)
