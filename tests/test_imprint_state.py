# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint state + model tests

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel


# ---------------------------------------------------------------------------
# ImprintState: immutability and data contracts
# ---------------------------------------------------------------------------


class TestImprintStateContracts:
    """Verify the frozen dataclass contract: immutability, array preservation,
    attribution semantics, and edge cases."""

    def test_frozen_rejects_mutation(self):
        """Frozen dataclass must prevent all field mutation."""
        state = ImprintState(m_k=np.zeros(4), last_update=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.last_update = 1.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.m_k = np.ones(4)

    def test_m_k_array_preserved_exactly(self):
        """m_k values must survive construction without modification."""
        m_k = np.array([0.1, 0.2, 0.3, 0.4])
        state = ImprintState(m_k=m_k, last_update=5.0)
        np.testing.assert_array_equal(state.m_k, m_k)
        assert state.last_update == 5.0

    def test_m_k_dtype_preserved(self):
        """Float64 arrays must retain their dtype (not downcast to float32)."""
        state = ImprintState(m_k=np.array([1, 2, 3], dtype=np.float64), last_update=0.0)
        assert state.m_k.dtype == np.float64

    def test_default_attribution_is_empty_dict(self):
        state = ImprintState(m_k=np.zeros(2), last_update=0.0)
        assert state.attribution == {}
        assert isinstance(state.attribution, dict)

    def test_attribution_preserves_layer_contributions(self):
        state = ImprintState(
            m_k=np.zeros(4), last_update=1.0,
            attribution={"layer_0": 0.5, "layer_1": 0.3, "layer_2": 0.2},
        )
        assert state.attribution["layer_0"] == 0.5
        assert sum(state.attribution.values()) == pytest.approx(1.0)

    def test_zero_length_m_k(self):
        """Edge case: empty oscillator set must not crash."""
        state = ImprintState(m_k=np.array([]), last_update=0.0)
        assert len(state.m_k) == 0


# ---------------------------------------------------------------------------
# ImprintModel: decay/accumulation dynamics
# ---------------------------------------------------------------------------


class TestImprintModelDynamics:
    """Verify the exponential decay + accumulation physics:
    m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure * dt,
    clipped to [0, saturation]."""

    @pytest.fixture()
    def model(self):
        return ImprintModel(decay_rate=0.1, saturation=1.0)

    def test_zero_exposure_pure_decay(self, model):
        """With zero exposure, m_k must decay exponentially."""
        state = ImprintState(m_k=np.array([1.0, 0.5, 0.25]), last_update=0.0)
        result = model.update(state, exposure=np.zeros(3), dt=1.0)

        expected_decay = np.exp(-0.1)
        np.testing.assert_allclose(result.m_k, state.m_k * expected_decay, rtol=1e-12)

    def test_accumulation_increases_m_k(self, model):
        """Positive exposure must increase m_k above decayed value."""
        state = ImprintState(m_k=np.array([0.0, 0.0]), last_update=0.0)
        result = model.update(state, exposure=np.array([1.0, 2.0]), dt=0.5)

        assert result.m_k[0] > 0.0
        assert result.m_k[1] > result.m_k[0], "Higher exposure → higher m_k"
        np.testing.assert_allclose(result.m_k, [0.5, 1.0], rtol=1e-12)

    def test_saturation_clipping(self):
        """m_k must never exceed saturation, regardless of exposure."""
        model = ImprintModel(decay_rate=0.0, saturation=0.5)
        state = ImprintState(m_k=np.array([0.4]), last_update=0.0)
        result = model.update(state, exposure=np.array([10.0]), dt=1.0)

        assert result.m_k[0] == pytest.approx(0.5), "Must clip at saturation"

    def test_non_negativity_clipping(self):
        """m_k must never go below 0, even with extreme decay."""
        model = ImprintModel(decay_rate=100.0, saturation=1.0)
        state = ImprintState(m_k=np.array([0.001]), last_update=0.0)
        result = model.update(state, exposure=np.zeros(1), dt=10.0)

        assert result.m_k[0] >= 0.0

    def test_timestamp_advances(self, model):
        """last_update must advance by dt each step."""
        state = ImprintState(m_k=np.zeros(2), last_update=3.0)
        result = model.update(state, exposure=np.zeros(2), dt=0.5)
        assert result.last_update == pytest.approx(3.5)

    def test_attribution_preserved_through_update(self, model):
        """Attribution dict must be copied through updates unchanged."""
        state = ImprintState(
            m_k=np.array([0.1, 0.2]), last_update=0.0,
            attribution={"src": 0.7},
        )
        result = model.update(state, exposure=np.ones(2), dt=0.1)
        assert result.attribution == {"src": 0.7}

    def test_multi_step_convergence(self):
        """Under constant exposure, m_k should converge to
        exposure / decay_rate (steady-state balance)."""
        model = ImprintModel(decay_rate=1.0, saturation=100.0)
        state = ImprintState(m_k=np.array([0.0]), last_update=0.0)
        exposure = np.array([2.0])

        for _ in range(5000):
            state = model.update(state, exposure, dt=0.01)

        # Steady state: dm/dt = 0 → -decay*m + exposure = 0 → m = exposure/decay = 2.0
        assert state.m_k[0] == pytest.approx(2.0, rel=0.01)

    def test_negative_decay_rate_rejected(self):
        with pytest.raises(ValueError, match="decay_rate must be non-negative"):
            ImprintModel(decay_rate=-0.1, saturation=1.0)

    def test_zero_saturation_rejected(self):
        with pytest.raises(ValueError, match="saturation must be positive"):
            ImprintModel(decay_rate=0.1, saturation=0.0)


# ---------------------------------------------------------------------------
# ImprintModel: modulation functions
# ---------------------------------------------------------------------------


class TestImprintModulation:
    """Verify that imprint-based modulation of K, alpha, and mu
    satisfies the expected mathematical contracts."""

    @pytest.fixture()
    def model(self):
        return ImprintModel(decay_rate=0.1, saturation=10.0)

    def test_coupling_modulation_scales_rows(self, model):
        """modulate_coupling: K_ij → K_ij * (1 + m_i).
        Row i is scaled by oscillator i's imprint."""
        knm = np.ones((3, 3))
        np.fill_diagonal(knm, 0.0)
        imprint = ImprintState(m_k=np.array([0.0, 1.0, 2.0]), last_update=0.0)

        result = model.modulate_coupling(knm, imprint)
        # Row 0: scale by (1+0)=1, Row 1: scale by (1+1)=2, Row 2: scale by (1+2)=3
        np.testing.assert_allclose(result[0], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(result[1], [2.0, 0.0, 2.0])
        np.testing.assert_allclose(result[2], [3.0, 3.0, 0.0])

    def test_zero_imprint_identity(self, model):
        """Zero imprint must not change K, alpha, or mu."""
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        alpha = np.array([[0.0, 0.1], [-0.1, 0.0]])
        mu = np.array([0.3, 0.7])
        imprint = ImprintState(m_k=np.zeros(2), last_update=0.0)

        np.testing.assert_allclose(model.modulate_coupling(knm, imprint), knm)
        np.testing.assert_allclose(model.modulate_lag(alpha, imprint), alpha)
        np.testing.assert_allclose(model.modulate_mu(mu, imprint), mu)

    def test_lag_modulation_antisymmetric(self, model):
        """modulate_lag adds m_i - m_j to alpha_ij.
        The offset must be antisymmetric: offset_ij = -offset_ji."""
        alpha = np.zeros((3, 3))
        imprint = ImprintState(m_k=np.array([0.0, 0.5, 1.0]), last_update=0.0)

        result = model.modulate_lag(alpha, imprint)
        # offset[0,1] = 0.0 - 0.5 = -0.5
        # offset[1,0] = 0.5 - 0.0 = +0.5
        assert result[0, 1] == pytest.approx(-0.5)
        assert result[1, 0] == pytest.approx(0.5)
        # Antisymmetry: result + result.T = 0
        np.testing.assert_allclose(result + result.T, 0.0, atol=1e-14)

    def test_mu_modulation_scales_bifurcation(self, model):
        """modulate_mu: μ_k → μ_k * (1 + m_k).
        Higher imprint → stronger bifurcation (easier oscillation)."""
        mu = np.array([0.5, 0.5])
        imprint = ImprintState(m_k=np.array([0.0, 1.0]), last_update=0.0)

        result = model.modulate_mu(mu, imprint)
        assert result[0] == pytest.approx(0.5)  # (1+0)*0.5
        assert result[1] == pytest.approx(1.0)  # (1+1)*0.5


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
