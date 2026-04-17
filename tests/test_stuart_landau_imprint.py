# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau imprint tests

"""Tests for Stuart-Landau imprint extensions (modulate_mu)."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel


class TestModulateMu:
    def test_zero_imprint_identity(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.array([1.0, 2.0, 3.0])
        imp = ImprintState(m_k=np.zeros(3), last_update=0.0)
        result = model.modulate_mu(mu, imp)
        np.testing.assert_allclose(result, mu)

    def test_positive_imprint_amplifies(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.array([1.0, 1.0])
        imp = ImprintState(m_k=np.array([0.5, 1.0]), last_update=0.0)
        result = model.modulate_mu(mu, imp)
        np.testing.assert_allclose(result, [1.5, 2.0])

    def test_shape_preserved(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.ones(8)
        imp = ImprintState(m_k=np.linspace(0, 1, 8), last_update=0.0)
        result = model.modulate_mu(mu, imp)
        assert result.shape == (8,)

    def test_negative_mu_stays_negative(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.array([-1.0])
        imp = ImprintState(m_k=np.array([0.5]), last_update=0.0)
        result = model.modulate_mu(mu, imp)
        assert result[0] < 0.0

    def test_mu_proportional_to_imprint(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.array([2.0])
        imp_low = ImprintState(m_k=np.array([0.1]), last_update=0.0)
        imp_high = ImprintState(m_k=np.array([1.0]), last_update=0.0)
        r_low = model.modulate_mu(mu, imp_low)
        r_high = model.modulate_mu(mu, imp_high)
        assert r_high[0] > r_low[0]

    def test_does_not_modify_input(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=2.0)
        mu = np.array([1.0, 2.0])
        mu_copy = mu.copy()
        imp = ImprintState(m_k=np.array([0.5, 0.5]), last_update=0.0)
        model.modulate_mu(mu, imp)
        np.testing.assert_allclose(mu, mu_copy)


class TestSLImprintPipelineWiring:
    """Pipeline: imprint → modulate_mu → SL engine."""

    def test_modulated_mu_drives_sl_engine(self):
        """modulate_mu → boosted mu → StuartLandauEngine → amplitudes.
        Proves imprint-modulated bifurcation feeds SL dynamics."""
        from scpn_phase_orchestrator.upde.stuart_landau import (
            StuartLandauEngine,
        )

        n = 4
        model = ImprintModel(decay_rate=0.0, saturation=5.0)
        mu_base = np.array([0.3, 0.3, 0.3, 0.3])
        imprint = ImprintState(m_k=np.array([0.0, 1.0, 2.0, 3.0]), last_update=0.0)
        mu_modulated = model.modulate_mu(mu_base, imprint)

        eng = StuartLandauEngine(n, dt=0.01)
        state = np.concatenate([np.zeros(n), np.full(n, 0.1)])
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        for _ in range(300):
            state = eng.step(
                state,
                omegas,
                mu_modulated,
                knm,
                knm,
                0.0,
                0.0,
                np.zeros((n, n)),
            )
        # Higher modulated mu → larger amplitude
        amps = state[n:]
        assert amps[3] > amps[0], f"Higher imprint should give larger amp: {amps}"
