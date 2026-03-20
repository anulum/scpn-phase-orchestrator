# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen reduction tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction


class TestOttAntonsen:
    def test_critical_coupling(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=1.0)
        assert oa.K_c == 1.0

    def test_below_kc_zero_R(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=1.0)
        assert oa.steady_state_R() == 0.0

    def test_above_kc_positive_R(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0)
        R_ss = oa.steady_state_R()
        # R_ss = √(1 - 2*0.5/4) = √(1 - 0.25) = √0.75 ≈ 0.866
        assert abs(R_ss - np.sqrt(0.75)) < 1e-10

    def test_at_kc_zero_R(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=1.0)
        assert oa.steady_state_R() == 0.0

    def test_step_preserves_finite(self):
        oa = OttAntonsenReduction(omega_0=1.0, delta=0.1, K=2.0, dt=0.01)
        z = complex(0.5, 0.0)
        z_new = oa.step(z)
        assert np.isfinite(z_new.real)
        assert np.isfinite(z_new.imag)

    def test_run_converges_above_kc(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        state = oa.run(complex(0.01, 0.0), n_steps=5000)
        expected_R = oa.steady_state_R()
        assert abs(state.R - expected_R) < 0.05

    def test_run_stays_zero_below_kc(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=0.5, dt=0.01)
        state = oa.run(complex(0.01, 0.0), n_steps=2000)
        assert state.R < 0.05

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            OttAntonsenReduction(omega_0=0.0, delta=-0.5, K=1.0)

    def test_predict_from_oscillators(self):
        rng = np.random.default_rng(42)
        omegas = rng.standard_cauchy(100) * 0.1 + 1.0
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=2.0, dt=0.01)
        state = oa.predict_from_oscillators(omegas, K=2.0)
        assert isinstance(state, OAState)
        assert 0 <= state.R <= 1.0

    def test_predict_identical_omegas(self):
        omegas = np.ones(10)
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=2.0, dt=0.01)
        state = oa.predict_from_oscillators(omegas, K=2.0)
        assert state.R > 0.5

    def test_oa_state_fields(self):
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        state = oa.run(complex(0.5, 0.0), n_steps=100)
        assert hasattr(state, "z")
        assert hasattr(state, "R")
        assert hasattr(state, "psi")
        assert hasattr(state, "K_c")
        assert state.K_c == 0.2
