# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SCPN Phase Orchestrator — Universal coupling prior tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.prior import CouplingPrior, UniversalPrior


class TestUniversalPrior:
    def test_default_values(self):
        prior = UniversalPrior()
        d = prior.default()
        assert abs(d.K_base - 0.47) < 1e-10
        assert abs(d.decay_alpha - 0.25) < 1e-10

    def test_sample_positive(self):
        prior = UniversalPrior()
        rng = np.random.default_rng(42)
        for _ in range(20):
            s = prior.sample(rng)
            assert s.K_base > 0
            assert s.decay_alpha > 0

    def test_sample_distribution(self):
        prior = UniversalPrior()
        rng = np.random.default_rng(42)
        samples = [prior.sample(rng) for _ in range(1000)]
        K_vals = [s.K_base for s in samples]
        assert abs(np.mean(K_vals) - 0.47) < 0.02
        assert abs(np.std(K_vals) - 0.09) < 0.02

    def test_estimate_Kc(self):
        prior = UniversalPrior()
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        result = prior.estimate_Kc(omegas, 4)
        assert isinstance(result, CouplingPrior)
        assert result.K_c_estimate > 0
        assert result.K_base == 0.47

    def test_estimate_Kc_identical_omegas(self):
        prior = UniversalPrior()
        omegas = np.ones(4)
        result = prior.estimate_Kc(omegas, 4)
        assert result.K_c_estimate == 0.0

    def test_log_probability_peak_at_mean(self):
        prior = UniversalPrior()
        lp_peak = prior.log_probability(0.47, 0.25)
        lp_off = prior.log_probability(0.8, 0.5)
        assert lp_peak > lp_off

    def test_log_probability_symmetric(self):
        prior = UniversalPrior()
        lp1 = prior.log_probability(0.47 + 0.1, 0.25)
        lp2 = prior.log_probability(0.47 - 0.1, 0.25)
        assert abs(lp1 - lp2) < 1e-10

    def test_custom_prior(self):
        prior = UniversalPrior(K_base_mean=1.0, K_base_std=0.2)
        d = prior.default()
        assert d.K_base == 1.0

    def test_sample_no_rng(self):
        prior = UniversalPrior()
        s = prior.sample()
        assert s.K_base > 0


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
