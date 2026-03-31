# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for synapse coupling bridge

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.adapters.synapse_coupling_bridge import (
    SynapseCouplingBridge,
)


class TestSynapseCouplingBridge:
    def test_initial_snapshot_zeros(self) -> None:
        bridge = SynapseCouplingBridge(4)
        snap = bridge.snapshot()
        np.testing.assert_array_equal(snap.knm_delta, 0.0)
        np.testing.assert_array_equal(snap.gap_coupling, 0.0)
        assert snap.mean_weight_change == 0.0

    def test_stdp_delta(self) -> None:
        bridge = SynapseCouplingBridge(3)
        w1 = np.array([[0, 0.5, 0.3], [0.5, 0, 0.2], [0.3, 0.2, 0]])
        bridge.update_stdp_weights(w1)
        w2 = np.array([[0, 0.7, 0.3], [0.4, 0, 0.2], [0.3, 0.2, 0]])
        bridge.update_stdp_weights(w2)
        snap = bridge.snapshot()
        assert snap.knm_delta[0, 1] > 0  # potentiation
        assert snap.knm_delta[1, 0] < 0  # depression

    def test_gap_symmetric(self) -> None:
        bridge = SynapseCouplingBridge(3)
        g = np.array([[0, 1.0, 0], [0.5, 0, 0], [0, 0, 0]])
        bridge.update_gap_conductances(g)
        snap = bridge.snapshot()
        np.testing.assert_allclose(snap.gap_coupling, snap.gap_coupling.T)

    def test_gap_zero_diagonal(self) -> None:
        bridge = SynapseCouplingBridge(3)
        g = np.ones((3, 3))
        bridge.update_gap_conductances(g)
        snap = bridge.snapshot()
        np.testing.assert_array_equal(np.diag(snap.gap_coupling), 0.0)

    def test_astrocyte_modulation_bounded(self) -> None:
        bridge = SynapseCouplingBridge(4)
        bridge.update_astrocyte_ca(np.array([0.0, 0.5, 1.0, 2.0]))
        snap = bridge.snapshot()
        assert np.all(snap.astrocyte_modulation >= 0)
        assert np.all(snap.astrocyte_modulation <= 1.0 + 1e-10)

    def test_apply_to_knm_nonneg(self) -> None:
        bridge = SynapseCouplingBridge(3)
        knm_base = np.ones((3, 3)) * 0.5
        np.fill_diagonal(knm_base, 0.0)
        result = bridge.apply_to_knm(knm_base)
        assert np.all(result >= 0)
        np.testing.assert_array_equal(np.diag(result), 0.0)

    def test_apply_to_imprint(self) -> None:
        bridge = SynapseCouplingBridge(3)
        bridge.update_astrocyte_ca(np.array([0.0, 0.5, 1.0]))
        m_k = np.ones(3)
        result = bridge.apply_to_imprint(m_k)
        assert result[2] > result[0]  # higher Ca²⁺ → stronger imprint

    def test_scale_parameters(self) -> None:
        bridge = SynapseCouplingBridge(3, stdp_scale=2.0, gap_scale=0.5)
        w = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        bridge.update_stdp_weights(np.zeros((3, 3)))
        bridge.update_stdp_weights(w)
        snap = bridge.snapshot()
        assert abs(snap.knm_delta[0, 1] - 2.0) < 1e-10  # 1.0 * scale=2


class TestSynapseBridgePipelineWiring:
    """Pipeline: STDP weights → apply_to_knm → engine."""

    def test_synapse_modified_knm_drives_engine(self):
        """SynapseCouplingBridge.apply_to_knm → modified K_nm → engine → R."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        bridge = SynapseCouplingBridge(n)
        knm_base = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm_base, 0.0)
        knm_modified = bridge.apply_to_knm(knm_base)

        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm_modified, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
