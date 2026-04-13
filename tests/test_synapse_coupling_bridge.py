# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Synapse coupling bridge tests

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from scpn_phase_orchestrator.adapters.synapse_coupling_bridge import (
    SynapseCouplingBridge,
    SynapseSnapshot,
)


class TestInitialState:
    def test_initial_snapshot_zero(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=4)
        snap = bridge.snapshot()
        assert_allclose(snap.knm_delta, 0.0)
        assert_allclose(snap.gap_coupling, 0.0)
        assert_allclose(snap.astrocyte_modulation, 0.0)
        assert snap.mean_weight_change == 0.0
        assert snap.mean_conductance == 0.0
        assert snap.mean_ca == 0.0

    def test_snapshot_type(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        assert isinstance(bridge.snapshot(), SynapseSnapshot)


class TestSTDP:
    def test_weight_increase_positive_delta(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        w0 = np.zeros((3, 3))
        w1 = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]])
        bridge.update_stdp_weights(w0)
        bridge.update_stdp_weights(w1)
        snap = bridge.snapshot()
        assert snap.knm_delta[0, 1] == 0.5
        assert snap.knm_delta[1, 2] == 0.3
        assert snap.knm_delta[0, 0] == 0.0  # diagonal always zero

    def test_weight_decrease_negative_delta(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2)
        bridge.update_stdp_weights(np.array([[0, 1.0], [1.0, 0]]))
        bridge.update_stdp_weights(np.array([[0, 0.3], [0.3, 0]]))
        snap = bridge.snapshot()
        assert_allclose(snap.knm_delta[0, 1], -0.7)

    def test_stdp_scale(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2, stdp_scale=2.0)
        bridge.update_stdp_weights(np.zeros((2, 2)))
        bridge.update_stdp_weights(np.array([[0, 0.5], [0.5, 0]]))
        snap = bridge.snapshot()
        assert_allclose(snap.knm_delta[0, 1], 1.0)

    def test_mean_weight_change(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2)
        bridge.update_stdp_weights(np.zeros((2, 2)))
        bridge.update_stdp_weights(np.array([[0, 0.4], [0.6, 0]]))
        snap = bridge.snapshot()
        assert snap.mean_weight_change > 0.0


class TestGapJunction:
    def test_symmetric(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        g = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=float)
        bridge.update_gap_conductances(g)
        snap = bridge.snapshot()
        assert_allclose(snap.gap_coupling, snap.gap_coupling.T)

    def test_diagonal_zero(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        g = np.ones((3, 3))
        bridge.update_gap_conductances(g)
        snap = bridge.snapshot()
        assert_allclose(np.diag(snap.gap_coupling), 0.0)

    def test_gap_scale(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2, gap_scale=3.0)
        bridge.update_gap_conductances(np.array([[0, 1], [1, 0]], dtype=float))
        snap = bridge.snapshot()
        assert_allclose(snap.gap_coupling[0, 1], 3.0)

    def test_mean_conductance(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2)
        bridge.update_gap_conductances(np.array([[0, 0.8], [0.8, 0]], dtype=float))
        snap = bridge.snapshot()
        assert snap.mean_conductance > 0.0


class TestAstrocyte:
    def test_normalisation(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        bridge.update_astrocyte_ca(np.array([0.0, 0.5, 1.0]))
        snap = bridge.snapshot()
        assert_allclose(snap.astrocyte_modulation[-1], 1.0)
        assert_allclose(snap.astrocyte_modulation[0], 0.0)

    def test_ca_scale(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2, ca_scale=2.0)
        bridge.update_astrocyte_ca(np.array([0.5, 1.0]))
        snap = bridge.snapshot()
        assert_allclose(snap.astrocyte_modulation[-1], 2.0)

    def test_mean_ca(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2)
        bridge.update_astrocyte_ca(np.array([0.4, 0.6]))
        snap = bridge.snapshot()
        assert_allclose(snap.mean_ca, 0.5)

    def test_zero_ca_no_nan(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        bridge.update_astrocyte_ca(np.zeros(3))
        snap = bridge.snapshot()
        assert np.all(np.isfinite(snap.astrocyte_modulation))


class TestApply:
    def test_apply_to_knm_non_negative(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        bridge.update_stdp_weights(np.zeros((3, 3)))
        w = np.array([[0, -5, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        bridge.update_stdp_weights(w)
        knm_base = np.ones((3, 3))
        result = bridge.apply_to_knm(knm_base)
        assert np.all(result >= 0.0)

    def test_apply_to_knm_diagonal_zero(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        bridge.update_gap_conductances(np.ones((3, 3)))
        knm = bridge.apply_to_knm(np.ones((3, 3)))
        assert_allclose(np.diag(knm), 0.0)

    def test_apply_to_imprint(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=3)
        bridge.update_astrocyte_ca(np.array([0.0, 0.5, 1.0]))
        m_k = np.ones(3)
        result = bridge.apply_to_imprint(m_k)
        assert result[-1] > result[0]

    def test_apply_to_imprint_preserves_zero(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=2)
        bridge.update_astrocyte_ca(np.array([1.0, 1.0]))
        m_k = np.array([0.0, 0.5])
        result = bridge.apply_to_imprint(m_k)
        assert result[0] == 0.0


class TestEdgeCases:
    def test_single_oscillator(self) -> None:
        bridge = SynapseCouplingBridge(n_oscillators=1)
        bridge.update_stdp_weights(np.array([[0.5]]))
        bridge.update_gap_conductances(np.array([[0.0]]))
        bridge.update_astrocyte_ca(np.array([1.0]))
        snap = bridge.snapshot()
        assert snap.knm_delta.shape == (1, 1)

    def test_large_n(self) -> None:
        n = 100
        bridge = SynapseCouplingBridge(n_oscillators=n)
        rng = np.random.default_rng(42)
        bridge.update_stdp_weights(rng.uniform(0, 1, (n, n)))
        bridge.update_stdp_weights(rng.uniform(0, 1, (n, n)))
        bridge.update_gap_conductances(rng.uniform(0, 0.5, (n, n)))
        bridge.update_astrocyte_ca(rng.uniform(0, 1, n))
        snap = bridge.snapshot()
        assert snap.knm_delta.shape == (n, n)
        assert np.all(np.isfinite(snap.knm_delta))
        knm = bridge.apply_to_knm(rng.uniform(0, 1, (n, n)))
        assert np.all(knm >= 0.0)
