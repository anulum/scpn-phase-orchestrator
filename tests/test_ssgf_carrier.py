# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF carrier tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier, SSGFState


class TestGeometryCarrier:
    def test_decode_shape(self):
        gc = GeometryCarrier(6, z_dim=4, seed=42)
        W = gc.decode()
        assert W.shape == (6, 6)

    def test_decode_non_negative(self):
        gc = GeometryCarrier(5, z_dim=4, seed=42)
        W = gc.decode()
        assert np.all(W >= 0)

    def test_decode_zero_diagonal(self):
        gc = GeometryCarrier(4, z_dim=3, seed=42)
        W = gc.decode()
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_different_z_different_W(self):
        gc = GeometryCarrier(4, z_dim=4, seed=42)
        W1 = gc.decode(np.zeros(4))
        W2 = gc.decode(np.ones(4))
        assert not np.allclose(W1, W2)

    def test_update_returns_state(self):
        gc = GeometryCarrier(4, z_dim=3, seed=42)
        state = gc.update(cost=0.5)
        assert isinstance(state, SSGFState)
        assert state.step == 1
        assert state.cost == 0.5
        assert state.W.shape == (4, 4)

    def test_update_with_cost_fn_moves_z(self):
        gc = GeometryCarrier(4, z_dim=3, lr=0.1, seed=42)
        z0 = gc.z.copy()
        # Cost = sum of coupling weights — gradient should reduce coupling
        gc.update(cost=0.0, cost_fn=lambda W: float(np.sum(W)))
        z1 = gc.z
        assert not np.allclose(z0, z1)

    def test_update_descends_cost(self):
        gc = GeometryCarrier(4, z_dim=3, lr=0.05, seed=42)
        def cost_fn(W):
            return float(np.sum((W - 0.5) ** 2))
        costs = []
        for _ in range(20):
            W = gc.decode()
            c = cost_fn(W)
            costs.append(c)
            gc.update(cost=c, cost_fn=cost_fn)
        # Cost should generally decrease
        assert costs[-1] < costs[0]

    def test_z_property(self):
        gc = GeometryCarrier(3, z_dim=2, seed=42)
        z = gc.z
        assert z.shape == (2,)
        z[0] = 999.0
        assert gc.z[0] != 999.0  # returns copy

    def test_reset(self):
        gc = GeometryCarrier(3, z_dim=2, seed=42)
        gc.update(cost=0.5)
        gc.update(cost=0.3)
        gc.reset(seed=42)
        z = gc.z
        gc2 = GeometryCarrier(3, z_dim=2, seed=42)
        np.testing.assert_array_equal(z, gc2.z)

    def test_z_dim_property(self):
        gc = GeometryCarrier(4, z_dim=7)
        assert gc.z_dim == 7
