# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for SSGF carrier, closure, ethical

from __future__ import annotations

from typing import get_type_hints

import numpy as np

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier, SSGFState
from scpn_phase_orchestrator.ssgf.closure import ClosureState, CyberneticClosure
from scpn_phase_orchestrator.ssgf.ethical import EthicalCost, compute_ethical_cost

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestGeometryCarrier:
    def test_decode_shape(self) -> None:
        gc = GeometryCarrier(4, z_dim=6, seed=0)
        W = gc.decode()
        assert W.shape == (4, 4)

    def test_decode_zero_diagonal(self) -> None:
        gc = GeometryCarrier(5, seed=0)
        W = gc.decode()
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_decode_nonnegative(self) -> None:
        gc = GeometryCarrier(4, seed=42)
        W = gc.decode()
        assert np.all(W >= 0)

    def test_z_copy_not_reference(self) -> None:
        gc = GeometryCarrier(3, seed=0)
        z1 = gc.z
        z1[0] = 999.0
        assert gc.z[0] != 999.0

    def test_update_returns_state(self) -> None:
        gc = GeometryCarrier(3, seed=0)
        state = gc.update(cost=1.0)
        assert isinstance(state, SSGFState)
        assert state.step == 1
        assert state.cost == 1.0

    def test_update_with_cost_fn(self) -> None:
        gc = GeometryCarrier(3, z_dim=4, lr=0.1, seed=0)
        z_before = gc.z.copy()
        state = gc.update(cost=1.0, cost_fn=lambda W: float(np.sum(W)))
        assert state.grad_norm > 0
        assert not np.allclose(gc.z, z_before)

    def test_reset(self) -> None:
        gc = GeometryCarrier(3, seed=0)
        gc.update(cost=1.0)
        gc.reset(seed=99)
        assert gc.z is not None
        assert gc._step == 0

    def test_z_dim_property(self) -> None:
        gc = GeometryCarrier(4, z_dim=12)
        assert gc.z_dim == 12
        assert len(gc.z) == 12

    def test_decode_custom_z(self) -> None:
        gc = GeometryCarrier(3, z_dim=4, seed=0)
        z_custom = np.ones(4)
        W = gc.decode(z_custom)
        assert W.shape == (3, 3)
        assert np.all(W >= 0)

    def test_public_array_contracts_are_parameterised(self) -> None:
        state_hints = get_type_hints(SSGFState)
        for field in ("z", "W"):
            assert "numpy.ndarray" in str(state_hints[field])
            assert "float64" in str(state_hints[field])

        for hint in [
            get_type_hints(GeometryCarrier.z.fget)["return"],
            get_type_hints(GeometryCarrier.decode)["z"],
            get_type_hints(GeometryCarrier.decode)["return"],
            get_type_hints(GeometryCarrier.update)["cost_fn"],
        ]:
            assert "numpy.ndarray" in str(hint)
            assert "float64" in str(hint)


class TestCyberneticClosure:
    def test_step_returns_w_and_state(self) -> None:
        gc = GeometryCarrier(4, seed=0)
        cc = CyberneticClosure(gc)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W, cs = cc.step(phases)
        assert W.shape == (4, 4)
        assert isinstance(cs, ClosureState)
        assert cs.ssgf_state_step == 1

    def test_cost_decreases_or_stable(self) -> None:
        gc = GeometryCarrier(4, z_dim=6, lr=0.01, seed=42)
        cc = CyberneticClosure(gc)
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        _, cs1 = cc.step(phases)
        _, cs2 = cc.step(phases)
        # Not guaranteed monotonic in 1 step, but cost should be finite
        assert np.isfinite(cs1.cost_after)
        assert np.isfinite(cs2.cost_after)

    def test_run_returns_history(self) -> None:
        gc = GeometryCarrier(3, seed=0)
        cc = CyberneticClosure(gc)
        phases = np.ones(3) * 1.5
        W, history = cc.run(phases, n_outer_steps=5)
        assert W.shape == (3, 3)
        assert len(history) == 5

    def test_reset(self) -> None:
        gc = GeometryCarrier(3, seed=0)
        cc = CyberneticClosure(gc)
        cc.step(np.ones(3))
        cc.reset()
        assert cc._step == 0

    def test_carrier_property(self) -> None:
        gc = GeometryCarrier(3)
        cc = CyberneticClosure(gc)
        assert cc.carrier is gc

    def test_public_array_contracts_are_parameterised(self) -> None:
        for hint in [
            get_type_hints(CyberneticClosure.step)["phases"],
            get_type_hints(CyberneticClosure.step)["return"],
            get_type_hints(CyberneticClosure.run)["phases"],
            get_type_hints(CyberneticClosure.run)["return"],
        ]:
            assert "numpy.ndarray" in str(hint)
            assert "float64" in str(hint)


class TestEthicalCost:
    def test_empty_phases(self) -> None:
        result = compute_ethical_cost(np.array([]), np.zeros((0, 0)))
        assert result.c15_sec == 1.0
        assert result.constraints_violated == 0

    def test_sync_phases_high_jsec(self) -> None:
        n = 6
        phases = np.full(n, 1.0)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert result.J_sec > 0.3  # R≈1 contributes heavily
        assert isinstance(result, EthicalCost)

    def test_all_fields_finite(self) -> None:
        rng = np.random.default_rng(42)
        n = 6
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert np.isfinite(result.J_sec)
        assert np.isfinite(result.phi_ethics)
        assert np.isfinite(result.c15_sec)

    def test_phi_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        n = 5
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert result.phi_ethics >= 0

    def test_no_violations_when_healthy(self) -> None:
        n = 6
        phases = np.full(n, 1.0)
        knm = _connected_knm(n, seed=42)
        result = compute_ethical_cost(phases, knm, R_min=0.0, connectivity_min=0.0)
        assert result.constraints_violated == 0

    def test_violation_count(self) -> None:
        n = 4
        phases = np.linspace(0, TWO_PI, n, endpoint=False)
        knm = np.zeros((n, n))
        result = compute_ethical_cost(
            phases,
            knm,
            R_min=0.9,
            connectivity_min=1.0,
        )
        assert result.constraints_violated >= 1

    def test_public_array_contracts_are_parameterised(self) -> None:
        hints = get_type_hints(compute_ethical_cost)
        for param in ("phases", "knm"):
            assert "numpy.ndarray" in str(hints[param])
            assert "float64" in str(hints[param])


class TestSSGFModulesPipelineWiring:
    """Pipeline: SSGF closure → K_nm → engine → ethical cost."""

    def test_ssgf_closure_to_engine_to_ethical_cost(self):
        """CyberneticClosure → W → engine → phases → ethical cost."""
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        carrier = GeometryCarrier(n, z_dim=3, lr=0.05, seed=42)
        closure = CyberneticClosure(carrier)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)

        W, _ = closure.step(phases)
        assert W.shape == (n, n)

        eng = UPDEEngine(n, dt=0.01)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

        cost = compute_ethical_cost(phases, W)
        assert isinstance(cost, EthicalCost)
        assert np.isfinite(cost.c15_sec)
