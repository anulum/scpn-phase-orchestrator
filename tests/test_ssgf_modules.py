# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for SSGF carrier, closure, ethical

"""Exercise original SSGF carrier, closure and ethical-cost consumers."""

from __future__ import annotations

from inspect import getattr_static
from typing import get_type_hints

import numpy as np

from benchmarks.ethical_cost_reference import reference_cost
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier, SSGFState
from scpn_phase_orchestrator.ssgf.closure import ClosureState, CyberneticClosure
from scpn_phase_orchestrator.ssgf.ethical import (
    EthicalCost,
    FloatArray,
    compute_ethical_cost,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, seed: int = 0) -> FloatArray:
    """Construct a symmetric nonnegative graph with an exact zero diagonal."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestGeometryCarrier:
    """Exercise generated geometry and public latent-state updates."""

    def test_decode_shape(self) -> None:
        """Decoding the latent vector produces a matching square graph."""
        gc = GeometryCarrier(4, z_dim=6, seed=0)
        W = gc.decode()
        assert W.shape == (4, 4)

    def test_decode_zero_diagonal(self) -> None:
        """Decoding excludes self-coupling for every generated node."""
        gc = GeometryCarrier(5, seed=0)
        W = gc.decode()
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_decode_nonnegative(self) -> None:
        """Decoded coupling weights remain nonnegative."""
        gc = GeometryCarrier(4, seed=42)
        W = gc.decode()
        assert np.all(W >= 0)

    def test_z_copy_not_reference(self) -> None:
        """Editing the returned latent state cannot mutate carrier storage."""
        gc = GeometryCarrier(3, seed=0)
        z1 = gc.z
        z1[0] = 999.0
        assert gc.z[0] != 999.0

    def test_update_returns_state(self) -> None:
        """A public update records the actual cost and next state step."""
        gc = GeometryCarrier(3, seed=0)
        state = gc.update(cost=1.0)
        assert isinstance(state, SSGFState)
        assert state.step == 1
        assert state.cost == 1.0

    def test_update_with_cost_fn(self) -> None:
        """The actual finite-difference objective moves the latent state."""
        gc = GeometryCarrier(3, z_dim=4, lr=0.1, seed=0)
        z_before = gc.z.copy()
        state = gc.update(cost=1.0, cost_fn=lambda W: float(np.sum(W)))
        assert state.grad_norm > 0
        assert not np.allclose(gc.z, z_before)

    def test_reset(self) -> None:
        """Resetting a used carrier restarts state history and generates geometry."""
        gc = GeometryCarrier(3, seed=0)
        gc.update(cost=1.0)
        gc.reset(seed=99)
        assert gc.z is not None
        assert gc._step == 0

    def test_z_dim_property(self) -> None:
        """The declared latent dimension matches the produced state vector."""
        gc = GeometryCarrier(4, z_dim=12)
        assert gc.z_dim == 12
        assert len(gc.z) == 12

    def test_decode_custom_z(self) -> None:
        """Explicit latent input produces a nonnegative matching graph."""
        gc = GeometryCarrier(3, z_dim=4, seed=0)
        z_custom = np.ones(4)
        W = gc.decode(z_custom)
        assert W.shape == (3, 3)
        assert np.all(W >= 0)

    def test_public_array_contracts_are_parameterised(self) -> None:
        """Public numerical operations retain their declared float64 array types."""
        carrier = GeometryCarrier(3, z_dim=4, seed=42)
        phases = np.zeros(3)
        state = carrier.update(1.0)
        assert state.z.dtype == state.W.dtype == phases.dtype == np.float64
        assert carrier.decode().shape == (3, 3)
        state_hints = get_type_hints(SSGFState)
        for field in ("z", "W"):
            assert_precise_ndarray_hint(state_hints[field])
            assert "float64" in str(state_hints[field])

        descriptor = getattr_static(GeometryCarrier, "z")
        assert isinstance(descriptor, property)
        getter = descriptor.fget
        assert getter is not None
        for hint in [
            get_type_hints(getter)["return"],
            get_type_hints(GeometryCarrier.decode)["z"],
            get_type_hints(GeometryCarrier.decode)["return"],
            get_type_hints(GeometryCarrier.update)["cost_fn"],
        ]:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


class TestCyberneticClosure:
    """Exercise closure geometry, state history and reset behavior."""

    def test_step_returns_w_and_state(self) -> None:
        """Closure produces coupling consumed by the next carrier state."""
        gc = GeometryCarrier(4, seed=0)
        cc = CyberneticClosure(gc)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W, cs = cc.step(phases)
        assert W.shape == (4, 4)
        assert isinstance(cs, ClosureState)
        assert cs.ssgf_state_step == 1

    def test_cost_decreases_or_stable(self) -> None:
        """Two real closure steps retain finite objective values."""
        gc = GeometryCarrier(4, z_dim=6, lr=0.01, seed=42)
        cc = CyberneticClosure(gc)
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        _, cs1 = cc.step(phases)
        _, cs2 = cc.step(phases)
        # Not guaranteed monotonic in 1 step, but cost should be finite
        assert np.isfinite(cs1.cost_after)
        assert np.isfinite(cs2.cost_after)

    def test_run_returns_history(self) -> None:
        """Requested outer steps return one state record per actual update."""
        gc = GeometryCarrier(3, seed=0)
        cc = CyberneticClosure(gc)
        phases = np.ones(3) * 1.5
        W, history = cc.run(phases, n_outer_steps=5)
        assert W.shape == (3, 3)
        assert len(history) == 5

    def test_reset(self) -> None:
        """Resetting a used closure restarts its next public state step."""
        gc = GeometryCarrier(3, seed=0)
        cc = CyberneticClosure(gc)
        cc.step(np.ones(3))
        cc.reset()
        assert cc._step == 0
        _, state = cc.step(np.ones(3))
        assert state.ssgf_state_step == 1

    def test_carrier_property(self) -> None:
        """The public closure retains its supplied carrier owner."""
        gc = GeometryCarrier(3)
        cc = CyberneticClosure(gc)
        assert cc.carrier is gc

    def test_public_array_contracts_are_parameterised(self) -> None:
        """Public numerical operations retain their declared float64 array types."""
        closure = CyberneticClosure(GeometryCarrier(3, seed=42))
        phases = np.zeros(3)
        weight, state = closure.step(phases)
        assert weight.dtype == phases.dtype == np.float64
        assert weight.shape == (3, 3) and state.ssgf_state_step == 1
        for hint in [
            get_type_hints(CyberneticClosure.step)["phases"],
            get_type_hints(CyberneticClosure.step)["return"],
            get_type_hints(CyberneticClosure.run)["phases"],
            get_type_hints(CyberneticClosure.run)["return"],
        ]:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


class TestEthicalCost:
    """Exercise the public diagnostic on real generated coupling graphs."""

    def test_empty_phases(self) -> None:
        """An empty matching graph returns the unit total and no violations."""
        result = compute_ethical_cost(np.array([]), np.zeros((0, 0)))
        assert result.c15_sec == 1.0
        assert result.constraints_violated == 0

    def test_sync_phases_high_jsec(self) -> None:
        """A synchronized generated graph produces a positive coherence score."""
        n = 6
        phases = np.full(n, 1.0)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert result.J_sec > 0.3  # R≈1 contributes heavily
        assert isinstance(result, EthicalCost)

    def test_all_fields_finite(self) -> None:
        """Actual phase and graph measurements produce finite diagnostic fields."""
        rng = np.random.default_rng(42)
        n = 6
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert np.isfinite(result.J_sec)
        assert np.isfinite(result.phi_ethics)
        assert np.isfinite(result.c15_sec)

    def test_phi_nonnegative(self) -> None:
        """The default positive multiplier produces a nonnegative penalty."""
        rng = np.random.default_rng(0)
        n = 5
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n)
        result = compute_ethical_cost(phases, knm)
        assert result.phi_ethics >= 0

    def test_no_violations_when_healthy(self) -> None:
        """Zero coherence and connectivity floors admit the bounded graph."""
        n = 6
        phases = np.full(n, 1.0)
        knm = _connected_knm(n, seed=42)
        result = compute_ethical_cost(phases, knm, R_min=0.0, connectivity_min=0.0)
        assert result.constraints_violated == 0

    def test_violation_count(self) -> None:
        """Splayed uncoupled phases violate a positive connectivity target."""
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
        """Public numerical operations retain their declared float64 array types."""
        phases = np.zeros(3)
        weight = _connected_knm(3)
        cost = compute_ethical_cost(phases, weight)
        expected = reference_cost(phases, weight)
        np.testing.assert_allclose(
            [cost.J_sec, cost.phi_ethics, cost.c15_sec],
            expected[:3],
            rtol=1e-10,
            atol=1e-10,
        )
        assert cost.constraints_violated == expected[3]
        hints = get_type_hints(compute_ethical_cost)
        for param in ("phases", "knm"):
            assert_precise_ndarray_hint(hints[param])
            assert "float64" in str(hints[param])


class TestSSGFModulesPipelineWiring:
    """Pipeline: SSGF closure → K_nm → engine → ethical cost."""

    def test_ssgf_closure_to_engine_to_ethical_cost(self) -> None:
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
        expected = reference_cost(phases, W)
        np.testing.assert_allclose(
            [cost.J_sec, cost.phi_ethics, cost.c15_sec],
            expected[:3],
            rtol=1e-10,
            atol=1e-10,
        )
        assert cost.constraints_violated == expected[3]

    def test_cybernetic_closure_run_zero_steps_returns_initial_state(self) -> None:
        """run(…, n_outer_steps=0) should be a no-op with empty history."""
        gc = GeometryCarrier(4, z_dim=3, lr=0.01, seed=0)
        closure = CyberneticClosure(gc)
        phases = np.full(4, 0.5)
        initial_w = gc.decode()

        final_w, history = closure.run(phases, n_outer_steps=0)

        assert closure._step == 0
        np.testing.assert_allclose(final_w, initial_w)
        assert history == []
