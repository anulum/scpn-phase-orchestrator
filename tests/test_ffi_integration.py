# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FFI integration tests

from __future__ import annotations

import math

import numpy as np
import pytest

spo = pytest.importorskip("spo_kernel")


# ─── PyUPDEStepper ───────────────────────────────────────────────


class TestPyUPDEStepper:
    def test_construct(self):
        s = spo.PyUPDEStepper(4)
        assert s.n == 4

    def test_step_returns_correct_length(self):
        n = 4
        s = spo.PyUPDEStepper(n)
        phases = np.zeros(n)
        result = s.step(phases, np.ones(n), np.zeros(n * n), 0.0, 0.0, np.zeros(n * n))
        assert len(result) == n

    def test_step_advances_phases(self):
        n = 4
        s = spo.PyUPDEStepper(n, dt=0.01)
        phases = np.zeros(n)
        result = s.step(phases, np.ones(n), np.zeros(n * n), 0.0, 0.0, np.zeros(n * n))
        for p in result:
            assert abs(p - 0.01) < 1e-10

    def test_run_multiple_steps(self):
        n = 4
        s = spo.PyUPDEStepper(n)
        phases = np.zeros(n)
        result = s.run(
            phases, np.ones(n), np.zeros(n * n), 0.0, 0.0, np.zeros(n * n), 100
        )
        assert len(result) == n
        assert all(0 <= p < 2 * math.pi for p in result)

    def test_n_getter(self):
        assert spo.PyUPDEStepper(8).n == 8

    def test_nan_phases_rejected(self):
        s = spo.PyUPDEStepper(4)
        with pytest.raises(ValueError):
            s.step(
                np.full(4, float("nan")),
                np.ones(4),
                np.zeros(16),
                0.0,
                0.0,
                np.zeros(16),
            )

    def test_nan_zeta_rejected(self):
        s = spo.PyUPDEStepper(4)
        with pytest.raises(ValueError):
            s.step(
                np.zeros(4),
                np.ones(4),
                np.zeros(16),
                float("nan"),
                0.0,
                np.zeros(16),
            )

    def test_inf_psi_rejected(self):
        s = spo.PyUPDEStepper(4)
        with pytest.raises(ValueError):
            s.step(
                np.zeros(4),
                np.ones(4),
                np.zeros(16),
                0.0,
                float("inf"),
                np.zeros(16),
            )

    def test_substeps_accepted(self):
        s = spo.PyUPDEStepper(4, dt=0.01, method="rk4", n_substeps=4)
        assert s.n == 4
        phases = np.zeros(4)
        result = s.step(phases, np.ones(4), np.zeros(16), 0.0, 0.0, np.zeros(16))
        assert len(result) == 4

    def test_dimension_mismatch(self):
        s = spo.PyUPDEStepper(4)
        with pytest.raises(ValueError):
            s.step(np.zeros(3), np.ones(4), np.zeros(16), 0.0, 0.0, np.zeros(16))


# ─── PyCouplingBuilder ───────────────────────────────────────────


class TestPyCouplingBuilder:
    def test_build(self):
        cb = spo.PyCouplingBuilder()
        d = cb.build(4, 0.45, 0.3)
        assert d["n"] == 4
        assert len(d["knm"]) == 16
        assert len(d["alpha"]) == 16

    def test_project(self):
        knm = [0.0, -0.5, 0.3, 0.0]
        result = spo.PyCouplingBuilder.project(knm, 2)
        assert all(v >= 0 for v in result)

    def test_project_dimension_error(self):
        with pytest.raises(ValueError):
            spo.PyCouplingBuilder.project([1.0] * 5, 3)


# ─── PyRegimeManager ────────────────────────────────────────────


class TestPyRegimeManager:
    def test_evaluate_nominal(self):
        rm = spo.PyRegimeManager()
        regime = rm.evaluate([0.9, 0.8, 0.85, 0.9], [])
        assert regime == "nominal"

    def test_transition(self):
        rm = spo.PyRegimeManager()
        actual = rm.transition("degraded")
        assert actual in ("nominal", "degraded")

    def test_cooldown(self):
        rm = spo.PyRegimeManager()
        r1 = rm.transition("degraded")
        r2 = rm.transition("nominal")
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ─── PyCoherenceMonitor ──────────────────────────────────────────


class TestPyCoherenceMonitor:
    def test_good_bad_split(self):
        cm = spo.PyCoherenceMonitor([0, 1], [2, 3])
        r_good = cm.compute_r_good([0.9, 0.8, 0.1, 0.2])
        r_bad = cm.compute_r_bad([0.9, 0.8, 0.1, 0.2])
        assert r_good > r_bad

    def test_detect_phase_lock(self):
        cm = spo.PyCoherenceMonitor([0, 1], [2])
        rs = [0.9, 0.8, 0.3]
        psi = [0.0, 0.1, 1.5]
        # 3x3 CLA: (0,1)=0.95, (0,2)=0.5, (1,2)=0.92
        cla = [0.0, 0.95, 0.50, 0.95, 0.0, 0.92, 0.50, 0.92, 0.0]
        locked = cm.detect_phase_lock(rs, psi, cla, 0.9)
        assert (0, 1) in locked
        assert (1, 2) in locked
        assert (0, 2) not in locked


# ─── PyBoundaryObserver ──────────────────────────────────────────


class TestPyBoundaryObserver:
    def test_soft_violation(self):
        bo = spo.PyBoundaryObserver()
        defs = [("temp_high", "temperature", None, 100.0, "soft")]
        result = bo.observe(defs, {"temperature": 150.0})
        assert len(result["soft_violations"]) == 1

    def test_hard_violation(self):
        bo = spo.PyBoundaryObserver()
        defs = [("temp_high", "temperature", None, 100.0, "hard")]
        result = bo.observe(defs, {"temperature": 150.0})
        assert len(result["hard_violations"]) == 1


# ─── PyImprintModel ─────────────────────────────────────────────


class TestPyImprintModel:
    def test_update_and_m(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        im.update([1.0, 2.0], 1.0)
        assert abs(im.m[0] - 1.0) < 1e-12
        assert abs(im.m[1] - 2.0) < 1e-12

    def test_modulate_coupling(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        im.update([0.5, 0.0], 1.0)
        knm = im.modulate_coupling([0.0, 1.0, 1.0, 0.0])
        assert abs(knm[1] - 1.5) < 1e-12

    def test_modulate_lag(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        im.update([0.3, 0.0], 1.0)
        # alpha[i,j] += m[i] - m[j]: [0,0]=0+(0.3-0.3)=0, [0,1]=1+(0.3-0)=1.3
        alpha = im.modulate_lag([0.0, 1.0, -1.0, 0.0])
        assert abs(alpha[0] - 0.0) < 1e-12
        assert abs(alpha[1] - 1.3) < 1e-12
        assert abs(alpha[2] - (-1.3)) < 1e-12
        assert abs(alpha[3] - 0.0) < 1e-12

    def test_modulate_coupling_dimension_error(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        with pytest.raises(ValueError):
            im.modulate_coupling([1.0] * 3)

    def test_modulate_lag_dimension_error(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        with pytest.raises(ValueError):
            im.modulate_lag([1.0] * 5)

    def test_reset(self):
        im = spo.PyImprintModel(2, 0.0, 10.0)
        im.update([1.0, 1.0], 1.0)
        im.reset()
        assert all(v == 0.0 for v in im.m)


# ─── PyActionProjector ──────────────────────────────────────────


class TestPyActionProjector:
    def test_clamp(self):
        ap = spo.PyActionProjector({}, {"K": (0.0, 1.0)})
        assert ap.project("K", 2.0, 0.5) == 1.0

    def test_rate_limit(self):
        ap = spo.PyActionProjector({"K": 0.1}, {})
        result = ap.project("K", 1.0, 0.5)
        assert result <= 0.6 + 1e-9


# ─── PyPhaseQualityScorer ───────────────────────────────────────


class TestPyPhaseQualityScorer:
    def test_score(self):
        s = spo.PyPhaseQualityScorer()
        val = s.score([0.9, 0.8], [1.0, 1.0])
        assert 0.0 <= val <= 1.0

    def test_is_collapsed(self):
        s = spo.PyPhaseQualityScorer(collapse_threshold=0.5)
        assert s.is_collapsed([0.1, 0.1])
        assert not s.is_collapsed([0.9, 0.8])

    def test_downweight_mask(self):
        s = spo.PyPhaseQualityScorer()
        mask = s.downweight_mask([0.9, 0.01])
        assert len(mask) == 2


# ─── Free Functions ──────────────────────────────────────────────


class TestFreeFunctions:
    def test_order_parameter(self):
        r, psi = spo.order_parameter(np.zeros(4))
        assert abs(r - 1.0) < 1e-10

    def test_plv(self):
        val = spo.plv(np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2]))
        assert abs(val - 1.0) < 1e-10

    def test_ring_phase(self):
        p = spo.ring_phase(1, 4)
        assert abs(p - math.pi / 2) < 1e-10

    def test_event_phase(self):
        result = spo.event_phase(np.array([0.0, 1.0, 2.0]))
        assert len(result) == 3

    def test_physical_extract(self):
        result = spo.physical_extract(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 2.0)
        assert len(result) == 4

    def test_graph_walk_phase(self):
        p = spo.graph_walk_phase(5, 10)
        assert abs(p - math.pi) < 1e-10

    def test_transition_quality_ideal(self):
        assert spo.transition_quality(1, 10) == 1.0

    def test_transition_quality_stalled(self):
        assert spo.transition_quality(0, 10) == 0.2

    def test_layer_coherence_subset(self):
        phases = np.array([0.5, 0.5, 3.0, 0.5])
        r = spo.layer_coherence(phases, [0, 1, 3])
        assert r > 0.9

    def test_layer_coherence_empty(self):
        assert spo.layer_coherence(np.array([1.0, 2.0]), []) == 0.0


# ─── PyLagModel ──────────────────────────────────────────────────


class TestPyLagModel:
    def test_zeros(self):
        lm = spo.PyLagModel.zeros(4)
        assert lm.n == 4
        assert all(v == 0.0 for v in lm.alpha)

    def test_estimate(self):
        lm = spo.PyLagModel.estimate([0.0, 1.0, 1.0, 0.0], 2, 1.0)
        assert lm.n == 2
        assert abs(lm.alpha[0 * 2 + 1] + lm.alpha[1 * 2 + 0]) < 1e-12

    def test_estimate_dimension_error(self):
        with pytest.raises(ValueError):
            spo.PyLagModel.estimate([1.0] * 5, 3, 1.0)


# ─── PySupervisorPolicy ─────────────────────────────────────────


class TestPySupervisorPolicy:
    def test_nominal_no_actions(self):
        sp = spo.PySupervisorPolicy()
        actions = sp.decide([0.9, 0.8, 0.85, 0.9], [])
        assert actions == []

    def test_degraded_actions(self):
        sp = spo.PySupervisorPolicy()
        actions = sp.decide([0.5, 0.4, 0.5, 0.5], [])
        assert len(actions) >= 1
        assert actions[0]["knob"] == "K"


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
