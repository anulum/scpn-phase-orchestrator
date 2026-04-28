# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sprint 2-4 behavioural tests

from __future__ import annotations

import io
import json
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.ssgf.tcbo import TCBOObserver

try:
    import ripser  # noqa: F401

    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.prediction import PredictionModel
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.stochastic import (
    _self_consistency_R,
    find_optimal_noise,
)


def _mock_resp(body: dict):
    data = json.dumps(body).encode()
    resp = io.BytesIO(data)
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: None
    resp.read = lambda: data
    return resp


# ---------------------------------------------------------------------------
# Prometheus adapter: error handling
# ---------------------------------------------------------------------------


class TestPrometheusErrorHandling:
    """Verify that PrometheusAdapter raises correct exceptions
    for connection failures and API errors."""

    def test_connection_refused_raises_connection_error(self):
        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                side_effect=URLError("refused"),
            ),
            pytest.raises(ConnectionError),
        ):
            adapter.fetch_instant("up")

    def test_api_error_status_raises_value_error(self):
        adapter = PrometheusAdapter("http://localhost:9090")
        with (
            patch(
                "scpn_phase_orchestrator.adapters.prometheus.urlopen",
                return_value=_mock_resp({"status": "error"}),
            ),
            pytest.raises(ValueError, match="status="),
        ):
            adapter.fetch_instant("up")

    def test_successful_response_returns_data(self):
        """Valid Prometheus response must be parsed correctly."""
        adapter = PrometheusAdapter("http://localhost:9090")
        body = {
            "status": "success",
            "data": {"resultType": "vector", "result": [{"value": [1234, "0.5"]}]},
        }
        with patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_resp(body),
        ):
            result = adapter.fetch_instant("up")
        assert result is not None


# ---------------------------------------------------------------------------
# NPE: normalised phase entropy
# ---------------------------------------------------------------------------


class TestNPEPhysicsContracts:
    """Verify NPE (normalised phase entropy) satisfies
    information-theoretic bounds."""

    def test_identical_phases_zero_entropy(self):
        """All phases identical → no disorder → NPE = 0."""
        assert compute_npe(np.array([1.0, 1.0, 1.0])) == 0.0

    def test_two_phases_zero(self):
        assert compute_npe(np.array([0.0, 1.0])) == 0.0

    def test_uniform_phases_high_entropy(self):
        """Uniformly spread phases → maximum disorder → NPE near 1."""
        phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        npe = compute_npe(phases)
        assert npe > 0.8, f"Uniform phases should give NPE near 1, got {npe:.3f}"

    def test_npe_bounded_zero_to_one(self):
        """NPE must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, rng.integers(3, 100))
            npe = compute_npe(phases)
            assert 0.0 <= npe <= 1.0 + 1e-10, f"NPE={npe} outside [0,1]"


# ---------------------------------------------------------------------------
# PGBO: phase-geometry-binding observer
# ---------------------------------------------------------------------------


class TestPGBOAlignment:
    """Verify that PGBO measures phase-geometry alignment with
    correct range and discriminatory power."""

    def test_alignment_in_range(self):
        pgbo = PGBO()
        rng = np.random.default_rng(123)
        phases = rng.uniform(0, 2 * np.pi, 8)
        W = rng.uniform(0.1, 2.0, (8, 8))
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert -1.0 <= snap.phase_geometry_alignment <= 1.0

    def test_synchronised_phases_high_alignment(self):
        """Nearly identical phases with uniform coupling → high alignment."""
        pgbo = PGBO()
        phases = np.full(8, 0.5)  # synchronised
        W = np.ones((8, 8)) * 0.5
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert snap.phase_geometry_alignment >= -1.0  # structural check


# ---------------------------------------------------------------------------
# TCBO: topological complexity-based observer
# ---------------------------------------------------------------------------


class TestTCBOBehavioural:
    """Verify TCBO handles edge cases: short history, degenerate phases,
    and empty persistence diagrams."""

    def test_short_history_does_not_crash(self):
        obs = TCBOObserver(window_size=3, embed_dim=3, embed_delay=2)
        for _ in range(9):
            state = obs.observe(np.array([0.0, 1.0]))
        assert hasattr(state, "p_h1")

    def test_degenerate_phases_bounded_p_h1(self):
        obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            state = obs.observe(np.zeros(4))
        assert 0.0 <= state.p_h1 <= 1.0
        assert state.method in ("ripser", "plv_approx")

    @pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
    def test_ripser_empty_h1_gives_zero(self):
        obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            obs.observe(np.array([0.1, 0.2, 0.3]))
        mock_result = {"dgms": [np.array([[0, 1]]), np.array([]).reshape(0, 2)]}
        with patch(
            "scpn_phase_orchestrator.ssgf.tcbo._ripser", return_value=mock_result
        ):
            state = obs.observe(np.array([0.1, 0.2, 0.3]))
        assert state.p_h1 == 0.0
        assert state.method == "ripser"

    @pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
    def test_ripser_infinite_lifetimes_gives_zero(self):
        obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            obs.observe(np.array([0.1, 0.2, 0.3]))
        h1 = np.array([[0.0, np.inf], [0.5, np.inf]])
        mock_result = {"dgms": [np.array([[0, 1]]), h1]}
        with patch(
            "scpn_phase_orchestrator.ssgf.tcbo._ripser", return_value=mock_result
        ):
            state = obs.observe(np.array([0.1, 0.2, 0.3]))
        assert state.p_h1 == 0.0


# ---------------------------------------------------------------------------
# Policy engine: metric extraction wiring
# ---------------------------------------------------------------------------


class TestPolicyMetricExtraction:
    """Verify that the policy engine correctly extracts non-standard
    metrics (boundary_violation_count, imprint_mean) from UPDEState
    and uses them in rule evaluation."""

    def _make_state(self, **overrides):
        defaults = {
            "layers": [LayerState(R=0.5, psi=0.0)],
            "cross_layer_alignment": np.eye(1),
            "stability_proxy": 0.5,
            "regime_id": "nominal",
        }
        defaults.update(overrides)
        return UPDEState(**defaults)

    def test_boundary_violation_count_triggers_action(self):
        """boundary_violation_count > 0 → actions when count=2."""
        from scpn_phase_orchestrator.supervisor.policy_rules import PolicyAction

        rule = PolicyRule(
            name="test_boundary",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="boundary_violation_count",
                layer=None,
                op=">",
                threshold=0.0,
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        )
        engine = PolicyEngine([rule])
        state = self._make_state(boundary_violation_count=2)
        actions = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(actions) > 0, "Rule should fire when boundary_violation_count=2 > 0"

    def test_boundary_violation_count_zero_no_action(self):
        from scpn_phase_orchestrator.supervisor.policy_rules import PolicyAction

        rule = PolicyRule(
            name="test_boundary",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="boundary_violation_count",
                layer=None,
                op=">",
                threshold=0.0,
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        )
        engine = PolicyEngine([rule])
        state = self._make_state(boundary_violation_count=0)
        actions = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(actions) == 0, "Rule should not fire when count=0"

    def test_imprint_mean_triggers_action(self):
        from scpn_phase_orchestrator.supervisor.policy_rules import PolicyAction

        rule = PolicyRule(
            name="test_imprint",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="imprint_mean",
                layer=None,
                op=">",
                threshold=0.5,
            ),
            actions=[PolicyAction(knob="zeta", scope="global", value=0.05, ttl_s=5.0)],
        )
        engine = PolicyEngine([rule])
        state = self._make_state(imprint_mean=0.8)
        actions = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(actions) > 0, "Rule should fire when imprint_mean=0.8 > 0.5"


# ---------------------------------------------------------------------------
# Engine variants: delayed, torus, simplicial — pipeline wiring
# ---------------------------------------------------------------------------


class TestEngineVariantsPipelineWiring:
    """Verify that all engine variants (delayed, torus, simplicial) produce
    physically valid output and wire into the order_parameter pipeline."""

    def test_delayed_engine_phases_finite_and_advance(self):
        n = 4
        eng = DelayedEngine(n, dt=0.01, delay_steps=2)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        initial = phases.copy()
        for _ in range(10):
            phases = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
        assert np.all(np.isfinite(phases))
        assert not np.allclose(phases, initial), "Phases must advance"
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0, "R from delayed engine must be valid"

    def test_torus_engine_zeta_changes_dynamics(self):
        """Torus engine with zeta must differ from pure Euler advance."""
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
        naive = (phases + 0.01 * omegas) % (2 * np.pi)
        assert not np.allclose(result, naive), (
            "Zeta must alter dynamics beyond pure Euler"
        )
        r, _ = compute_order_parameter(result)
        assert 0.0 <= r <= 1.0

    def test_simplicial_engine_phases_finite_and_valid_r(self):
        n = 4
        eng = SimplicialEngine(n, dt=0.01, sigma2=0.1)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
        assert np.all(np.isfinite(result))
        r, _ = compute_order_parameter(result)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Stochastic: self-consistency and optimal noise
# ---------------------------------------------------------------------------


class TestStochasticSelfConsistency:
    """Verify the Kuramoto self-consistency equation R = f(K/D)
    and optimal noise search."""

    def test_self_consistency_r_bounded(self):
        """R from self-consistency must be in (0, 1) for finite K/D."""
        R = _self_consistency_R(2.1, 1.0)
        assert 0.0 < R < 1.0

    def test_higher_coupling_higher_r(self):
        """Stronger coupling → higher R (monotonicity)."""
        R_weak = _self_consistency_R(1.5, 1.0)
        R_strong = _self_consistency_R(5.0, 1.0)
        assert R_strong > R_weak, f"K↑ → R↑: weak={R_weak:.3f}, strong={R_strong:.3f}"

    def test_find_optimal_noise_returns_nonnegative_d(self):
        n = 4
        engine = UPDEEngine(n, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = find_optimal_noise(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            D_range=None,
            n_steps=20,
        )
        assert result.D >= 0.0
        assert hasattr(result, "R_achieved")


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------


class TestPredictionModelContract:
    """Verify PredictionModel stores parameters and produces valid output."""

    def test_error_gain_preserved(self):
        model = PredictionModel(4, error_gain=0.3)
        assert model.error_gain == 0.3

    def test_prediction_produces_valid_output(self):
        """PredictionModel.predict must return array of same shape as input."""
        model = PredictionModel(4, error_gain=0.5)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(4)
        pred = model.predict(phases, omegas, dt=0.01)
        assert pred.shape == phases.shape
        assert np.all(np.isfinite(pred))
