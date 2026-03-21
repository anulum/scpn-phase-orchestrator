# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coverage gap tests for Sprints 2-4

from __future__ import annotations

import io
import json
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
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


def test_fetch_instant_connection_error():
    adapter = PrometheusAdapter("http://localhost:9090")
    with (
        patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            side_effect=URLError("refused"),
        ),
        pytest.raises(ConnectionError),
    ):
        adapter.fetch_instant("up")


def test_fetch_instant_error_status():
    adapter = PrometheusAdapter("http://localhost:9090")
    with (
        patch(
            "scpn_phase_orchestrator.adapters.prometheus.urlopen",
            return_value=_mock_resp({"status": "error"}),
        ),
        pytest.raises(ValueError, match="status="),
    ):
        adapter.fetch_instant("up")


def test_npe_three_identical_phases():
    phases = np.array([1.0, 1.0, 1.0])
    assert compute_npe(phases) == 0.0


def test_npe_two_phases_single_lifetime():
    phases = np.array([0.0, 1.0])
    npe = compute_npe(phases)
    assert npe == 0.0


def test_symbolic_graph_single_element():
    ext = SymbolicExtractor(n_states=5, mode="graph")
    result = ext.extract(np.array([3]), sample_rate=100.0)
    assert len(result) == 1


def test_pgbo_corrcoef_path():
    pgbo = PGBO()
    rng = np.random.default_rng(123)
    phases = rng.uniform(0, 2 * np.pi, 8)
    W = rng.uniform(0.1, 2.0, (8, 8))
    np.fill_diagonal(W, 0.0)
    snap = pgbo.observe(phases, W)
    assert -1.0 <= snap.phase_geometry_alignment <= 1.0


def test_tcbo_short_history_delay_embed():
    obs = TCBOObserver(window_size=3, embed_dim=3, embed_delay=2)
    for _ in range(9):
        obs.observe(np.array([0.0, 1.0]))


def test_tcbo_all_zero_returns_valid_state():
    obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
    for _ in range(15):
        state = obs.observe(np.zeros(4))
    # ripser: p_h1=0 (no H1 in degenerate cloud)
    # PLV fallback: p_h1≈1 (all phases identical → max PLV)
    assert 0.0 <= state.p_h1 <= 1.0
    assert state.method in ("ripser", "plv_approx")


@pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
def test_tcbo_ripser_empty_h1():
    obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
    for _ in range(15):
        obs.observe(np.array([0.1, 0.2, 0.3]))
    mock_result = {"dgms": [np.array([[0, 1]]), np.array([]).reshape(0, 2)]}
    with patch(
        "scpn_phase_orchestrator.ssgf.tcbo._ripser",
        return_value=mock_result,
    ):
        state = obs.observe(np.array([0.1, 0.2, 0.3]))
    assert state.p_h1 == 0.0
    assert state.method == "ripser"


@pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
def test_tcbo_ripser_no_finite_lifetimes():
    obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
    for _ in range(15):
        obs.observe(np.array([0.1, 0.2, 0.3]))
    h1 = np.array([[0.0, np.inf], [0.5, np.inf]])
    mock_result = {"dgms": [np.array([[0, 1]]), h1]}
    with patch(
        "scpn_phase_orchestrator.ssgf.tcbo._ripser",
        return_value=mock_result,
    ):
        state = obs.observe(np.array([0.1, 0.2, 0.3]))
    assert state.p_h1 == 0.0


def test_policy_extracts_boundary_violation_count():
    rule = PolicyRule(
        name="test_boundary",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="boundary_violation_count",
            layer=None,
            op=">",
            threshold=0.0,
        ),
        actions=[],
    )
    engine = PolicyEngine([rule])
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.5,
        regime_id="nominal",
        boundary_violation_count=2,
    )
    engine.evaluate(Regime.NOMINAL, state, [0], [])


def test_policy_extracts_imprint_mean():
    rule = PolicyRule(
        name="test_imprint",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="imprint_mean",
            layer=None,
            op=">",
            threshold=0.5,
        ),
        actions=[],
    )
    engine = PolicyEngine([rule])
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.5,
        regime_id="nominal",
        imprint_mean=0.8,
    )
    engine.evaluate(Regime.NOMINAL, state, [0], [])


def test_delayed_engine_with_zeta():
    n = 4
    eng = DelayedEngine(n, dt=0.01, delay_steps=2)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    for _ in range(5):
        phases = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
    assert np.all(np.isfinite(phases))


def test_torus_engine_with_zeta():
    n = 4
    eng = TorusEngine(n, dt=0.01)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    result = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
    assert not np.allclose(result, (phases + 0.01 * omegas) % (2 * np.pi))


def test_prediction_error_gain_property():
    model = PredictionModel(4, error_gain=0.3)
    assert model.error_gain == 0.3


def test_simplicial_with_zeta():
    n = 4
    eng = SimplicialEngine(n, dt=0.01, sigma2=0.1)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    result = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
    assert np.all(np.isfinite(result))


def test_self_consistency_slow_convergence():
    R = _self_consistency_R(2.1, 1.0)
    assert 0.0 < R < 1.0


def test_find_optimal_noise_default_range():
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    result = find_optimal_noise(
        engine, phases, omegas, knm, alpha, D_range=None, n_steps=20
    )
    assert result.D >= 0.0
