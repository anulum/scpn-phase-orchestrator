# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — TopologicalIntegration observer tests

from __future__ import annotations

import importlib.util
from typing import get_type_hints
from unittest.mock import patch

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.topological_integration import (
    TopologicalIntegrationObserver,
    TopologicalIntegrationState,
)
from tests.typing_contracts import assert_precise_ndarray_hint


class TestTopologicalIntegrationState:
    def test_dataclass(self):
        s = TopologicalIntegrationState(
            p_h1=0.8, is_integrated=True, s_h1=0.5, method="ripser"
        )
        assert s.p_h1 == 0.8
        assert s.is_integrated is True
        assert s.method == "ripser"


class TestTopologicalIntegrationObserverInit:
    def test_default_params(self):
        obs = TopologicalIntegrationObserver()
        assert obs.tau_h1 == pytest.approx(0.72)

    def test_custom_params(self):
        obs = TopologicalIntegrationObserver(
            tau_h1=0.5, embed_dim=5, embed_delay=2, window_size=100
        )
        assert obs.tau_h1 == 0.5

    def test_public_array_contracts_are_parameterised(self) -> None:
        for hint in [
            get_type_hints(TopologicalIntegrationObserver.observe)["phases"],
            get_type_hints(TopologicalIntegrationObserver._delay_embed)["return"],
        ]:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


class TestTopologicalIntegrationInsufficient:
    def test_insufficient_data(self):
        obs = TopologicalIntegrationObserver(window_size=50)
        phases = np.zeros(8)
        result = obs.observe(phases)
        assert result.method == "insufficient_data"
        assert result.p_h1 == 0.0
        assert result.is_integrated is False


_HAS_RIPSER = importlib.util.find_spec("ripser") is not None


@pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
class TestTopologicalIntegrationRipser:
    def test_synchronized_low_p_h1(self):
        obs = TopologicalIntegrationObserver(window_size=20, embed_dim=2, embed_delay=1)
        for _ in range(25):
            obs.observe(np.zeros(4))
        result = obs.observe(np.zeros(4))
        assert result.method == "ripser"
        assert result.p_h1 < 0.9

    def test_chaotic_higher_p_h1(self):
        obs = TopologicalIntegrationObserver(window_size=20, embed_dim=2, embed_delay=1)
        rng = np.random.default_rng(42)
        for _ in range(30):
            phases = rng.uniform(0, 2 * np.pi, 6)
            result = obs.observe(phases)
        assert result.method == "ripser"
        assert result.s_h1 >= 0.0

    def test_empty_h1_diagram(self):
        obs = TopologicalIntegrationObserver(window_size=5, embed_dim=2, embed_delay=1)
        for _ in range(10):
            obs.observe(np.array([0.0, 0.0]))
        result = obs.observe(np.array([0.0, 0.0]))
        assert result.p_h1 >= 0.0

    def test_window_truncation(self):
        obs = TopologicalIntegrationObserver(window_size=10, embed_dim=2, embed_delay=1)
        for i in range(50):
            obs.observe(np.array([float(i), float(i) * 0.5]))
        assert len(obs._history) <= 10 + 2 * 1


class TestTopologicalIntegrationDelayEmbed:
    def test_embed_shape(self):
        obs = TopologicalIntegrationObserver(window_size=20, embed_dim=3, embed_delay=2)
        for _ in range(25):
            obs.observe(np.ones(4))
        cloud = obs._delay_embed()
        assert cloud.ndim == 2
        assert cloud.shape[1] == 3 * 4

    def test_embed_small_T(self):
        obs = TopologicalIntegrationObserver(window_size=3, embed_dim=3, embed_delay=2)
        obs._history = [np.ones(2)] * 3
        cloud = obs._delay_embed()
        assert cloud.shape[0] >= 2


class TestTopologicalIntegrationReset:
    def test_reset_clears(self):
        obs = TopologicalIntegrationObserver()
        for _ in range(10):
            obs.observe(np.zeros(4))
        obs.reset()
        assert len(obs._history) == 0
        result = obs.observe(np.zeros(4))
        assert result.method == "insufficient_data"


class TestTopologicalIntegrationPipelineWiring:
    """Pipeline: engine phases → TopologicalIntegration → topological complexity."""

    def test_engine_phases_to_observer(self):
        """UPDEEngine → phases → TopologicalIntegrationObserver.observe → p_h1∈[0,1]."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        obs = TopologicalIntegrationObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            state = obs.observe(phases)

        assert isinstance(state, TopologicalIntegrationState)
        assert 0.0 <= state.p_h1 <= 1.0


# Salvaged module-specific behavioural contracts from deleted sprint file.


class TestTopologicalIntegrationBehavioural:
    """Edge cases: short history, degenerate phases, empty persistence diagrams."""

    def test_short_history_does_not_crash(self):
        obs = TopologicalIntegrationObserver(window_size=3, embed_dim=3, embed_delay=2)
        for _ in range(9):
            state = obs.observe(np.array([0.0, 1.0]))
        assert hasattr(state, "p_h1")

    def test_degenerate_phases_bounded_p_h1(self):
        obs = TopologicalIntegrationObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            state = obs.observe(np.zeros(4))
        assert 0.0 <= state.p_h1 <= 1.0
        assert state.method in ("ripser", "plv_approx")

    @pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
    def test_ripser_empty_h1_gives_zero(self):
        obs = TopologicalIntegrationObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            obs.observe(np.array([0.1, 0.2, 0.3]))
        mock_result = {"dgms": [np.array([[0, 1]]), np.array([]).reshape(0, 2)]}
        with patch(
            "scpn_phase_orchestrator.ssgf.topological_integration._ripser",
            return_value=mock_result,
        ):
            state = obs.observe(np.array([0.1, 0.2, 0.3]))
        assert state.p_h1 == 0.0
        assert state.method == "ripser"

    @pytest.mark.skipif(not _HAS_RIPSER, reason="ripser not installed")
    def test_ripser_infinite_lifetimes_gives_zero(self):
        obs = TopologicalIntegrationObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            obs.observe(np.array([0.1, 0.2, 0.3]))
        h1 = np.array([[0.0, np.inf], [0.5, np.inf]])
        mock_result = {"dgms": [np.array([[0, 1]]), h1]}
        with patch(
            "scpn_phase_orchestrator.ssgf.topological_integration._ripser",
            return_value=mock_result,
        ):
            state = obs.observe(np.array([0.1, 0.2, 0.3]))
        assert state.p_h1 == 0.0


# ---------------------------------------------------------------------------
# Policy engine: metric extraction wiring
# ---------------------------------------------------------------------------
