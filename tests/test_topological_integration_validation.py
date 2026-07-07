# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — TopologicalIntegration validation contracts

"""Validation contracts for the TopologicalIntegrationObserver constructor and its
phase-observation boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.topological_integration import (
    TopologicalIntegrationObserver,
)


def test_u1_topological_integration_observer_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        TopologicalIntegrationObserver(tau_h1=1.5)


def test_u1_topological_integration_observer_rejects_boolean_tau() -> None:
    with pytest.raises(TypeError, match="finite real values"):
        TopologicalIntegrationObserver(tau_h1=True)  # type: ignore[arg-type]


def test_u1_topological_integration_observer_rejects_boolean_beta() -> None:
    with pytest.raises(TypeError, match="finite real values"):
        TopologicalIntegrationObserver(beta=True)  # type: ignore[arg-type]


def test_u1_topological_integration_observer_rejects_boolean_embed_dim() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        TopologicalIntegrationObserver(embed_dim=True)  # type: ignore[arg-type]


def test_u1_topological_integration_observer_rejects_boolean_embed_delay() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        TopologicalIntegrationObserver(embed_delay=True)  # type: ignore[arg-type]


def test_u1_topological_integration_observer_rejects_boolean_window_size() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        TopologicalIntegrationObserver(window_size=True)  # type: ignore[arg-type]


def test_u1_topological_integration_observer_observe_rejects_non_finite_phase() -> None:
    obs = TopologicalIntegrationObserver()
    with pytest.raises(ValueError, match="finite values"):
        obs.observe(np.array([0.0, np.nan], dtype=float))


def test_u1_topological_integration_observer_observe_rejects_boolean_phase_vector() -> (
    None
):
    obs = TopologicalIntegrationObserver()
    with pytest.raises(ValueError, match="boolean dtype"):
        obs.observe(np.array([True, False], dtype=bool))


def test_u1_topological_integration_observer_observe_rejects_non_array_input() -> None:
    obs = TopologicalIntegrationObserver()
    with pytest.raises(TypeError, match="numpy.ndarray"):
        obs.observe([0.0, 0.1])  # type: ignore[arg-type]


def test_u1_topological_integration_observer_observe_rejects_non_vector_input() -> None:
    obs = TopologicalIntegrationObserver()
    with pytest.raises(ValueError, match="1D vector"):
        obs.observe(np.zeros((2, 2), dtype=float))


def test_u1_topological_integration_observer_observe_rejects_empty_input() -> None:
    obs = TopologicalIntegrationObserver()
    with pytest.raises(ValueError, match="non-empty"):
        obs.observe(np.array([], dtype=float))
