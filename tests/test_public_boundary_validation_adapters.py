# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — adapter boundary validation tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import PeerState
from scpn_phase_orchestrator.adapters.remanentia_bridge import CoherenceMemorySnapshot
from scpn_phase_orchestrator.adapters.synapse_channel_bridge import (
    AgentState,
    _normalise_hub_message,
    _validate_agents,
)
from scpn_phase_orchestrator.adapters.synapse_coupling_bridge import SynapseSnapshot


def test_peer_state_rejects_invalid_node_id_control_characters() -> None:
    with pytest.raises(ValueError, match="control characters"):
        PeerState(node_id="peer\none", R=0.5, psi=1.0, timestamp=1.0)


def test_peer_state_rejects_non_unit_interval_r() -> None:
    with pytest.raises(ValueError, match=r"R must be in \[0, 1\]"):
        PeerState(node_id="peer-one", R=1.1, psi=1.0, timestamp=1.0)


def test_coherence_memory_snapshot_rejects_invalid_novelty_score() -> None:
    with pytest.raises(
        ValueError, match=r"novelty_score must be a finite float in \[0, 1\]"
    ):
        CoherenceMemorySnapshot(
            R_global=0.7,
            regime="nominal",
            n_entities=12,
            n_memories=34,
            novelty_score=1.5,
            consolidation_suggested=True,
        )


def test_coherence_memory_snapshot_rejects_non_bool_consolidation_flag() -> None:
    with pytest.raises(ValueError, match="consolidation_suggested must be a bool"):
        CoherenceMemorySnapshot(
            R_global=0.7,
            regime="nominal",
            n_entities=12,
            n_memories=34,
            novelty_score=0.4,
            consolidation_suggested=1,
        )


def test_agent_state_rejects_negative_message_count() -> None:
    with pytest.raises(
        ValueError, match="message_count must be a non-negative integer"
    ):
        AgentState(message_count=-1)


def test_agent_state_rejects_nonfinite_phase() -> None:
    with pytest.raises(ValueError, match="phase_p must be finite"):
        AgentState(phase_p=float("nan"))


def test_agent_state_wraps_phase_values_into_two_pi() -> None:
    state = AgentState(phase_p=7.5, phase_i=8.0, phase_s=9.0)
    assert 0.0 <= state.phase_p < 2.0 * 3.141592653589793
    assert 0.0 <= state.phase_i < 2.0 * 3.141592653589793
    assert 0.0 <= state.phase_s < 2.0 * 3.141592653589793


@pytest.mark.parametrize("bad", [-1.0, float("nan"), True, "0.0"])
def test_agent_state_rejects_invalid_last_heartbeat(bad: object) -> None:
    with pytest.raises(ValueError, match="last_heartbeat must be a finite"):
        AgentState(last_heartbeat=bad)


def test_agent_state_rejects_non_list_heartbeat_intervals() -> None:
    with pytest.raises(ValueError, match="heartbeat_intervals must be a list"):
        AgentState(heartbeat_intervals="not-a-list")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), True, "1.0"])
def test_agent_state_rejects_non_positive_heartbeat_interval(bad: object) -> None:
    with pytest.raises(ValueError, match="heartbeat_intervals must be a list"):
        AgentState(heartbeat_intervals=[bad])


def test_agent_state_rejects_non_list_task_events() -> None:
    with pytest.raises(ValueError, match="task_events must be a list"):
        AgentState(task_events="not-a-list")


@pytest.mark.parametrize("bad", [-1.0, float("nan"), True, "0.0"])
def test_agent_state_rejects_invalid_task_event(bad: object) -> None:
    with pytest.raises(ValueError, match="task_events must be a list"):
        AgentState(task_events=[bad])


def test_agent_state_accepts_and_floats_valid_sequences() -> None:
    state = AgentState(heartbeat_intervals=[1, 2.5], task_events=[0, 0.5])
    assert state.heartbeat_intervals == [1.0, 2.5]
    assert state.task_events == [0.0, 0.5]
    assert all(isinstance(value, float) for value in state.heartbeat_intervals)
    assert all(isinstance(value, float) for value in state.task_events)


@pytest.mark.parametrize("bad", ["", 5])
def test_agent_state_rejects_invalid_current_task(bad: object) -> None:
    with pytest.raises(ValueError, match="current_task must be None or a non-empty"):
        AgentState(current_task=bad)


def test_agent_state_rejects_control_chars_in_current_task() -> None:
    with pytest.raises(ValueError, match="current_task must not contain control"):
        AgentState(current_task="task\x01")


def test_validate_agents_treats_none_as_empty_roster() -> None:
    assert _validate_agents(None) == []


def test_normalise_hub_message_rejects_control_chars_in_header() -> None:
    assert _normalise_hub_message({"sender": "node\x01", "type": "claim"}) is None


def test_synapse_snapshot_rejects_shape_mismatch_between_matrices() -> None:
    with pytest.raises(ValueError, match=r"gap_coupling must have shape \(2, 2\)"):
        SynapseSnapshot(
            knm_delta=np.zeros((2, 2)),
            gap_coupling=np.zeros((3, 3)),
            astrocyte_modulation=np.zeros(2),
            mean_weight_change=0.0,
            mean_conductance=0.0,
            mean_ca=0.0,
        )


def test_synapse_snapshot_rejects_negative_astrocyte_modulation() -> None:
    with pytest.raises(
        ValueError,
        match="astrocyte_modulation must contain only non-negative values",
    ):
        SynapseSnapshot(
            knm_delta=np.zeros((2, 2)),
            gap_coupling=np.zeros((2, 2)),
            astrocyte_modulation=np.array([0.1, -0.2]),
            mean_weight_change=0.0,
            mean_conductance=0.0,
            mean_ca=0.0,
        )
