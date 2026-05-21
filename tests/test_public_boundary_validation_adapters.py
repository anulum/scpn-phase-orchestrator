# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — adapter boundary validation tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import PeerState
from scpn_phase_orchestrator.adapters.remanentia_bridge import CoherenceMemorySnapshot
from scpn_phase_orchestrator.adapters.synapse_channel_bridge import AgentState


def test_peer_state_rejects_invalid_node_id_control_characters() -> None:
    with pytest.raises(ValueError, match="control characters"):
        PeerState(node_id="peer\none", R=0.5, psi=1.0, timestamp=1.0)


def test_peer_state_rejects_non_unit_interval_r() -> None:
    with pytest.raises(ValueError, match=r"R must be in \[0, 1\]"):
        PeerState(node_id="peer-one", R=1.1, psi=1.0, timestamp=1.0)


def test_coherence_memory_snapshot_rejects_invalid_novelty_score() -> None:
    with pytest.raises(ValueError, match=r"novelty_score must be a finite float in \[0, 1\]"):
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
    with pytest.raises(ValueError, match="message_count must be a non-negative integer"):
        AgentState(message_count=-1)


def test_agent_state_rejects_nonfinite_phase() -> None:
    with pytest.raises(ValueError, match="phase_p must be finite"):
        AgentState(phase_p=float("nan"))


def test_agent_state_wraps_phase_values_into_two_pi() -> None:
    state = AgentState(phase_p=7.5, phase_i=8.0, phase_s=9.0)
    assert 0.0 <= state.phase_p < 2.0 * 3.141592653589793
    assert 0.0 <= state.phase_i < 2.0 * 3.141592653589793
    assert 0.0 <= state.phase_s < 2.0 * 3.141592653589793
