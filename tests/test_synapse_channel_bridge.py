# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for SYNAPSE_CHANNEL bridge

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.adapters.synapse_channel_bridge import (
    SynapseChannelBridge,
)


class TestSynapseChannelBridge:
    def _make_bridge(self) -> SynapseChannelBridge:
        return SynapseChannelBridge(
            agents=["Claude", "Codex", "Gemini", "Human"],
        )

    def test_init_n_oscillators(self) -> None:
        bridge = self._make_bridge()
        assert bridge.n_oscillators == 4

    def test_get_phases_initial_zero(self) -> None:
        bridge = self._make_bridge()
        phases = bridge.get_phases()
        assert phases.shape == (4,)
        np.testing.assert_array_equal(phases, 0.0)

    def test_get_coupling_initial_zero(self) -> None:
        bridge = self._make_bridge()
        knm = bridge.get_coupling()
        assert knm.shape == (4, 4)
        np.testing.assert_array_equal(knm, 0.0)

    def test_process_heartbeat(self) -> None:
        bridge = self._make_bridge()
        bridge._process_message(
            {
                "sender": "Claude",
                "type": "heartbeat",
            }
        )
        state = bridge._states["Claude"]
        assert state.last_heartbeat > 0

    def test_process_heartbeat_interval(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Claude"].last_heartbeat = 100.0
        bridge._process_message(
            {
                "sender": "Claude",
                "type": "heartbeat",
            }
        )
        state = bridge._states["Claude"]
        assert len(state.heartbeat_intervals) == 1

    def test_process_chat(self) -> None:
        bridge = self._make_bridge()
        bridge._process_message(
            {
                "sender": "Codex",
                "type": "chat",
            }
        )
        assert bridge._states["Codex"].message_count == 1

    def test_process_claim(self) -> None:
        bridge = self._make_bridge()
        bridge._process_message(
            {
                "sender": "Gemini",
                "type": "claim_granted",
                "payload": "fix_bug_42",
            }
        )
        assert bridge._states["Gemini"].current_task == "fix_bug_42"

    def test_process_release(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Gemini"].current_task = "fix_bug_42"
        bridge._process_message(
            {
                "sender": "Gemini",
                "type": "release_granted",
            }
        )
        assert bridge._states["Gemini"].current_task is None

    def test_unknown_sender_ignored(self) -> None:
        bridge = self._make_bridge()
        bridge._process_message(
            {
                "sender": "UnknownAgent",
                "type": "heartbeat",
            }
        )

    def test_coupling_both_active(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Claude"].current_task = "task_a"
        bridge._states["Codex"].current_task = "task_b"
        knm = bridge.get_coupling()
        assert knm[0, 1] == 1.0
        assert knm[1, 0] == 1.0

    def test_coupling_one_idle(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Claude"].current_task = "task_a"
        knm = bridge.get_coupling()
        assert knm[0, 1] == 0.3

    def test_phases_advance_with_heartbeats(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Claude"].heartbeat_intervals = [2.0, 2.0, 2.0]
        phases = bridge.get_phases()
        assert phases[0] > 0  # freq=0.5, advance=π

    def test_agent_summary(self) -> None:
        bridge = self._make_bridge()
        bridge._states["Claude"].current_task = "review"
        bridge._states["Claude"].message_count = 3
        summary = bridge.get_agent_summary()
        assert summary["Claude"]["task"] == "review"
        assert summary["Claude"]["messages"] == 3

    def test_heartbeat_interval_cap(self) -> None:
        bridge = self._make_bridge()
        state = bridge._states["Claude"]
        state.last_heartbeat = 1.0
        for _ in range(25):
            bridge._process_message(
                {
                    "sender": "Claude",
                    "type": "heartbeat",
                }
            )
        assert len(state.heartbeat_intervals) <= 20

    def test_task_events_cap(self) -> None:
        bridge = self._make_bridge()
        for _ in range(25):
            bridge._process_message(
                {
                    "sender": "Claude",
                    "type": "claim_granted",
                    "payload": "task",
                }
            )
        assert len(bridge._states["Claude"].task_events) <= 20
