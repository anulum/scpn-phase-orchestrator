# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — constructor validation tests across subsystems

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.alerter import WebhookAlerter
from scpn_phase_orchestrator.apps.queuewaves.collector import MetricBuffer
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard
from scpn_phase_orchestrator.runtime.distributed.sync import (
    DistributedSyncConfig,
    GossipIngestResult,
    PhaseGossipNode,
    PhaseSyncMessage,
    simulate_lossy_phase_gossip,
)
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.supervisor.events import EventBus


class TestWebhookAlerterValidation:
    @pytest.mark.parametrize(
        "cooldown_seconds",
        [-1.0, float("nan"), float("inf"), True, "300"],
    )
    def test_rejects_invalid_cooldown(self, cooldown_seconds: Any) -> None:
        with pytest.raises(
            ValueError, match="cooldown_seconds must be a finite non-negative real"
        ):
            WebhookAlerter(sinks=[], cooldown_seconds=cooldown_seconds)

    def test_accepts_zero_cooldown(self) -> None:
        WebhookAlerter(sinks=[], cooldown_seconds=0.0)

    def test_normalises_integer_cooldown_to_float(self) -> None:
        alerter = WebhookAlerter(sinks=[], cooldown_seconds=30)
        assert alerter._cooldown == 30.0


class TestMetricBufferValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=-5)


class TestSindyValidation:
    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            PhaseSINDy(threshold=-0.01)

    def test_rejects_zero_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=0)

    def test_rejects_negative_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=-3)


class TestLyapunovGuardValidation:
    def test_rejects_zero_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=0.0)

    def test_rejects_negative_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=-0.5)

    def test_default_basin_threshold_is_half_pi(self) -> None:
        m = LyapunovGuard()
        assert abs(m._basin_threshold - np.pi / 2.0) < 1e-12


class TestEventBusValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=-10)


class TestPgboValidation:
    def test_rejects_empty_cost_weights(self) -> None:
        with pytest.raises(ValueError, match="at least one weight"):
            PGBO(cost_weights=())

    def test_rejects_negative_cost_weight(self) -> None:
        with pytest.raises(ValueError, match="cost_weights must be non-negative"):
            PGBO(cost_weights=(1.0, -0.2, 0.1))

    def test_accepts_zero_cost_weight(self) -> None:
        PGBO(cost_weights=(1.0, 0.0, 0.1))


class TestDistributedSyncValidation:
    def test_config_rejects_malformed_inputs(self) -> None:
        with pytest.raises(ValueError, match="node_id must contain only letters"):
            DistributedSyncConfig(node_id="bad node", n_oscillators=4)

        with pytest.raises(ValueError, match="n_oscillators must be positive"):
            DistributedSyncConfig(node_id="edge-a", n_oscillators=0)

        with pytest.raises(ValueError, match="protocol_version must be positive"):
            DistributedSyncConfig(node_id="edge-a", n_oscillators=4, protocol_version=0)

        with pytest.raises(
            ValueError, match="peer_timeout_s must be positive and finite"
        ):
            DistributedSyncConfig(
                node_id="edge-a",
                n_oscillators=4,
                peer_timeout_s=0.0,
            )

    def test_from_wire_fails_closed_on_malformed_mapping(self) -> None:
        node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
        result = node.ingest({"node_id": "edge-b", "sequence": 1})

        assert not result.accepted
        assert "phase sync message kind mismatch" in result.reason

    def test_sync_audit_records_are_deterministic(self) -> None:
        result = simulate_lossy_phase_gossip(
            {
                "edge-a": np.array([0.0, 0.1, 0.2]),
                "edge-b": np.array([1.1, 1.2, 1.3]),
            },
            rounds=2,
            config=DistributedSyncConfig(
                node_id="template",
                n_oscillators=3,
                phase_blend=0.25,
                max_phase_step_rad=0.2,
            ),
        )

        audit = result.to_audit_record()
        assert audit["kind"] == "lossy_phase_gossip_replay"
        assert audit["nodes"] == ["edge-a", "edge-b"]
        assert audit["rounds"] == 2
        assert isinstance(audit["final_mean_pairwise_error"], float)


class TestGossipIngestRecordContract:
    def test_gossip_ingest_audit_record_keeps_reason_and_sequence(self) -> None:
        record = GossipIngestResult(
            accepted=False,
            reason="protocol_version mismatch",
            peer_id="edge-b",
            sequence=7,
        ).to_audit_record()

        assert record == {
            "accepted": False,
            "reason": "protocol_version mismatch",
            "peer_id": "edge-b",
            "sequence": 7,
        }

    def test_phase_sync_message_round_trip_wire_contract(self) -> None:
        message = PhaseSyncMessage.from_phases(
            node_id="edge-a",
            sequence=3,
            phases=np.array([0.1, 1.2]),
            wall_time_s=12.5,
        )
        wire_a = message.to_wire()
        wire_b = message.to_wire()

        assert wire_a == wire_b


# Pipeline wiring: these constructors sit on the boundary between binding
# spec / CLI flags / YAML config and the runtime subsystems. A ValueError
# at the boundary beats a confusing NaN or a zero-size queue 300 seconds
# into the simulation.
