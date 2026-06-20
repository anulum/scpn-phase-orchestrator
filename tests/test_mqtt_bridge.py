# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MQTT edge sensor bridge tests

"""Tests for the MQTT phase bridge.

The configuration, payload decoding, message ingestion, and phase extraction are
exercised without a broker or ``paho-mqtt``; the live subscribe path is exercised
with an injected fake client that replays messages through the bridge callback.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters import mqtt_bridge
from scpn_phase_orchestrator.adapters.mqtt_bridge import (
    HAS_PAHO_MQTT,
    MqttBridgeConfig,
    MqttPhaseBridge,
    MqttTag,
)

HOST = "broker.local"


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------


def test_tag_round_trips_audit_record() -> None:
    tag = MqttTag(topic="plant/temp", name="temp", payload_format="json")
    record = tag.to_audit_record()
    assert record["topic"] == "plant/temp"
    assert record["payload_format"] == "json"
    assert json.loads(json.dumps(record)) == record


def test_tag_rejects_empty_topic() -> None:
    with pytest.raises(ValueError, match="topic"):
        MqttTag(topic="  ", name="temp")


def test_tag_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="channel"):
        MqttTag(topic="t", name="temp", channel="Z")


def test_tag_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        MqttTag(topic="t", name="temp", sample_rate_hz=0.0)


def test_tag_rejects_unknown_payload_format() -> None:
    with pytest.raises(ValueError, match="payload_format"):
        MqttTag(topic="t", name="temp", payload_format="binary")


def test_tag_rejects_non_finite_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        MqttTag(topic="t", name="temp", scale=float("inf"))


def test_config_round_trips_audit_record() -> None:
    config = MqttBridgeConfig(broker_host=HOST, tags=(MqttTag(topic="t", name="temp"),))
    record = config.to_audit_record()
    assert record["broker_host"] == HOST
    assert json.loads(json.dumps(record)) == record


def test_config_rejects_empty_host() -> None:
    with pytest.raises(ValueError, match="broker_host"):
        MqttBridgeConfig(broker_host="", tags=(MqttTag(topic="t", name="temp"),))


def test_config_rejects_bad_port() -> None:
    with pytest.raises(ValueError, match="broker_port"):
        MqttBridgeConfig(
            broker_host=HOST, tags=(MqttTag(topic="t", name="temp"),), broker_port=0
        )


def test_config_rejects_bad_keepalive() -> None:
    with pytest.raises(ValueError, match="keepalive_s"):
        MqttBridgeConfig(
            broker_host=HOST, tags=(MqttTag(topic="t", name="temp"),), keepalive_s=0
        )


def test_config_rejects_empty_client_id() -> None:
    with pytest.raises(ValueError, match="client_id"):
        MqttBridgeConfig(
            broker_host=HOST, tags=(MqttTag(topic="t", name="temp"),), client_id=""
        )


def test_config_rejects_no_tags() -> None:
    with pytest.raises(ValueError, match="at least one tag"):
        MqttBridgeConfig(broker_host=HOST, tags=())


def test_config_rejects_duplicate_topics() -> None:
    with pytest.raises(ValueError, match="topics must be unique"):
        MqttBridgeConfig(
            broker_host=HOST,
            tags=(MqttTag(topic="t", name="a"), MqttTag(topic="t", name="b")),
        )


def test_config_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        MqttBridgeConfig(
            broker_host=HOST,
            tags=(MqttTag(topic="a", name="x"), MqttTag(topic="b", name="x")),
        )


# ---------------------------------------------------------------------
# Payload decoding
# ---------------------------------------------------------------------


def _bridge() -> MqttPhaseBridge:
    return MqttPhaseBridge.from_tags(
        HOST,
        [
            MqttTag(topic="plant/temp", name="temp", sample_rate_hz=16.0),
            MqttTag(
                topic="plant/flow",
                name="flow",
                scale=2.0,
                offset=1.0,
                payload_format="json",
            ),
        ],
    )


def test_decode_raw_payload() -> None:
    bridge = _bridge()
    assert bridge.decode_payload(bridge.config.tags[0], "0.5") == pytest.approx(0.5)
    assert bridge.decode_payload(bridge.config.tags[0], b"0.25") == pytest.approx(0.25)


def test_decode_raw_rejects_non_number() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="not a number"):
        bridge.decode_payload(bridge.config.tags[0], "hot")


def test_decode_raw_rejects_non_finite() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="must be finite"):
        bridge.decode_payload(bridge.config.tags[0], "inf")


def test_decode_json_number_and_object() -> None:
    bridge = _bridge()
    flow = bridge.config.tags[1]
    assert bridge.decode_payload(flow, "1.0") == pytest.approx(3.0)  # 2*1 + 1
    assert bridge.decode_payload(flow, '{"value": 0.5}') == pytest.approx(2.0)


def test_decode_json_rejects_invalid_json() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="not valid JSON"):
        bridge.decode_payload(bridge.config.tags[1], "{bad}")


def test_decode_json_rejects_non_real_value() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="value"):
        bridge.decode_payload(bridge.config.tags[1], '{"value": "x"}')


# ---------------------------------------------------------------------
# Ingestion and extraction
# ---------------------------------------------------------------------


def test_ingest_messages_groups_by_topic_and_ignores_unknown() -> None:
    bridge = _bridge()
    messages: list[tuple[str, bytes | str]] = [
        ("plant/temp", "1.0"),
        ("plant/flow", "0.5"),
        ("other/topic", "9.9"),
        ("plant/temp", b"1.1"),
    ]
    samples = bridge.ingest_messages(messages)
    assert samples["temp"] == [1.0, 1.1]
    assert samples["flow"] == [2.0]  # json 0.5 -> 2*0.5 + 1


def test_extract_phases_returns_state_per_tag() -> None:
    bridge = _bridge()
    sinusoid = (0.5 + 0.1 * np.sin(2.0 * np.pi * np.arange(64) / 16.0)).tolist()
    phases = bridge.extract_phases({"temp": sinusoid, "flow": [0.3, 0.4, 0.5, 0.6]})
    assert set(phases) == {"temp", "flow"}
    assert phases["temp"].channel == "P"
    assert np.isfinite(phases["temp"].theta)


def test_extract_phases_missing_tag_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="missing samples for tag 'flow'"):
        bridge.extract_phases({"temp": [0.1, 0.2]})


def test_extract_phases_empty_series_rejected() -> None:
    bridge = MqttPhaseBridge.from_tags(HOST, [MqttTag(topic="t", name="temp")])
    with pytest.raises(ValueError, match="has no samples"):
        bridge.extract_phases({"temp": []})


def test_extract_phases_non_finite_sample_rejected() -> None:
    bridge = MqttPhaseBridge.from_tags(HOST, [MqttTag(topic="t", name="temp")])
    with pytest.raises(ValueError, match="sample"):
        bridge.extract_phases({"temp": [0.1, float("nan")]})


def test_bridge_audit_record() -> None:
    bridge = _bridge()
    record = bridge.to_audit_record()
    assert record["paho_mqtt_available"] == HAS_PAHO_MQTT
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Live subscribe (injected fake client)
# ---------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload


class _FakeClient:
    """Replays per-topic messages through the registered on_message callback."""

    def __init__(self, per_topic: int) -> None:
        self.on_message: Any = None
        self._topics: list[str] = []
        self._per_topic = per_topic
        self.disconnected = False

    def connect(self, host: str, port: int, keepalive: int) -> None:
        return None

    def subscribe(self, topic: str) -> None:
        self._topics.append(topic)

    def loop_start(self) -> None:
        for topic in self._topics:
            # Deliver one extra message to exercise the count guard.
            for i in range(self._per_topic + 1):
                self.on_message(
                    self, None, _FakeMessage(topic, str(1.0 + 0.1 * i).encode())
                )
        # An unknown-topic message must be ignored by the callback.
        self.on_message(self, None, _FakeMessage("unknown/topic", b"9.0"))

    def loop_stop(self) -> None:
        return None

    def disconnect(self) -> None:
        self.disconnected = True


def test_collect_live_with_injected_client() -> None:
    bridge = _bridge()
    fake = _FakeClient(per_topic=3)
    samples = bridge.collect_live(samples_per_tag=3, timeout_s=2.0, client=fake)
    assert samples["temp"] == [1.0, 1.1, 1.2]  # capped at 3, extra ignored
    assert len(samples["flow"]) == 3
    assert fake.disconnected


def test_collect_live_times_out_with_partial_samples() -> None:
    bridge = MqttPhaseBridge.from_tags(HOST, [MqttTag(topic="t", name="temp")])
    fake = _FakeClient(per_topic=1)  # only 2 messages, target is 5
    samples = bridge.collect_live(samples_per_tag=5, timeout_s=0.05, client=fake)
    assert len(samples["temp"]) == 2


def test_collect_live_rejects_bad_count() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="samples_per_tag"):
        bridge.collect_live(samples_per_tag=0, client=_FakeClient(1))


def test_collect_live_rejects_bad_timeout() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="timeout_s"):
        bridge.collect_live(samples_per_tag=1, timeout_s=0.0, client=_FakeClient(1))


def test_collect_live_default_client_uses_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = MqttPhaseBridge.from_tags(HOST, [MqttTag(topic="t", name="temp")])
    monkeypatch.setattr(bridge, "connect", lambda: _FakeClient(per_topic=2))
    samples = bridge.collect_live(samples_per_tag=2, timeout_s=2.0)
    assert len(samples["temp"]) == 2


def test_connect_without_paho_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mqtt_bridge, "HAS_PAHO_MQTT", False)
    bridge = _bridge()
    with pytest.raises(RuntimeError, match="paho-mqtt is not installed"):
        bridge.connect()


@pytest.mark.skipif(not HAS_PAHO_MQTT, reason="paho-mqtt extra not installed")
def test_connect_constructs_client_without_connecting() -> None:
    bridge = MqttPhaseBridge.from_tags(
        HOST, [MqttTag(topic="t", name="temp")], use_tls=True
    )
    client = bridge.connect()
    assert client is not None
