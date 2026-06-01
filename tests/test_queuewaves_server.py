# SPDX-License-Identifier: AGPL-3.0-or-later

"""QueueWaves server WebSocket ingress boundary contracts.

The WebSocket is a read-only telemetry stream. Client messages are accepted
only as bounded keepalives, never as command, configuration, or actuation
payloads.
"""

from scpn_phase_orchestrator.apps.queuewaves.server import (
    _is_keepalive_message,
    _websocket_message_exceeds_limit,
)


def test_keepalive_contract_accepts_only_explicit_keepalive_frames() -> None:
    for payload in ("", "ping", "pong", '{"type":"ping"}', '{"type":"pong"}'):
        assert _is_keepalive_message(payload) is True


def test_keepalive_contract_rejects_command_like_or_ambiguous_json() -> None:
    rejected = (
        '{"type":"tick"}',
        '{"type":"ping","data":{"command":"reset"}}',
        '{"type":true}',
        '["ping"]',
        "null",
        '{"type":NaN}',
        '{"type":"ping","type":"pong"}',
    )

    for payload in rejected:
        assert _is_keepalive_message(payload) is False


def test_websocket_message_limit_is_enforced_in_utf8_bytes() -> None:
    assert _websocket_message_exceeds_limit("a" * 1024) is False
    assert _websocket_message_exceeds_limit("é" * 512) is False
    assert _websocket_message_exceeds_limit("é" * 513) is True
