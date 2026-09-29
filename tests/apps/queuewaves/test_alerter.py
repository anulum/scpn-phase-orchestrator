# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves alerter tests

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from scpn_phase_orchestrator.apps.queuewaves.alerter import (
    WebhookAlerter,
    _format_generic,
    _format_slack,
)
from scpn_phase_orchestrator.apps.queuewaves.config import AlertSink
from scpn_phase_orchestrator.apps.queuewaves.detector import Anomaly


@pytest.fixture()
def webhook_server() -> Iterator[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """Receive real webhook requests and return route-specific HTTP statuses."""
    received: list[tuple[str, dict[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            """Capture JSON before acknowledging or refusing a webhook."""
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            received.append((self.path, payload))
            if self.path.startswith("/disconnect"):
                self.close_connection = True
                return
            self.send_response(500 if self.path.startswith("/fail") else 200)
            self.end_headers()
            self.wfile.write(b"PRIVATE_TOPOLOGY")

        def log_message(self, format: str, *args: object) -> None:
            """Keep the HTTP access log out of application-log assertions."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _anomaly(
    svc: str = "svc-a",
    type_: str = "retry_storm_forming",
    severity: str = "warning",
) -> Anomaly:
    return Anomaly(
        type=type_,
        severity=severity,
        service=svc,
        value=0.55,
        threshold=0.50,
        tick=1,
        message="test anomaly",
    )


def test_format_generic() -> None:
    payload = _format_generic(_anomaly(), suppressed=3)
    assert payload["type"] == "retry_storm_forming"
    assert payload["suppressed_count"] == 3


def test_format_slack() -> None:
    payload = _format_slack(_anomaly())
    assert "attachments" in payload
    assert (
        "retry_storm_forming" in payload["attachments"][0]["blocks"][0]["text"]["text"]
    )


def test_dedup_suppresses_within_cooldown() -> None:
    alerter = WebhookAlerter([], cooldown_seconds=300.0)
    a = _anomaly()
    sent1 = alerter.send_sync([a])
    assert len(sent1) == 1
    sent2 = alerter.send_sync([a])
    assert len(sent2) == 0


def test_dedup_allows_different_keys() -> None:
    alerter = WebhookAlerter([], cooldown_seconds=300.0)
    a1 = _anomaly(svc="svc-a")
    a2 = _anomaly(svc="svc-b")
    sent = alerter.send_sync([a1, a2])
    assert len(sent) == 2


def test_dedup_allows_after_cooldown() -> None:
    alerter = WebhookAlerter([], cooldown_seconds=0.0)
    a = _anomaly()
    alerter.send_sync([a])
    sent = alerter.send_sync([a])
    assert len(sent) == 1


def test_different_anomaly_types_not_deduped() -> None:
    alerter = WebhookAlerter([], cooldown_seconds=300.0)
    a1 = _anomaly(type_="retry_storm_forming")
    a2 = _anomaly(type_="chronic_degradation")
    sent = alerter.send_sync([a1, a2])
    assert len(sent) == 2


def test_format_generic_all_fields_present() -> None:
    """Generic payload must contain all anomaly fields for downstream parsing."""
    payload = _format_generic(_anomaly())
    for field in ["type", "severity", "service", "value", "threshold", "message"]:
        assert field in payload, f"Generic payload missing {field!r}"


def test_format_slack_color_maps_severity() -> None:
    """Warning severity → yellow/orange, not red or grey."""
    payload = _format_slack(_anomaly())
    color = payload["attachments"][0]["color"]
    assert color != "#FF0000", "Warning should not be red (that's critical)"
    assert color != "#808080", "Warning should not be grey (that's unknown)"


def test_dedup_state_persists_across_calls() -> None:
    """After first send, alerter's dedup state must remember the key."""
    alerter = WebhookAlerter([], cooldown_seconds=300.0)
    a = _anomaly()
    sent1 = alerter.send_sync([a])
    assert len(sent1) == 1
    # Second send of same anomaly should be suppressed
    sent2 = alerter.send_sync([a])
    assert len(sent2) == 0
    # Third send still suppressed
    sent3 = alerter.send_sync([a])
    assert len(sent3) == 0


_HAS_HTTPX = importlib.util.find_spec("httpx") is not None


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_no_sinks() -> None:
    async def _run() -> None:
        alerter = WebhookAlerter([], cooldown_seconds=300.0)
        sent = await alerter.send([_anomaly()])
        assert len(sent) == 1

    asyncio.run(_run())


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_dedup_suppresses() -> None:
    async def _run() -> None:
        alerter = WebhookAlerter([], cooldown_seconds=300.0)
        a = _anomaly()
        await alerter.send([a])
        sent = await alerter.send([a])
        assert len(sent) == 0

    asyncio.run(_run())


# Salvaged module-specific behavioural contracts from deleted bucket files.


def test_format_slack_with_suppressed() -> None:
    """Include the number of deduplicated anomalies in Slack text."""
    payload = _format_slack(_anomaly(), suppressed=5)
    text = payload["attachments"][0]["blocks"][0]["text"]["text"]
    assert "5 suppressed" in text


def test_format_slack_critical_color() -> None:
    """Mark critical anomalies with the configured Slack colour."""
    payload = _format_slack(_anomaly(severity="critical"))
    assert payload["attachments"][0]["color"] == "#FF0000"


def test_format_slack_unknown_severity() -> None:
    """Use the fallback Slack colour for an unknown severity."""
    a = Anomaly(
        type="x",
        severity="unknown",
        service="s",
        value=0.0,
        threshold=0.0,
        tick=0,
        message="m",
    )
    payload = _format_slack(a)
    assert payload["attachments"][0]["color"] == "#808080"


_HAS_HTTPX = importlib.util.find_spec("httpx") is not None


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_with_sink_post_failure(
    webhook_server: tuple[str, list[tuple[str, dict[str, Any]]]],
) -> None:
    """A refused HTTP delivery must not stop the remaining sink or anomaly."""
    base_url, received = webhook_server

    async def _run() -> None:
        sinks = [
            AlertSink(url=f"{base_url}/fail", format="generic"),
            AlertSink(url=f"{base_url}/accepted", format="generic"),
        ]
        anomalies = [_anomaly(svc="svc-a"), _anomaly(svc="svc-b")]
        alerter = WebhookAlerter(sinks, cooldown_seconds=0.0)
        sent = await alerter.send(anomalies)
        assert sent == anomalies

    asyncio.run(_run())
    assert [(path, payload["service"]) for path, payload in received] == [
        ("/fail", "svc-a"),
        ("/fail", "svc-b"),
        ("/accepted", "svc-a"),
        ("/accepted", "svc-b"),
    ]


@pytest.mark.parametrize("failure_route", ["fail", "disconnect"])
def test_async_send_failure_log_does_not_leak_sink_url(
    caplog: pytest.LogCaptureFixture,
    webhook_server: tuple[str, list[tuple[str, dict[str, Any]]]],
    failure_route: str,
) -> None:
    """Real HTTP failures must not echo sink URLs or response-body details."""
    base_url, received = webhook_server
    sink_url = f"{base_url}/{failure_route}?tenant=PRIVATE_TOPOLOGY"

    async def _run() -> None:
        sink = AlertSink(url=sink_url, format="generic")
        alerter = WebhookAlerter([sink], cooldown_seconds=0.0)
        with caplog.at_level("WARNING"):
            sent = await alerter.send([_anomaly()])
        assert len(sent) == 1
        text = caplog.text
        assert "alert POST failed for configured sink" in text
        assert sink_url not in text
        assert "PRIVATE_TOPOLOGY" not in text
        assert "500" not in text

    asyncio.run(_run())
    assert len(received) == 1
    assert received[0][0] == f"/{failure_route}?tenant=PRIVATE_TOPOLOGY"


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_slack_format() -> None:
    """Slack-formatted sink with unreachable URL."""

    async def _run() -> None:
        sink = AlertSink(url="http://127.0.0.1:1/slack", format="slack")
        alerter = WebhookAlerter([sink], cooldown_seconds=0.0)
        sent = await alerter.send([_anomaly()])
        assert len(sent) == 1

    asyncio.run(_run())


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
@pytest.mark.parametrize("sink_format", ["generic", "slack"])
def test_async_send_successful_post(
    webhook_server: tuple[str, list[tuple[str, dict[str, Any]]]],
    sink_format: str,
) -> None:
    """Transmit the configured payload once and suppress an immediate repeat."""
    base_url, received = webhook_server
    anomaly = _anomaly()

    async def _run() -> None:
        sink = AlertSink(url=f"{base_url}/accepted", format=sink_format)
        alerter = WebhookAlerter([sink], cooldown_seconds=300.0)
        assert await alerter.send([anomaly]) == [anomaly]
        assert await alerter.send([anomaly]) == []

    asyncio.run(_run())
    assert len(received) == 1
    path, payload = received[0]
    assert path == "/accepted"
    if sink_format == "generic":
        assert payload == {
            "type": anomaly.type,
            "severity": anomaly.severity,
            "service": anomaly.service,
            "value": anomaly.value,
            "threshold": anomaly.threshold,
            "tick": anomaly.tick,
            "message": anomaly.message,
            "suppressed_count": 0,
        }
    else:
        attachment = payload["attachments"][0]
        assert attachment["color"] == "#FFA500"
        assert attachment["blocks"][0]["text"] == {
            "type": "mrkdwn",
            "text": ":warning: *retry_storm_forming* [warning]\ntest anomaly",
        }


# Salvaged module-specific behavioural contracts from deleted broad tests.
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
