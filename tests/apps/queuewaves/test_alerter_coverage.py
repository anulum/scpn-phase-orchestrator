"""Coverage tests for alerter.py — Slack formatting with suppressed count,
async send with actual sinks, and POST failure handling."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from scpn_phase_orchestrator.apps.queuewaves.alerter import (
    WebhookAlerter,
    _format_slack,
)
from scpn_phase_orchestrator.apps.queuewaves.config import AlertSink
from scpn_phase_orchestrator.apps.queuewaves.detector import Anomaly


def _anomaly(svc: str = "svc-a", severity: str = "warning") -> Anomaly:
    return Anomaly(
        type="retry_storm_forming",
        severity=severity,
        service=svc,
        value=0.55,
        threshold=0.50,
        tick=1,
        message="test anomaly",
    )


def test_format_slack_with_suppressed():
    payload = _format_slack(_anomaly(), suppressed=5)
    text = payload["attachments"][0]["blocks"][0]["text"]["text"]
    assert "5 suppressed" in text


def test_format_slack_critical_color():
    payload = _format_slack(_anomaly(severity="critical"))
    assert payload["attachments"][0]["color"] == "#FF0000"


def test_format_slack_unknown_severity():
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
def test_async_send_with_sink_post_failure():
    """Sink URL is unreachable — send should still return the anomaly list
    (HTTP errors are caught internally)."""

    async def _run():
        sink = AlertSink(url="http://127.0.0.1:1/nonexistent", format="generic")
        alerter = WebhookAlerter([sink], cooldown_seconds=0.0)
        sent = await alerter.send([_anomaly()])
        assert len(sent) == 1

    asyncio.run(_run())


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_slack_format():
    """Slack-formatted sink with unreachable URL."""

    async def _run():
        sink = AlertSink(url="http://127.0.0.1:1/slack", format="slack")
        alerter = WebhookAlerter([sink], cooldown_seconds=0.0)
        sent = await alerter.send([_anomaly()])
        assert len(sent) == 1

    asyncio.run(_run())


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_async_send_successful_post():
    """Mock httpx.AsyncClient to return 200 → resp.raise_for_status() succeeds."""
    from unittest.mock import AsyncMock, patch

    import httpx

    async def _run():
        mock_resp = httpx.Response(
            200,
            request=httpx.Request("POST", "http://fake:8080/hook"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            sink = AlertSink(url="http://fake:8080/hook", format="generic")
            alerter = WebhookAlerter([sink], cooldown_seconds=0.0)
            sent = await alerter.send([_anomaly()])
            assert len(sent) == 1
            mock_client.post.assert_called_once()

    asyncio.run(_run())
