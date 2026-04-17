# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves webhook alerter

from __future__ import annotations

import logging
import time

from scpn_phase_orchestrator.apps.queuewaves.config import AlertSink
from scpn_phase_orchestrator.apps.queuewaves.detector import Anomaly

__all__ = ["WebhookAlerter"]

logger = logging.getLogger(__name__)

try:
    from httpx import HTTPError as _HTTPError
except ImportError:  # pragma: no cover
    _HTTPError = OSError  # type: ignore[assignment,misc]

_SEND_ERRORS: tuple[type[BaseException], ...] = (OSError, RuntimeError, _HTTPError)

_SEVERITY_COLORS = {"critical": "#FF0000", "warning": "#FFA500"}
_SEVERITY_EMOJI = {"critical": ":rotating_light:", "warning": ":warning:"}


def _format_slack(anomaly: Anomaly, suppressed: int = 0) -> dict:
    emoji = _SEVERITY_EMOJI.get(anomaly.severity, ":question:")
    color = _SEVERITY_COLORS.get(anomaly.severity, "#808080")
    text = anomaly.message
    if suppressed > 0:
        text += f" ({suppressed} suppressed since last alert)"
    return {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"{emoji} *{anomaly.type}* [{anomaly.severity}]\n{text}"
                            ),
                        },
                    }
                ],
            }
        ],
    }


def _format_generic(anomaly: Anomaly, suppressed: int = 0) -> dict:
    return {
        "type": anomaly.type,
        "severity": anomaly.severity,
        "service": anomaly.service,
        "value": anomaly.value,
        "threshold": anomaly.threshold,
        "tick": anomaly.tick,
        "message": anomaly.message,
        "suppressed_count": suppressed,
    }


class WebhookAlerter:
    """Posts anomaly alerts to configured webhook sinks with deduplication."""

    def __init__(self, sinks: list[AlertSink], cooldown_seconds: float = 300.0):
        self._sinks = sinks
        self._cooldown = cooldown_seconds
        self._last_fired: dict[str, float] = {}
        self._suppressed_count: dict[str, int] = {}

    async def send(self, anomalies: list[Anomaly]) -> list[Anomaly]:
        """Post anomalies to all sinks. Returns the list of actually sent anomalies."""
        import httpx

        now = time.time()
        to_send: list[tuple[Anomaly, int]] = []

        for a in anomalies:
            key = f"{a.type}:{a.service}"
            last = self._last_fired.get(key, 0.0)
            if now - last < self._cooldown:
                self._suppressed_count[key] = self._suppressed_count.get(key, 0) + 1
                continue
            suppressed = self._suppressed_count.pop(key, 0)
            self._last_fired[key] = now
            to_send.append((a, suppressed))

        if not to_send:
            return []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for sink in self._sinks:
                for anomaly, suppressed in to_send:
                    if sink.format == "slack":
                        payload = _format_slack(anomaly, suppressed)
                    else:
                        payload = _format_generic(anomaly, suppressed)
                    try:
                        resp = await client.post(sink.url, json=payload)
                        resp.raise_for_status()
                    except _SEND_ERRORS:
                        logger.warning(
                            "alert POST failed to %s",
                            sink.url,
                            exc_info=True,
                        )
        return [a for a, _ in to_send]

    def send_sync(self, anomalies: list[Anomaly]) -> list[Anomaly]:
        """Synchronous dedup-only path for testing (no HTTP)."""
        now = time.time()
        sent: list[Anomaly] = []
        for a in anomalies:
            key = f"{a.type}:{a.service}"
            last = self._last_fired.get(key, 0.0)
            if now - last < self._cooldown:
                self._suppressed_count[key] = self._suppressed_count.get(key, 0) + 1
                continue
            self._suppressed_count.pop(key, 0)
            self._last_fired[key] = now
            sent.append(a)
        return sent
