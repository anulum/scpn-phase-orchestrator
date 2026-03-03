from __future__ import annotations

from scpn_phase_orchestrator.apps.queuewaves.alerter import (
    WebhookAlerter,
    _format_generic,
    _format_slack,
)
from scpn_phase_orchestrator.apps.queuewaves.detector import Anomaly


def _anomaly(svc: str = "svc-a", type_: str = "retry_storm_forming") -> Anomaly:
    return Anomaly(
        type=type_,
        severity="warning",
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
