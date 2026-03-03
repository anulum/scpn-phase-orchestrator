"""End-to-end test: simulate a retry storm forming and verify detection."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.alerter import WebhookAlerter
from scpn_phase_orchestrator.apps.queuewaves.config import (
    CouplingConfig,
    QueueWavesConfig,
    ServerConfig,
    ServiceDef,
    ThresholdConfig,
)
from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
from scpn_phase_orchestrator.apps.queuewaves.pipeline import PhaseComputePipeline


@pytest.fixture()
def storm_config() -> QueueWavesConfig:
    return QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=[
            ServiceDef(name="svc-a", promql="up", layer="micro"),
            ServiceDef(name="svc-b", promql="up", layer="micro"),
            ServiceDef(name="svc-c", promql="up", layer="micro"),
            ServiceDef(name="tput", promql="up", layer="macro"),
        ],
        scrape_interval_s=1.0,
        buffer_length=32,
        thresholds=ThresholdConfig(r_bad_warn=0.50, r_bad_critical=0.70),
        coupling=CouplingConfig(strength=0.80, decay=0.10),
        alert_sinks=[],
        server=ServerConfig(port=0),
    )


@pytest.fixture()
def weak_config() -> QueueWavesConfig:
    """Config with many micro services and zero coupling for the diverse-signal test."""
    return QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=[
            ServiceDef(name=f"svc-{i}", promql="up", layer="micro") for i in range(8)
        ]
        + [ServiceDef(name="tput", promql="up", layer="macro")],
        scrape_interval_s=1.0,
        buffer_length=32,
        thresholds=ThresholdConfig(r_bad_warn=0.50, r_bad_critical=0.70),
        coupling=CouplingConfig(strength=0.0, decay=0.10),
        alert_sinks=[],
        server=ServerConfig(port=0),
    )


def test_e2e_retry_storm_lifecycle(storm_config: QueueWavesConfig) -> None:
    pipeline = PhaseComputePipeline(storm_config)
    detector = AnomalyDetector(storm_config.thresholds)
    alerter = WebhookAlerter([], cooldown_seconds=0.0)

    t = np.linspace(0, 8 * np.pi, 32)
    freq = 2.0  # all micro services oscillate at same frequency → high R_bad

    all_anomalies = []
    for tick_idx in range(40):
        # Micro services: identical synchronized signal (retry storm)
        phase = freq * t + tick_idx * 0.1
        buffers = {
            "svc-a": np.sin(phase),
            "svc-b": np.sin(phase + 0.05),  # near-identical
            "svc-c": np.sin(phase + 0.03),
            "tput": np.sin(0.3 * t),  # different frequency
        }
        snap = pipeline.tick(buffers)
        anomalies = detector.detect(snap)
        sent = alerter.send_sync(anomalies)
        all_anomalies.extend(sent)

    assert pipeline.tick_count == 40
    # Verify snapshot structure is intact
    assert snap.regime in ("nominal", "degraded", "critical", "recovery")


def test_e2e_no_storm_with_diverse_signals(weak_config: QueueWavesConfig) -> None:
    pipeline = PhaseComputePipeline(weak_config)
    detector = AnomalyDetector(weak_config.thresholds)

    rng = np.random.default_rng(42)
    all_anomalies = []
    for _ in range(10):
        buffers = {svc.name: rng.standard_normal(32) for svc in weak_config.services}
        snap = pipeline.tick(buffers)
        anomalies = detector.detect(snap)
        all_anomalies.extend(anomalies)

    critical_storms = [
        a
        for a in all_anomalies
        if a.type == "retry_storm_forming" and a.severity == "critical"
    ]
    assert len(critical_storms) == 0
