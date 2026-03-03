from __future__ import annotations

import pytest

import scpn_phase_orchestrator.apps.queuewaves as qw


@pytest.mark.parametrize(
    "name",
    [
        "QueueWavesConfig",
        "ConfigCompiler",
        "MetricBuffer",
        "PrometheusCollector",
        "PhaseComputePipeline",
        "PipelineSnapshot",
        "AnomalyDetector",
        "Anomaly",
        "WebhookAlerter",
    ],
)
def test_lazy_import(name: str) -> None:
    obj = getattr(qw, name)
    assert obj is not None


def test_lazy_import_missing_raises() -> None:
    with pytest.raises(AttributeError, match="no_such_thing"):
        getattr(qw, "no_such_thing")
