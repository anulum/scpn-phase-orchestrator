# SCPN Phase Orchestrator — QueueWaves cascade failure detector
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

__all__ = [
    "QueueWavesConfig",
    "ConfigCompiler",
    "MetricBuffer",
    "PrometheusCollector",
    "PhaseComputePipeline",
    "PipelineSnapshot",
    "AnomalyDetector",
    "Anomaly",
    "WebhookAlerter",
]


def __getattr__(name: str):  # noqa: C901
    """Lazy imports — avoid pulling in fastapi/httpx at package level."""
    _config = {"QueueWavesConfig", "ConfigCompiler"}
    _collect = {"MetricBuffer", "PrometheusCollector"}
    _pipe = {"PhaseComputePipeline", "PipelineSnapshot"}
    _detect = {"AnomalyDetector", "Anomaly"}

    if name in _config:
        from scpn_phase_orchestrator.apps.queuewaves import config as _m

        return getattr(_m, name)
    if name in _collect:
        from scpn_phase_orchestrator.apps.queuewaves import collector as _m

        return getattr(_m, name)
    if name in _pipe:
        from scpn_phase_orchestrator.apps.queuewaves import pipeline as _m

        return getattr(_m, name)
    if name in _detect:
        from scpn_phase_orchestrator.apps.queuewaves import detector as _m

        return getattr(_m, name)
    if name == "WebhookAlerter":
        from scpn_phase_orchestrator.apps.queuewaves import alerter

        return alerter.WebhookAlerter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
