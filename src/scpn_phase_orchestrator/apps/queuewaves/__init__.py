# SCPN Phase Orchestrator — QueueWaves cascade failure detector
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from typing import Any

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

_MODULES = {
    "QueueWavesConfig": "config",
    "ConfigCompiler": "config",
    "MetricBuffer": "collector",
    "PrometheusCollector": "collector",
    "PhaseComputePipeline": "pipeline",
    "PipelineSnapshot": "pipeline",
    "AnomalyDetector": "detector",
    "Anomaly": "detector",
    "WebhookAlerter": "alerter",
}


def __getattr__(name: str) -> Any:
    """Lazy imports — avoid pulling in fastapi/httpx at package level."""
    mod_name = _MODULES.get(name)
    if mod_name is not None:
        import importlib

        mod = importlib.import_module(
            f"scpn_phase_orchestrator.apps.queuewaves.{mod_name}"
        )
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
