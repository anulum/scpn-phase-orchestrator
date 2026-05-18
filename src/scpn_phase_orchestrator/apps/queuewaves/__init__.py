# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves cascade failure detector

"""Lazy public facade for the QueueWaves cascade-failure detector app.

QueueWaves wires Prometheus metric collection into phase extraction, SPO
pipeline diagnostics, anomaly detection, and optional webhook/HTTP presentation
surfaces. Public symbols are lazily imported so FastAPI, httpx, and related
optional dependencies are loaded only when the caller requests runtime pieces.
The facade itself does not scrape metrics, send alerts, or start servers.
"""

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
