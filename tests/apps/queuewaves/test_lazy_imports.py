# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves lazy import tests

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
        qw.no_such_thing  # noqa: B018


def test_lazy_imported_classes_are_callable() -> None:
    """Lazily imported names must be actual classes, not broken stubs."""
    for name in ["QueueWavesConfig", "MetricBuffer", "AnomalyDetector", "Anomaly"]:
        obj = getattr(qw, name)
        assert callable(obj), f"{name} should be a callable class"


def test_lazy_import_consistency() -> None:
    """Importing the same name twice must return the same object (no re-exec)."""
    a = qw.PrometheusCollector
    b = qw.PrometheusCollector
    assert a is b
