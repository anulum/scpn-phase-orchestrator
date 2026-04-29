# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves pipeline tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import (
    ConfigCompiler,
    QueueWavesConfig,
    ServiceDef,
)
from scpn_phase_orchestrator.apps.queuewaves.pipeline import PhaseComputePipeline


def _make_config(n_services: int = 3) -> QueueWavesConfig:
    services = [
        ServiceDef(name=f"svc_{i}", promql=f"rate(req{{svc='{i}'}}[1m])", layer=ly)
        for i, ly in enumerate(["micro", "meso", "macro"][:n_services])
    ]
    return QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=services,
        scrape_interval_s=1.0,
        buffer_length=16,
    )


def test_compiler_produces_valid_spec():
    cfg = _make_config()
    compiler = ConfigCompiler()
    spec = compiler.compile(cfg)
    assert spec.name == "queuewaves"
    assert len(spec.layers) == 3
    assert spec.objectives.bad_layers == [0]
    assert 1 in spec.objectives.good_layers


def test_compiler_preserves_named_extension_channels():
    cfg = QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=[
            ServiceDef(name="cpu", promql="cpu", layer="micro", channel="P"),
            ServiceDef(name="events", promql="events", layer="meso", channel="I"),
            ServiceDef(name="state", promql="state", layer="macro", channel="S"),
            ServiceDef(
                name="quantum",
                promql="quantum",
                layer="macro",
                channel="Q_control",
                extractor_type="event",
            ),
        ],
        scrape_interval_s=1.0,
        buffer_length=16,
    )

    spec = ConfigCompiler().compile(cfg)

    assert spec.oscillator_families["cpu"].channel == "P"
    assert spec.oscillator_families["cpu"].extractor_type == "hilbert"
    assert spec.oscillator_families["events"].channel == "I"
    assert spec.oscillator_families["events"].extractor_type == "event"
    assert spec.oscillator_families["state"].channel == "S"
    assert spec.oscillator_families["state"].extractor_type == "ring"
    assert spec.oscillator_families["quantum"].channel == "Q_control"
    assert spec.oscillator_families["quantum"].extractor_type == "event"


def test_named_extension_channel_defaults_to_hilbert_extractor():
    svc = ServiceDef(name="custom", promql="custom", layer="micro", channel="QoS")

    assert svc.resolved_extractor_type() == "hilbert"


def test_service_rejects_invalid_channel_identifier():
    with pytest.raises(ValueError, match="valid binding channel identifier"):
        ServiceDef(name="bad", promql="bad", layer="micro", channel="bad channel")


def test_service_rejects_invalid_extractor_type():
    with pytest.raises(ValueError, match="valid extractor"):
        ServiceDef(
            name="bad",
            promql="bad",
            layer="micro",
            channel="Q",
            extractor_type="unsafe",
        )


def test_pipeline_tick():
    cfg = _make_config()
    pipeline = PhaseComputePipeline(cfg)
    rng = np.random.default_rng(0)
    buffers = {f"svc_{i}": rng.standard_normal(32) for i in range(3)}
    snap = pipeline.tick(buffers)
    assert snap.tick == 1
    assert 0.0 <= snap.r_good <= 1.0
    assert 0.0 <= snap.r_bad <= 1.0
    assert snap.regime in ("nominal", "degraded", "critical", "recovery")
    assert len(snap.services) == 3


def test_pipeline_multiple_ticks():
    cfg = _make_config()
    pipeline = PhaseComputePipeline(cfg)
    rng = np.random.default_rng(7)
    for _ in range(10):
        buffers = {f"svc_{i}": rng.standard_normal(32) for i in range(3)}
        snap = pipeline.tick(buffers)
    assert snap.tick == 10
    assert len(snap.plv_matrix) == 3


def test_pipeline_snapshot_to_dict():
    cfg = _make_config()
    pipeline = PhaseComputePipeline(cfg)
    rng = np.random.default_rng(1)
    buffers = {f"svc_{i}": rng.standard_normal(32) for i in range(3)}
    snap = pipeline.tick(buffers)
    d = snap.to_dict()
    assert isinstance(d, dict)
    assert d["tick"] == 1
    assert "services" in d
    assert "plv_matrix" in d


def test_pipeline_empty_buffers():
    cfg = _make_config()
    pipeline = PhaseComputePipeline(cfg)
    snap = pipeline.tick({})
    assert snap.tick == 1


def test_pipeline_short_signal():
    cfg = _make_config()
    pipeline = PhaseComputePipeline(cfg)
    buffers = {"svc_0": np.array([1.0, 2.0])}
    snap = pipeline.tick(buffers)
    assert snap.tick == 1


# Pipeline wiring: QueueWaves has its own pipeline (Prometheus → collector →
# PhaseComputePipeline → detector → alerter). Tests above exercise tick(),
# snapshot_to_dict, empty buffers, and short signals through this pipeline.
