# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves pipeline tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.apps.queuewaves.config import QueueWavesConfig
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PhaseComputePipeline,
    PipelineSnapshot,
)


def test_pipeline_tick_returns_snapshot(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    rng = np.random.default_rng(0)
    buffers = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
    snap = pipe.tick(buffers)
    assert isinstance(snap, PipelineSnapshot)
    assert snap.tick == 1
    assert 0.0 <= snap.r_good <= 1.0
    assert 0.0 <= snap.r_bad <= 1.0
    assert snap.regime in ("nominal", "degraded", "critical", "recovery")


def test_pipeline_accumulates_ticks(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    rng = np.random.default_rng(1)
    buffers = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
    for _ in range(5):
        snap = pipe.tick(buffers)
    assert snap.tick == 5
    assert pipe.tick_count == 5


def test_pipeline_snapshot_to_dict(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    rng = np.random.default_rng(2)
    buffers = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
    snap = pipe.tick(buffers)
    d = snap.to_dict()
    assert "r_good" in d
    assert "r_bad" in d
    assert "services" in d
    assert isinstance(d["plv_matrix"], list)


def test_pipeline_services_in_snapshot(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    rng = np.random.default_rng(3)
    buffers = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
    snap = pipe.tick(buffers)
    names = {s.name for s in snap.services}
    assert "svc-a" in names
    assert "throughput" in names


def test_pipeline_empty_buffers_still_ticks(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    snap = pipe.tick({})
    assert snap.tick == 1


def test_pipeline_sine_convergence(minimal_config: QueueWavesConfig) -> None:
    """Synchronized sine inputs should produce coherent R values."""
    pipe = PhaseComputePipeline(minimal_config)
    t = np.linspace(0, 4 * np.pi, 16)
    buffers = {svc.name: np.sin(t) for svc in minimal_config.services}
    for _ in range(20):
        snap = pipe.tick(buffers)
    # All services receiving identical signal → high coherence
    assert snap.r_bad > 0.3 or snap.r_good > 0.3


def test_pipeline_plv_matrix_shape(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    rng = np.random.default_rng(4)
    buffers = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
    snap = pipe.tick(buffers)
    n_layers = len({svc.layer for svc in minimal_config.services})
    assert len(snap.plv_matrix) == n_layers
    for row in snap.plv_matrix:
        assert len(row) == n_layers


def test_pipeline_regime_property(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    assert pipe.regime in ("nominal", "degraded", "critical", "recovery")


def test_pipeline_imprint_levels_property(minimal_config: QueueWavesConfig) -> None:
    pipe = PhaseComputePipeline(minimal_config)
    levels = pipe.imprint_levels
    assert levels.shape == (pipe._n_osc,)
    np.testing.assert_allclose(levels, 0.0)


def test_pipeline_empty_layer_osc_range() -> None:
    """A layer with no oscillators should yield r=0.0, psi=0.0 (line 181)."""
    from scpn_phase_orchestrator.apps.queuewaves.config import (
        CouplingConfig,
        QueueWavesConfig,
        ServerConfig,
        ServiceDef,
        ThresholdConfig,
    )
    from scpn_phase_orchestrator.binding.types import HierarchyLayer

    cfg = QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=[
            ServiceDef(name="a", promql="up", layer="micro"),
        ],
        scrape_interval_s=1.0,
        buffer_length=16,
        thresholds=ThresholdConfig(),
        coupling=CouplingConfig(),
        alert_sinks=[],
        server=ServerConfig(port=0),
    )
    pipe = PhaseComputePipeline(cfg)

    # Inject an empty-oscillator layer into the spec and ranges
    empty_layer = HierarchyLayer(name="ghost", index=99, oscillator_ids=[])
    pipe._spec.layers.append(empty_layer)
    pipe._layer_osc_ranges[99] = []

    rng = np.random.default_rng(5)
    buffers = {"a": rng.standard_normal(16)}
    snap = pipe.tick(buffers)
    assert snap.tick == 1
