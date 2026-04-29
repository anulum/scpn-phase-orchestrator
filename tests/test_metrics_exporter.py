# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MetricsExporter tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.metrics_exporter import MetricsExporter
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(
    r_values: list[float],
    regime: str = "nominal",
    pac_max: float = 0.3,
) -> UPDEState:
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    n = len(layers)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(n),
        stability_proxy=sum(r_values) / n if n else 0.0,
        regime_id=regime,
        pac_max=pac_max,
    )


def test_export_returns_string():
    exp = MetricsExporter()
    state = _make_state([0.9, 0.85])
    result = exp.export(state, "nominal", 1.23)
    assert isinstance(result, str)
    assert result.endswith("\n")


def test_export_contains_r_global():
    exp = MetricsExporter()
    state = _make_state([0.8, 0.6])
    result = exp.export(state, "nominal", 2.0)
    assert "spo_r_global" in result
    assert "0.700000" in result


def test_export_contains_latency():
    exp = MetricsExporter()
    state = _make_state([0.5])
    result = exp.export(state, "degraded", 5.5)
    assert "spo_latency_ms" in result
    assert "5.500" in result


def test_export_contains_regime_label():
    exp = MetricsExporter()
    state = _make_state([0.9])
    result = exp.export(state, "critical", 0.1)
    assert 'regime="critical"' in result


def test_exposition_lines_count():
    exp = MetricsExporter()
    state = _make_state([0.7, 0.8, 0.9])
    lines = exp.exposition_lines(state, "nominal", 1.0)
    # 5 metrics * 2 (HELP+TYPE) + 5 values + 3 per-layer = 18
    assert len(lines) >= 15


def test_custom_prefix():
    exp = MetricsExporter(prefix="myapp")
    state = _make_state([0.5])
    result = exp.export(state, "nominal", 1.0)
    assert "myapp_r_global" in result
    assert "spo_r_global" not in result


@pytest.mark.parametrize("prefix", ["", "1spo", "bad-name", "bad name"])
def test_invalid_prefix_rejected(prefix: str):
    with pytest.raises(ValueError, match="prefix"):
        MetricsExporter(prefix=prefix)


def test_prometheus_colon_prefix_is_valid():
    exp = MetricsExporter(prefix="spo:internal")
    state = _make_state([0.5])
    assert "spo:internal_r_global" in exp.export(state, "nominal", 1.0)


def test_export_pac_max():
    exp = MetricsExporter()
    state = _make_state([0.9], pac_max=0.42)
    result = exp.export(state, "nominal", 1.0)
    assert "spo_pac_max" in result
    assert "0.420000" in result


def test_per_layer_r_values():
    exp = MetricsExporter()
    state = _make_state([0.1, 0.2, 0.3])
    lines = exp.exposition_lines(state, "nominal", 1.0)
    layer_lines = [line for line in lines if "spo_layer_r" in line]
    assert len(layer_lines) == 3
    assert 'layer="0"' in layer_lines[0]
    assert 'layer="2"' in layer_lines[2]


class TestMetricsExporterPipelineWiring:
    """Pipeline: engine → UPDEState → MetricsExporter → Prometheus text."""

    def test_engine_state_to_prometheus_export(self):
        """Engine → R → UPDEState → export → Prometheus text format.
        Proves metrics exporter consumes engine output."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)

        state = _make_state([r])
        exp = MetricsExporter()
        text = exp.export(state, "nominal", 0.5)
        assert "spo_r_global" in text
        assert str(round(r, 6))[:4] in text
