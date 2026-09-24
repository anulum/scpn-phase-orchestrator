# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prometheus and OpenTelemetry export the same r_global

"""Both runtime exporters publish ``r_global`` as the mean layer ``R``.

The OpenTelemetry exporter set its ``spo.r_global`` gauge from
``stability_proxy``, a producer-defined quantity (a weighted mean, a
fidelity), while Prometheus published the mean layer ``R`` under the same
name. It also skipped the layer-``R`` validation Prometheus applies.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.observability import MetricsExporter, OTelExporter
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(r_values: list[float], *, stability_proxy: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in r_values],
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=stability_proxy,
        regime_id="nominal",
    )


@pytest.mark.parametrize("bad_r", [math.nan, math.inf])
def test_otel_record_step_validates_layer_r_like_prometheus(bad_r: float) -> None:
    state = _state([0.4, bad_r], stability_proxy=0.9)
    with pytest.raises(ValueError, match="layer 1 R"):
        MetricsExporter().export(state, "nominal", 1.0)
    with pytest.raises(ValueError, match="layer 1 R"):
        OTelExporter().record_step(state, step_idx=0)


def test_prometheus_r_global_is_the_mean_layer_r() -> None:
    text = MetricsExporter().export(
        _state([0.2, 0.6], stability_proxy=0.97), "nominal", 1.0
    )
    assert 'spo_r_global{regime="nominal"} 0.400000' in text
    assert 'spo_stability_proxy{regime="nominal"} 0.970000' in text


_OTEL_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

    from scpn_phase_orchestrator.runtime.observability import OTelExporter
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    state = UPDEState(
        layers=[LayerState(R=0.2, psi=0.0), LayerState(R=0.6, psi=0.0)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.97,
        regime_id="nominal",
    )
    OTelExporter().record_step(state, step_idx=0)
    values = {}
    for resource in reader.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                values[metric.name] = [p.value for p in metric.data.data_points]
    print(json.dumps(values))
    """
)


def _otel_sdk_installed() -> bool:
    # find_spec on a dotted name imports the parents and raises
    # ModuleNotFoundError when "opentelemetry" itself is absent (as in CI).
    try:
        return importlib.util.find_spec("opentelemetry.sdk.metrics") is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(
    not _otel_sdk_installed(),
    reason="opentelemetry-sdk not installed; the OTel export path is inactive",
)
def test_otel_r_global_gauge_is_the_mean_layer_r() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _OTEL_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    values = json.loads(completed.stdout.strip().splitlines()[-1])
    assert values["spo.r_global"] == [pytest.approx(0.4)]
    assert values["spo.stability_proxy"] == [pytest.approx(0.97)]
    assert values["spo.steps_total"] == [1]
