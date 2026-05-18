# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime observability tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters import metrics_exporter as legacy_metrics
from scpn_phase_orchestrator.adapters import opentelemetry as legacy_otel
from scpn_phase_orchestrator.runtime import observability
from scpn_phase_orchestrator.runtime.observability import (
    MetricsExporter,
    OTelExporter,
    RuntimeMetricSnapshot,
    RuntimeObservability,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state() -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=0.8, psi=0.0), LayerState(R=0.6, psi=0.1)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.7,
        regime_id="nominal",
        pac_max=0.2,
    )


def test_legacy_observability_aliases_target_runtime_module() -> None:
    assert legacy_metrics is observability
    assert legacy_otel is observability


def test_runtime_observability_exports_default_prometheus_text() -> None:
    runtime_obs = RuntimeObservability()

    text = runtime_obs.prometheus_text(
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="nominal",
            latency_ms=1.25,
            step_idx=7,
        )
    )

    assert "spo_r_global" in text
    assert "spo_stability_proxy" in text
    assert "spo_step 7" in text
    assert 'regime="nominal"' in text


def test_runtime_observability_records_step_without_otel_backend() -> None:
    runtime_obs = RuntimeObservability()

    runtime_obs.record_step(
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="nominal",
            latency_ms=0.0,
            step_idx=3,
        )
    )


def test_runtime_span_validates_attributes_without_otel_backend() -> None:
    runtime_obs = RuntimeObservability()

    with pytest.raises(ValueError, match="attributes"), runtime_obs.span(
        "spo.step",
        cast(dict[str, object], {"bad": object()}),
    ):
        pass


def test_metrics_exporter_rejects_malformed_step_index() -> None:
    exporter = MetricsExporter()

    with pytest.raises(ValueError, match="step_idx"):
        exporter.export(_state(), "nominal", 0.0, step_idx=cast(int, True))


@pytest.mark.parametrize("service_name", ["", "1spo", "bad service"])
def test_otel_exporter_rejects_malformed_service_name(service_name: str) -> None:
    with pytest.raises(ValueError, match="service_name"):
        OTelExporter(service_name)


def test_otel_exporter_rejects_nonfinite_state_before_noop_return() -> None:
    exporter = OTelExporter("spo")
    bad_state = UPDEState(
        layers=[LayerState(R=0.8, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=float("nan"),
        regime_id="nominal",
    )

    with pytest.raises(ValueError, match="stability_proxy"):
        exporter.record_step(bad_state, step_idx=0)
