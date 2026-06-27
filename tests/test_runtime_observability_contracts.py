# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime observability contracts

"""Strict runtime observability contract tests for Prometheus and OTel paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.observability import (
    MetricsExporter,
    OTelExporter,
    PrometheusEvidenceSource,
    RuntimeMetricSnapshot,
    RuntimeObservability,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


class _EvidenceObject(PrometheusEvidenceSource):
    """Evidence source that exposes a concrete audit record."""

    def __init__(self, record: object) -> None:
        self._record = record

    def to_audit_record(self) -> Mapping[str, object]:
        """Return the configured audit record."""
        return cast("Mapping[str, object]", self._record)


class _MissingEvidenceMethod:
    """Object without the required Prometheus evidence method."""


def _state() -> UPDEState:
    """Return a deterministic runtime metric snapshot state."""
    return UPDEState(
        layers=[LayerState(R=0.75, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.75,
        regime_id="nominal",
        pac_max=0.1,
    )


def _digital_twin_record() -> dict[str, object]:
    """Return a valid digital-twin operator evidence record."""
    return {
        "contract_hash": "0" * 64,
        "accepted_count": 2,
        "rejected_count": 1,
        "adapter_count": 3,
        "unhealthy_adapter_count": 1,
        "latest_sequence": 42,
        "capability_counts": {"phase_observation": 2},
        "direction_counts": {"twin_to_spo": 3},
        "max_abs_twin_residual": 0.125,
        "mismatch_reasons": ["direction_not_allowed", "direction_not_allowed"],
        "status": "warning",
    }


def test_prometheus_evidence_base_requires_override() -> None:
    """The evidence protocol base fails closed until implemented."""
    with pytest.raises(NotImplementedError):
        PrometheusEvidenceSource().to_audit_record()


def test_runtime_prometheus_rejects_nonreal_state_metric() -> None:
    """Prometheus export rejects non-real UPDE layer metrics."""
    state = UPDEState(
        layers=[LayerState(R=cast(float, "not-real"), psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.75,
        regime_id="nominal",
    )

    with pytest.raises(ValueError, match="layer 0 R"):
        RuntimeObservability().prometheus_text(
            RuntimeMetricSnapshot(
                upde_state=state,
                regime="nominal",
                latency_ms=1.0,
            )
        )


def test_digital_twin_prometheus_rejects_malformed_evidence_fields() -> None:
    """Digital-twin Prometheus export validates every evidence field."""
    cases: tuple[tuple[str, object, str], ...] = (
        ("accepted_count", cast(object, True), "accepted_count"),
        ("rejected_count", -1, "rejected_count"),
        ("capability_counts", [], "capability_counts must be a mapping"),
        ("capability_counts", {"": 1}, "keys must be non-empty strings"),
        (
            "direction_counts",
            {"bad\nkey": 1},
            "keys must not contain control characters",
        ),
        ("mismatch_reasons", "not-a-sequence", "mismatch_reasons"),
        ("mismatch_reasons", [""], "entries must be non-empty strings"),
        (
            "mismatch_reasons",
            ["bad\nreason"],
            "entries must not contain control characters",
        ),
        ("max_abs_twin_residual", -0.1, "max_abs_twin_residual"),
    )

    for field, value, match in cases:
        record = _digital_twin_record()
        record[field] = value

        with pytest.raises(ValueError, match=match):
            MetricsExporter().export_digital_twin_operator_evidence(record)


def test_digital_twin_prometheus_accepts_evidence_object() -> None:
    """Digital-twin Prometheus export accepts protocol evidence objects."""
    text = RuntimeObservability().digital_twin_prometheus_text(
        _EvidenceObject(_digital_twin_record())
    )

    assert "spo_digital_twin_sync_accepted_total" in text
    assert 'status="warning"} 1' in text
    assert 'reason="direction_not_allowed"} 2' in text


def test_digital_twin_prometheus_rejects_missing_evidence_method() -> None:
    """Digital-twin Prometheus export rejects objects without audit records."""
    with pytest.raises(ValueError, match="to_audit_record"):
        MetricsExporter().export_digital_twin_operator_evidence(
            cast(PrometheusEvidenceSource, _MissingEvidenceMethod())
        )


def test_digital_twin_prometheus_rejects_non_mapping_evidence_record() -> None:
    """Digital-twin Prometheus export rejects malformed audit-record objects."""
    with pytest.raises(ValueError, match="must return a mapping"):
        MetricsExporter().export_digital_twin_operator_evidence(
            _EvidenceObject(["not", "a", "mapping"])
        )


def test_runtime_span_rejects_nonfinite_attribute_value() -> None:
    """Runtime spans reject non-finite primitive attributes."""
    with (
        pytest.raises(ValueError, match="finite"),
        RuntimeObservability().span("spo.step", {"latency": float("nan")}),
    ):
        pass


def test_otel_noop_fallback_span_and_metrics_paths() -> None:
    """The optional OTel fallback validates inputs and then records no data."""
    exporter = OTelExporter("spo")
    exporter._enabled = False

    with exporter.span("spo.step", {"step": 1}) as span:
        span.set_attribute("checked", True)
        span.set_status(None)
        span.end()

    exporter.record_step(_state(), step_idx=0)
    exporter.record_regime_change("nominal", "warning")


def test_runtime_facade_properties_and_noop_paths() -> None:
    """RuntimeObservability exposes OTel status and no-op step/span paths."""
    exporter = OTelExporter("spo")
    exporter._enabled = False
    observability = RuntimeObservability(otel_exporter=exporter)

    assert observability.otel_enabled is False
    observability.record_step(
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="nominal",
            latency_ms=1.0,
            step_idx=None,
        )
    )
    with observability.span("spo.step", {"step": 1}) as span:
        span.set_attribute("facade", True)
