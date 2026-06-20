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
from scpn_phase_orchestrator.binding import (
    build_digital_twin_binding_contract,
    build_digital_twin_operator_evidence,
    build_digital_twin_sync_envelope,
    load_binding_spec,
    validate_digital_twin_sync_envelope,
)
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


def test_runtime_observability_records_invalid_step_idx_as_fail_closed() -> None:
    runtime_obs = RuntimeObservability()

    with pytest.raises(ValueError, match="step_idx must be a non-negative integer"):
        runtime_obs.record_step(
            RuntimeMetricSnapshot(
                upde_state=_state(),
                regime="nominal",
                latency_ms=0.0,
                step_idx=-1,
            )
        )


def test_runtime_metric_snapshot_rejects_invalid_constructor_contracts() -> None:
    with pytest.raises(ValueError, match="upde_state"):
        RuntimeMetricSnapshot(  # type: ignore[arg-type]
            upde_state="not-state",
            regime="nominal",
            latency_ms=0.1,
        )
    with pytest.raises(ValueError, match="regime"):
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="",
            latency_ms=0.1,
        )
    with pytest.raises(ValueError, match="latency_ms"):
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="nominal",
            latency_ms=-0.1,
        )
    with pytest.raises(ValueError, match="step_idx"):
        RuntimeMetricSnapshot(
            upde_state=_state(),
            regime="nominal",
            latency_ms=0.1,
            step_idx=True,
        )


def test_metrics_exporter_enforces_prometheus_prefix() -> None:
    with pytest.raises(
        ValueError,
        match="prefix must be a valid Prometheus metric prefix",
    ):
        MetricsExporter(prefix="1spo")


def test_runtime_span_validates_attributes_without_otel_backend() -> None:
    runtime_obs = RuntimeObservability()

    with (
        pytest.raises(ValueError, match="attributes"),
        runtime_obs.span("spo.step", cast(dict[str, object], {"bad": object()})),
    ):
        pass


def test_runtime_span_rejects_invalid_name() -> None:
    runtime_obs = RuntimeObservability()

    with (
        pytest.raises(ValueError, match="span name"),
        runtime_obs.span("bad span name"),
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


def test_runtime_observability_prometheus_text_stable_and_escape_safe() -> None:
    runtime_obs = RuntimeObservability()
    snapshot = RuntimeMetricSnapshot(
        upde_state=_state(),
        regime='nomi"\\al',
        latency_ms=1.25,
        step_idx=7,
    )

    text_a = runtime_obs.prometheus_text(snapshot)
    text_b = runtime_obs.prometheus_text(snapshot)

    assert text_a == text_b
    assert 'regime="nomi\\"\\\\al"' in text_a


def test_runtime_observability_rejects_nonfinite_latency_ms() -> None:
    runtime_obs = RuntimeObservability()

    with pytest.raises(
        ValueError,
        match="latency_ms must be a finite non-negative real value",
    ):
        runtime_obs.prometheus_text(
            RuntimeMetricSnapshot(
                upde_state=_state(),
                regime="nominal",
                latency_ms=float("nan"),
                step_idx=7,
            )
        )


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


def test_runtime_observability_exports_digital_twin_operator_evidence() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    accepted = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=41,
        payload={"TwinResidual": -0.031},
    )
    rejected = build_digital_twin_sync_envelope(
        contract,
        capability="control_action_proposal",
        direction="twin_to_spo",
        sequence=42,
        payload={"knob": "K"},
    )
    evidence = build_digital_twin_operator_evidence(
        contract,
        (
            validate_digital_twin_sync_envelope(contract, accepted),
            validate_digital_twin_sync_envelope(contract, rejected),
        ),
    )

    text = RuntimeObservability().digital_twin_prometheus_text(evidence)

    assert "spo_digital_twin_sync_accepted_total" in text
    assert "spo_digital_twin_sync_rejected_total" in text
    assert "spo_digital_twin_max_abs_residual" in text
    assert "spo_digital_twin_latest_sequence" in text
    assert f'contract_hash="{contract.contract_hash}"' in text
    assert 'status="degraded"} 1' in text
    assert 'capability="phase_observation"} 1' in text
    assert 'direction="twin_to_spo"} 1' in text
    assert 'reason="direction_not_allowed"} 1' in text
    assert " 0.031000" in text


def test_digital_twin_operator_metrics_reject_invalid_evidence() -> None:
    exporter = MetricsExporter()
    malformed = {
        "contract_hash": "not-a-sha",
        "accepted_count": 0,
        "rejected_count": 0,
        "adapter_count": 0,
        "unhealthy_adapter_count": 0,
        "latest_sequence": None,
        "capability_counts": {},
        "direction_counts": {},
        "max_abs_twin_residual": None,
        "mismatch_reasons": [],
        "status": "healthy",
    }

    with pytest.raises(ValueError, match="contract_hash"):
        exporter.export_digital_twin_operator_evidence(malformed)

    malformed["contract_hash"] = "0" * 64
    malformed["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        exporter.export_digital_twin_operator_evidence(malformed)


def test_metrics_exporter_export_twin_confidence() -> None:
    from scpn_phase_orchestrator.monitor.twin_confidence import (
        TwinConfidenceScore,
        summarise_twin_confidence,
    )

    def _score(status: str, confidence: float) -> TwinConfidenceScore:
        return TwinConfidenceScore(
            confidence=confidence,
            status=status,
            phase_js_divergence=0.0,
            order_wasserstein=0.0,
            phase_js_z=0.0,
            order_w1_z=0.0,
            composite_z=0.0,
            phase_js_within_band=True,
            order_w1_within_band=True,
            backend="python",
            score_hash="x",
        )

    summary = summarise_twin_confidence(
        [_score("healthy", 1.0), _score("critical", 0.0)]
    )
    exporter = MetricsExporter(prefix="spo")
    text = exporter.export_twin_confidence(summary)
    assert "spo_twin_confidence_mean " in text
    assert "spo_twin_confidence_worst_status_level 2" in text


def test_runtime_observability_twin_confidence_facade() -> None:
    from scpn_phase_orchestrator.monitor.twin_confidence import (
        TwinConfidenceScore,
        summarise_twin_confidence,
    )

    summary = summarise_twin_confidence(
        [
            TwinConfidenceScore(
                confidence=0.8,
                status="healthy",
                phase_js_divergence=0.0,
                order_wasserstein=0.0,
                phase_js_z=0.0,
                order_w1_z=0.0,
                composite_z=0.0,
                phase_js_within_band=True,
                order_w1_within_band=True,
                backend="python",
                score_hash="x",
            )
        ]
    )
    obs = RuntimeObservability(metric_prefix="run")
    text = obs.twin_confidence_prometheus_text(summary)
    assert "run_twin_confidence_mean " in text
    assert "run_twin_confidence_tick_count 1" in text
