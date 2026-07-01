# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — N-channel runtime execution tests

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scpn_phase_orchestrator.binding import (
    ChannelRuntimeExecutor,
    load_binding_spec,
)
from scpn_phase_orchestrator.upde.metrics import LayerState


def _write_runtime_spec(tmp_path) -> str:
    spec = {
        "name": "channel-runtime-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.02,
        "control_period_s": 0.1,
        "layers": [
            {
                "name": "plant",
                "index": 0,
                "oscillator_ids": ["p0"],
                "family": "plant",
            },
            {
                "name": "forecast",
                "index": 1,
                "oscillator_ids": ["f0"],
                "family": "forecast",
            },
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
            "forecast": {
                "channel": "Forecast",
                "extractor_type": "event",
                "config": {},
            },
        },
        "channels": {
            "P": {"role": "plant", "units": "rad"},
            "Forecast": {
                "role": "delayed_forecast",
                "required": False,
                "replay_semantics": "external",
                "metric_semantics": "delayed confidence interval",
            },
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "Forecast": {"confidence_weight": 0.25},
        },
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [],
        "actuators": [],
    }
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(path)


def _write_runtime_spec_payload(tmp_path, payload: dict) -> str:
    path = tmp_path / "binding_spec_payload.yaml"
    path.write_text(yaml.dump(payload), encoding="utf-8")
    return str(path)


def test_channel_runtime_holds_delayed_evidence_and_weights_uncertainty(tmp_path):
    spec = load_binding_spec(_write_runtime_spec(tmp_path))
    executor = ChannelRuntimeExecutor.from_spec(spec)

    first = executor.execute([LayerState(R=0.8, psi=0.1), LayerState(R=0.6, psi=0.2)])
    second = executor.execute([LayerState(R=0.9, psi=0.3), LayerState(R=0.2, psi=0.4)])

    assert first.layers[0].R == 0.8
    assert first.layers[1].R == 0.15
    assert first.evidence[1].evidence_source == "current_tick_prime"
    assert second.layers[0].R == 0.9
    assert second.layers[1].R == 0.15
    assert second.evidence[1].raw_R == 0.2
    assert second.evidence[1].executed_R == 0.15
    assert second.evidence[1].evidence_source == "held_previous_tick"
    assert second.evidence[1].confidence_weight == 0.25


def test_channel_runtime_audit_record_identifies_delayed_uncertain_layers(tmp_path):
    spec = load_binding_spec(_write_runtime_spec(tmp_path))
    executor = ChannelRuntimeExecutor.from_spec(spec)

    execution = executor.execute(
        [LayerState(R=0.7, psi=0.1), LayerState(R=0.4, psi=0.2)]
    )
    record = execution.to_audit_record()
    layer_records = record["layers"]

    assert record["delayed_layers"] == [1]
    assert record["uncertain_layers"] == [1]
    assert layer_records[1]["delay_policy"] == "hold_last_runtime_evidence"
    assert layer_records[1]["uncertainty_policy"] == (
        "confidence_weight_runtime_contribution"
    )


def test_channel_runtime_rejects_layer_count_mismatch(tmp_path):
    spec = load_binding_spec(_write_runtime_spec(tmp_path))
    executor = ChannelRuntimeExecutor.from_spec(spec)

    with pytest.raises(ValueError, match="raw layer count must match"):
        executor.execute([LayerState(R=0.7, psi=0.1)])


def test_channel_runtime_non_finite_confidence_fails_closed_to_unit_weight(tmp_path):
    spec_path = Path(_write_runtime_spec(tmp_path))
    with spec_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["drivers"]["Forecast"]["confidence_weight"] = float("inf")
    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    spec = load_binding_spec(str(spec_path))
    executor = ChannelRuntimeExecutor.from_spec(spec)

    execution = executor.execute(
        [LayerState(R=0.7, psi=0.1), LayerState(R=0.4, psi=0.2)]
    )

    assert execution.layers[1].R == 0.4
    assert execution.evidence[1].confidence_weight == 1.0


def test_channel_runtime_defaults_to_current_tick_policy_for_undeclared_channel(
    tmp_path,
):
    spec = {
        "name": "channel-runtime-default-policy",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.02,
        "control_period_s": 0.1,
        "layers": [
            {
                "name": "latent",
                "index": 0,
                "oscillator_ids": ["h0"],
                "family": "latent",
            },
        ],
        "oscillator_families": {
            "latent": {"channel": "H", "extractor_type": "event", "config": {}}
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
        },
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = _write_runtime_spec_payload(tmp_path, spec)

    executor = ChannelRuntimeExecutor.from_spec(load_binding_spec(spec_path))
    execution = executor.execute([LayerState(R=0.42, psi=0.11)])

    assert execution.layers[0].R == 0.42
    assert execution.evidence[0].channel == "H"
    assert execution.evidence[0].delay_policy == "use_current_tick_evidence"
    assert (
        execution.evidence[0].uncertainty_policy == "deterministic_runtime_contribution"
    )
    assert execution.evidence[0].evidence_source == "current_tick"
    assert execution.evidence[0].confidence_weight == 1.0

    record = execution.to_audit_record()
    assert record["delayed_layers"] == []
    assert record["uncertain_layers"] == []


def test_channel_runtime_defaults_to_physical_channel_for_unassigned_layer(
    tmp_path,
) -> None:
    spec = {
        "name": "channel-runtime-unassigned-layer",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.02,
        "control_period_s": 0.1,
        "layers": [
            {"name": "unassigned", "index": 0, "oscillator_ids": ["p0"]},
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}}
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = _write_runtime_spec_payload(tmp_path, spec)

    executor = ChannelRuntimeExecutor.from_spec(load_binding_spec(spec_path))
    execution = executor.execute([LayerState(R=0.72, psi=0.31)])

    assert execution.evidence[0].channel == "P"
    assert execution.evidence[0].executed_R == 0.72


def test_channel_runtime_confidence_weights_clamped_to_unit_interval(tmp_path):
    spec = {
        "name": "channel-runtime-confidence",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.02,
        "control_period_s": 0.1,
        "layers": [
            {
                "name": "plant",
                "index": 0,
                "oscillator_ids": ["p0"],
                "family": "plant",
            },
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}}
        },
        "channels": {
            "P": {
                "role": "uncertain",
                "replay_semantics": "confidence",
            }
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }

    def run_with_weight(value: float) -> float:
        payload = {
            **spec,
            "drivers": {
                "physical": {"confidence_weight": value},
                "informational": {},
                "symbolic": {},
            },
        }
        spec_path = _write_runtime_spec_payload(tmp_path, payload)
        executor = ChannelRuntimeExecutor.from_spec(load_binding_spec(spec_path))
        execution = executor.execute([LayerState(R=0.8, psi=0.2)])
        return execution.layers[0].R

    assert run_with_weight(3.0) == 0.8
    assert run_with_weight(-0.5) == 0.0


def test_channel_runtime_rejects_undefined_layer_family(
    tmp_path,
) -> None:
    spec = {
        "name": "channel-runtime-undefined-family",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.02,
        "control_period_s": 0.1,
        "layers": [
            {"name": "known", "index": 0, "oscillator_ids": ["p0"], "family": "plant"},
            {
                "name": "unknown",
                "index": 1,
                "oscillator_ids": ["x0"],
                "family": "ghost",
            },
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0, 1], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    path = _write_runtime_spec_payload(tmp_path, spec)

    with pytest.raises(
        ValueError,
        match=("layer 'unknown': family 'ghost' is not defined in oscillator_families"),
    ):
        ChannelRuntimeExecutor.from_spec(load_binding_spec(path))
