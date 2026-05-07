# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — N-channel runtime execution tests

from __future__ import annotations

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
