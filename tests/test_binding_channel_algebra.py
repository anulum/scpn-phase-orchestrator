# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding channel algebra tests

from __future__ import annotations

from typing import cast, get_type_hints

import yaml

from scpn_phase_orchestrator.binding import (
    ChannelAlgebraReport,
    build_channel_algebra_report,
    load_binding_spec,
)
from scpn_phase_orchestrator.binding.types import BindingSpec


def _write_spec(tmp_path, spec: dict) -> str:
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(path)


def _nchannel_spec() -> dict:
    return {
        "name": "channel-algebra-test",
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
                "name": "operator",
                "index": 1,
                "oscillator_ids": ["h0"],
                "family": "operator",
            },
            {
                "name": "risk",
                "index": 2,
                "oscillator_ids": ["r0"],
                "family": "risk",
            },
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
            "operator": {"channel": "H", "extractor_type": "event", "config": {}},
            "risk": {"channel": "Risk", "extractor_type": "graph", "config": {}},
        },
        "channels": {
            "P": {"role": "plant", "units": "rad"},
            "H": {"role": "operator", "replay_semantics": "event"},
            "Risk": {
                "role": "derived_risk",
                "required": False,
                "replay_semantics": "derived",
                "coupling_participation": False,
                "derived_from": ["P", "H"],
                "derive_rule": "phase_lag(P,H)",
            },
            "Hidden": {
                "role": "diagnostic",
                "required": False,
                "supervisor_visibility": False,
            },
            "Forecast": {
                "role": "delayed_forecast",
                "required": False,
                "replay_semantics": "external",
                "metric_semantics": "delayed confidence interval",
            },
            "Estimate": {
                "role": "state_estimate",
                "required": False,
                "metric_semantics": "uncertain probabilistic state",
            },
        },
        "channel_groups": {
            "supervised": {
                "channels": ["P", "H", "Risk"],
                "description": "policy-visible channels",
            },
            "plant_view": {"channels": ["P", "Risk"]},
        },
        "cross_channel_couplings": [
            {"source": "H", "target": "P", "strength": 0.2, "mode": "directed"}
        ],
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {
            "physical": {"frequency": 1.0},
            "informational": {},
            "symbolic": {},
            "H": {"cadence_hz": 2.0},
            "Risk": {},
        },
        "objectives": {"good_layers": [0, 1], "bad_layers": [2]},
        "boundaries": [],
        "actuators": [],
    }


def test_channel_algebra_contract_is_typed() -> None:
    hints = get_type_hints(build_channel_algebra_report)

    assert hints["spec"] is BindingSpec
    assert hints["return"] is ChannelAlgebraReport


def test_channel_algebra_report_classifies_nchannel_relationships(tmp_path) -> None:
    spec = load_binding_spec(_write_spec(tmp_path, _nchannel_spec()))

    report = build_channel_algebra_report(spec)

    assert report.channels == (
        "Estimate",
        "Forecast",
        "H",
        "Hidden",
        "I",
        "P",
        "Risk",
        "S",
    )
    assert report.declared_channels == (
        "Estimate",
        "Forecast",
        "H",
        "Hidden",
        "P",
        "Risk",
    )
    assert report.required_channels == ("H", "P")
    assert report.optional_channels == ("Estimate", "Forecast", "Hidden", "Risk")
    assert report.derived_channels == ("Risk",)
    assert report.delayed_channels == ("Forecast",)
    assert report.uncertain_channels == ("Estimate", "Forecast")
    assert report.missing_required_channels == ()
    assert report.runtime_evidence_channels == ("H", "P", "Risk")
    assert "Hidden" not in report.supervisor_visible_channels
    assert "Risk" not in report.coupling_participating_channels
    assert report.replay_semantics["Risk"] == "derived"
    assert report.runtime_policies["Forecast"].delay_policy == (
        "hold_last_runtime_evidence"
    )
    assert report.runtime_policies["Forecast"].uncertainty_policy == (
        "confidence_weight_runtime_contribution"
    )
    assert report.runtime_policies["Forecast"].missing_policy == (
        "drop_optional_channel"
    )
    assert report.runtime_policies["P"].delay_policy == "use_current_tick_evidence"
    assert report.runtime_policies["P"].missing_policy == "block_required_channel"
    assert report.channel_groups["supervised"] == ("P", "H", "Risk")
    assert report.channel_membership["Risk"] == ("plant_view", "supervised")
    assert report.coupling_edges[0].source == "H"
    assert report.coupling_edges[0].target == "P"


def test_channel_algebra_report_flags_missing_required_runtime_evidence(
    tmp_path,
) -> None:
    raw_spec = _nchannel_spec()
    raw_spec["channels"]["Telemetry"] = {
        "role": "external",
        "required": True,
        "replay_semantics": "external",
    }

    spec = load_binding_spec(_write_spec(tmp_path, raw_spec))

    report = build_channel_algebra_report(spec)

    assert "Telemetry" in report.required_channels
    assert report.missing_required_channels == ("Telemetry",)


def test_channel_algebra_report_serialises_for_audit(tmp_path) -> None:
    spec = load_binding_spec(_write_spec(tmp_path, _nchannel_spec()))

    record = build_channel_algebra_report(spec).to_audit_record()
    membership = cast("dict[str, object]", record["channel_membership"])
    edges = cast("list[object]", record["coupling_edges"])
    first_edge = cast("dict[str, object]", edges[0])

    assert record["derived_channels"] == ["Risk"]
    assert record["delayed_channels"] == ["Forecast"]
    assert record["uncertain_channels"] == ["Estimate", "Forecast"]
    runtime_policies = cast("dict[str, object]", record["runtime_policies"])
    forecast_policy = cast("dict[str, object]", runtime_policies["Forecast"])
    assert forecast_policy["delay_policy"] == "hold_last_runtime_evidence"
    assert forecast_policy["uncertainty_policy"] == (
        "confidence_weight_runtime_contribution"
    )
    assert membership["P"] == ["plant_view", "supervised"]
    assert first_edge["mode"] == "directed"
