# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Resolved binding configuration tests

from __future__ import annotations

from typing import cast

import yaml

from scpn_phase_orchestrator.binding import (
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
)


def _write_spec(tmp_path, spec: dict) -> str:
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(path)


def test_resolved_binding_config_summarises_runtime_choices(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "name": "resolved-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.01,
            "control_period_s": 0.1,
            "layers": [
                {
                    "name": "plant",
                    "index": 0,
                    "oscillator_ids": ["p0", "p1"],
                    "family": "phys",
                },
                {"name": "events", "index": 1, "oscillator_ids": ["e0"]},
            ],
            "oscillator_families": {
                "phys": {
                    "channel": "P",
                    "extractor_type": "physical",
                    "config": {"window": 16},
                },
            },
            "coupling": {
                "base_strength": 0.45,
                "decay_alpha": 0.3,
                "templates": {"local": "nearest"},
            },
            "drivers": {
                "physical": {"frequency": 1.0, "zeta": 0.2},
                "informational": {},
                "symbolic": {},
            },
            "objectives": {"good_layers": [0], "bad_layers": [1]},
            "boundaries": [],
            "actuators": [],
        },
    )

    summary = resolved_binding_config(load_binding_spec(path))
    lines = format_resolved_binding_config(summary)

    assert summary["engine_mode"] == "kuramoto"
    assert summary["control_interval_steps"] == 10
    assert summary["oscillator_count"] == 3
    assert summary["unassigned_layer_count"] == 1
    assert any(
        "1 layer(s) have no explicit oscillator family" in line for line in lines
    )
    channels = summary["channels"]
    assert channels["P"]["families"] == ["phys"]
    assert channels["P"]["extractors"] == ["hilbert"]
    assert channels["P"]["driver_keys"] == ["frequency", "zeta"]
    assert channels["P"]["oscillator_count"] == 2


def test_resolved_binding_config_includes_named_channels(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "name": "resolved-extra-test",
            "version": "1.0.0",
            "safety_tier": "research",
            "sample_period_s": 0.02,
            "control_period_s": 0.1,
            "layers": [
                {
                    "name": "operator",
                    "index": 0,
                    "oscillator_ids": ["h0", "h1"],
                    "family": "human",
                }
            ],
            "oscillator_families": {
                "human": {"channel": "H", "extractor_type": "event", "config": {}},
            },
            "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
            "drivers": {
                "physical": {},
                "informational": {},
                "symbolic": {},
                "H": {"cadence_hz": 2.0},
            },
            "objectives": {"good_layers": [0], "bad_layers": []},
            "boundaries": [],
            "actuators": [],
        },
    )

    summary = resolved_binding_config(load_binding_spec(path))
    lines = format_resolved_binding_config(summary)

    assert "H" in summary["channels"]
    assert summary["channels"]["H"]["driver_keys"] == ["cadence_hz"]
    assert any("channel H:" in line for line in lines)


def test_resolved_binding_config_includes_channel_algebra(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "name": "resolved-nchannel-test",
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
                    "derived_from": ["P", "H"],
                    "derive_rule": "phase_lag(P,H)",
                },
            },
            "channel_groups": {
                "supervised": {
                    "channels": ["P", "H", "Risk"],
                    "description": "policy visible channels",
                }
            },
            "cross_channel_couplings": [
                {"source": "H", "target": "P", "strength": 0.2, "mode": "directed"}
            ],
            "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
            "drivers": {
                "physical": {},
                "informational": {},
                "symbolic": {},
                "H": {"cadence_hz": 2.0},
                "Risk": {},
            },
            "objectives": {"good_layers": [0, 1], "bad_layers": [2]},
            "boundaries": [],
            "actuators": [],
        },
    )

    summary = resolved_binding_config(load_binding_spec(path))
    lines = format_resolved_binding_config(summary)

    assert summary["channels"]["Risk"]["derived_from"] == ["P", "H"]
    assert summary["channels"]["Risk"]["replay_semantics"] == "derived"
    assert summary["channel_groups"]["supervised"]["channels"] == ["P", "H", "Risk"]
    assert summary["cross_channel_couplings"][0]["source"] == "H"
    channel_algebra = cast("dict[str, object]", summary["channel_algebra"])
    assert channel_algebra["required_channels"] == ["H", "P"]
    assert channel_algebra["optional_channels"] == ["Risk"]
    assert channel_algebra["derived_channels"] == ["Risk"]
    assert channel_algebra["delayed_channels"] == []
    assert channel_algebra["uncertain_channels"] == []
    assert channel_algebra["missing_required_channels"] == []
    assert any("metadata: role=derived_risk" in line for line in lines)
    assert any("channel_groups: supervised" in line for line in lines)
    assert any("channel_algebra:" in line for line in lines)


def test_resolved_binding_config_surfaces_missing_required_channel(tmp_path):
    path = _write_spec(
        tmp_path,
        {
            "name": "resolved-missing-channel-test",
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
                "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
            },
            "channels": {
                "P": {"role": "plant"},
                "Telemetry": {
                    "role": "external",
                    "required": True,
                    "replay_semantics": "external",
                },
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
        },
    )

    summary = resolved_binding_config(load_binding_spec(path))
    lines = format_resolved_binding_config(summary)
    channel_algebra = cast("dict[str, object]", summary["channel_algebra"])

    assert channel_algebra["missing_required_channels"] == ["Telemetry"]
    assert any("missing_required_channels: Telemetry" in line for line in lines)


def test_format_resolved_binding_config_handles_non_list_metadata_gracefully():
    summary = {
        "name": "resolved-format-malformed",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "control_interval_steps": 10,
        "engine_mode": "kuramoto",
        "layer_count": 1,
        "oscillator_count": 2,
        "channels": {
            "P": {
                "families": "physical",
                "extractors": "hilbert",
                "driver_keys": "gain",
                "declared": True,
                "role": "plant",
                "layer_count": 1,
                "oscillator_count": 2,
                "supervisor_visibility": True,
                "replay_semantics": "phase",
            }
        },
        "channel_groups": {},
        "cross_channel_couplings": [],
        "channel_algebra": {
            "required_channels": "P",
            "optional_channels": ["Q"],
            "derived_channels": "Risk",
            "missing_required_channels": "P",
            "delayed_channels": "P",
            "uncertain_channels": ["P"],
            "supervisor_visible_channels": "P",
            "coupling_participating_channels": ["P", "Q"],
        },
        "features": {
            "amplitude": False,
            "geometry_prior": False,
            "imprint_model": False,
            "protocol_net": False,
        },
    }

    lines = format_resolved_binding_config(summary)

    assert any("families=none" in line for line in lines)
    assert any("extractors=none" in line for line in lines)
    assert any("driver_keys=none" in line for line in lines)
    assert not any("missing_required_channels" in line for line in lines)


def test_resolved_binding_config_is_deterministic_across_channel_order(tmp_path):
    base = {
        "name": "resolved-channel-order-test",
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
            }
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}, "Q": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
        "channels": {
            "P": {
                "role": "plant",
                "replay_semantics": "phase",
                "derived_from": [],
            },
            "Q": {
                "role": "queue",
                "replay_semantics": "derived",
                "derived_from": ["P"],
                "derive_rule": "Q = P",
            },
        },
    }

    path_alpha = tmp_path / "resolved-order-a.yaml"
    path_alpha.write_text(yaml.dump(base), encoding="utf-8")
    summary_alpha = resolved_binding_config(load_binding_spec(path_alpha))

    swapped_channels = {
        "Q": base["channels"]["Q"],
        "P": base["channels"]["P"],
    }
    path_beta = tmp_path / "resolved-order-b.yaml"
    path_beta.write_text(
        yaml.dump({**base, "channels": swapped_channels}),
        encoding="utf-8",
    )
    summary_beta = resolved_binding_config(load_binding_spec(path_beta))

    assert format_resolved_binding_config(
        summary_alpha
    ) == format_resolved_binding_config(summary_beta)
    assert summary_alpha["channels"] == summary_beta["channels"]
    assert summary_alpha["channel_algebra"] == summary_beta["channel_algebra"]
