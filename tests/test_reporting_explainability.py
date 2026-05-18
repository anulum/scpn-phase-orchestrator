# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Explainability report tests

from __future__ import annotations

from scpn_phase_orchestrator.reporting.explainability import (
    build_explainability_report,
)


def test_explainability_report_ignores_non_finite_layer_and_stability_values() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": float("nan"),
                "layers": [
                    {"R": True},
                    {"R": float("nan")},
                    {"R": float("inf")},
                    {"R": 0.5},
                ],
                "actions": [
                    {
                        "knob": "K",
                        "scope": "global",
                        "value": 0.1,
                        "ttl_s": 1.0,
                    },
                ],
            },
        ],
    )

    assert report.final_stability == 0.0
    assert report.metric_summary == (
        "Layer 0: mean R=0.000, final R=0.000, min R=0.000, max R=0.000",
        "Layer 1: mean R=0.000, final R=0.000, min R=0.000, max R=0.000",
        "Layer 2: mean R=0.000, final R=0.000, min R=0.000, max R=0.000",
        "Layer 3: mean R=0.500, final R=0.500, min R=0.500, max R=0.500",
        "Stability proxy: mean=0.000, final=0.000, min=0.000, max=0.000",
    )
    assert report.action_explanations[0].evidence == (
        "mean layer R=0.125",
        "stability proxy=0.000",
        "L0 R=0.000",
        "L1 R=0.000",
        "L2 R=0.000",
        "L3 R=0.500",
    )


def test_explainability_report_ignores_malformed_action_numeric_fields() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.75,
                "layers": [{"R": 0.5}],
                "actions": [
                    {
                        "knob": "K",
                        "scope": "global",
                        "value": float("nan"),
                        "ttl_s": float("inf"),
                    },
                    {
                        "knob": "B",
                        "scope": "local",
                        "value": True,
                        "ttl_s": 2.0,
                    },
                ],
            },
        ],
    )

    first, second = report.action_explanations
    assert first.value == 0.0
    assert first.ttl_s == 0.0
    assert second.value == 0.0
    assert second.ttl_s == 2.0
