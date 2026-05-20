# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Explainability report tests

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from scpn_phase_orchestrator.reporting.explainability import (
    _make_pdf_bytes,
    _pdf_escape,
    _wrap_pdf_lines,
    build_explainability_report,
    render_markdown,
    write_markdown,
    write_pdf,
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


def test_explainability_report_ignores_malformed_step_identifiers() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.75,
                "layers": [{"R": 0.5}],
            },
            {
                "step": "not-an-integer",
                "regime": "degraded",
                "stability": 0.25,
                "layers": [{"R": 0.25}],
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

    assert report.regime_transitions == ("Step 0: nominal -> degraded",)
    assert report.action_explanations[0].step == 0


def test_explainability_report_ignores_malformed_layer_containers() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.75,
                "layers": None,
            },
            {
                "step": 1,
                "regime": "degraded",
                "stability": 0.25,
                "layers": ["not-a-layer", {"R": 0.4}],
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

    assert report.layers == 1
    assert report.metric_summary == (
        "Layer 0: mean R=0.400, final R=0.400, min R=0.400, max R=0.400",
        "Stability proxy: mean=0.500, final=0.250, min=0.250, max=0.750",
    )
    assert report.action_explanations[0].evidence == (
        "mean layer R=0.400",
        "stability proxy=0.250",
        "L0 R=0.400",
    )


def test_explainability_report_honors_zero_action_limit() -> None:
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
                        "value": 0.1,
                        "ttl_s": 1.0,
                    },
                ],
            },
        ],
        max_actions=0,
    )

    assert report.action_explanations == ()


def test_explainability_report_filters_step_records_and_ignores_non_steps() -> None:
    report = build_explainability_report(
        [
            {"event": "bootstrap", "ts": "2026-05-19T00:00:00Z"},
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.90,
                "layers": [{"R": 0.40}, {"R": 0.35}],
            },
            {
                "step": 1,
                "regime": "degraded",
                "stability": 0.60,
                "layers": [{"R": 0.20}, {"R": 0.55}],
                "event": "adaptive-loop",
                "ts": "2026-05-19T00:00:30Z",
                "_hash": "skipped",
            },
        ]
    )

    assert report.steps == 2
    assert report.layers == 2
    assert report.final_regime == "degraded"
    assert report.regime_transitions == ("Step 1: nominal -> degraded",)


def test_explainability_report_rejects_audit_without_step_records() -> None:
    with pytest.raises(ValueError, match="no step records in audit log"):
        build_explainability_report(
            [
                {"event": "bootstrap", "ts": "2026-05-19T00:00:00Z"},
                {"ts": "2026-05-19T00:00:01Z", "message": "no step"},
            ]
        )


def test_explainability_report_rejects_audit_without_valid_step_records() -> None:
    with pytest.raises(ValueError, match="no step records in audit log"):
        build_explainability_report([1, "corrupted", None, []])


def test_render_markdown_orders_regime_distribution() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "zeta",
                "stability": 0.9,
                "layers": [{"R": 0.1}],
            },
            {
                "step": 1,
                "regime": "alpha",
                "stability": 0.8,
                "layers": [{"R": 0.2}],
            },
            {
                "step": 2,
                "regime": "gamma",
                "stability": 0.7,
                "layers": [{"R": 0.3}],
            },
            {
                "step": 3,
                "regime": "alpha",
                "stability": 0.6,
                "layers": [{"R": 0.4}],
            },
        ]
    )
    markdown = render_markdown(report)
    dist_section = markdown.split("## Regime Distribution", 1)[1].split(
        "## Regime Transitions", 1
    )[0]
    rows = [line for line in dist_section.splitlines() if line.startswith("- ")]
    assert rows == [
        "- alpha: 2 steps (50.0%)",
        "- gamma: 1 steps (25.0%)",
        "- zeta: 1 steps (25.0%)",
    ]


def test_render_markdown_includes_missing_sections() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.5,
                "layers": [{"R": 0.33}],
            }
        ]
    )
    markdown = render_markdown(report)

    assert "- No control actions recorded." in markdown
    assert "- No auxiliary events recorded." in markdown


def test_render_markdown_notifies_absent_regime_transitions() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "stable",
                "stability": 0.82,
                "layers": [{"R": 0.55}],
            },
            {
                "step": 1,
                "regime": "stable",
                "stability": 0.80,
                "layers": [{"R": 0.49}],
            },
        ]
    )
    markdown = render_markdown(report)

    assert "- No regime transitions recorded." in markdown


def test_action_explanation_uses_layer_fallback_evidence() -> None:
    report = build_explainability_report(
        [
            {
                "step": 7,
                "regime": "nominal",
                "stability": 0.75,
                "layers": None,
                "actions": [
                    {
                        "knob": "throttle",
                        "scope": "global",
                        "value": 0.3,
                        "ttl_s": 12.5,
                    }
                ],
            }
        ]
    )

    assert report.metric_summary == (
        "Stability proxy: mean=0.750, final=0.750, min=0.750, max=0.750",
    )
    assert report.action_explanations[0].evidence == (
        "no layer R values recorded",
        "stability proxy=0.750",
    )


def test_action_explanation_skips_non_mapping_action_payloads() -> None:
    report = build_explainability_report(
        [
            {
                "step": 2,
                "regime": "watch",
                "stability": 0.5,
                "layers": [{"R": 0.44}],
                "actions": [
                    "not-a-dict-action",
                    {
                        "knob": "kappa",
                        "scope": "local",
                        "value": 0.4,
                        "ttl_s": 8.0,
                    },
                ],
            }
        ]
    )

    assert len(report.action_explanations) == 1
    assert report.action_explanations[0].knob == "kappa"


def test_render_markdown_renders_scores_and_bounds() -> None:
    report = build_explainability_report(
        [
            {
                "step": 2,
                "regime": "watch",
                "stability": 0.33333,
                "layers": [{"R": 0.42}],
                "actions": [
                    {
                        "knob": "gain",
                        "scope": "local",
                        "value": 0.5,
                        "ttl_s": 10.4,
                    }
                ],
            }
        ]
    )
    markdown = render_markdown(report)

    assert "- Final stability proxy: 0.3333" in markdown
    assert "- Step 2: gain=0.5000 (local, ttl=10.4s) in watch." in markdown


def test_render_markdown_keeps_event_payload_when_detail_missing() -> None:
    report = build_explainability_report(
        [
            {
                "step": 3,
                "event": "heartbeat",
                "layers": [],
                "_hash": "ignore-me",
                "component": "watcher",
            }
        ]
    )

    assert report.events == (
        "Step 3: heartbeat — {'layers': [], 'component': 'watcher'}",
    )


def test_explainability_report_json_payload_is_serializable() -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.5,
                "layers": [{"R": 0.2}, {"R": 0.8}],
            }
        ]
    )

    payload = asdict(report)
    restored = json.loads(json.dumps(payload))

    assert restored["steps"] == report.steps
    assert restored["metric_summary"] == list(report.metric_summary)


def test_pdf_escape_escapes_pdf_control_characters() -> None:
    assert _pdf_escape(r"control\\token (alpha) /beta") == (
        r"control\\\\token \(alpha\) /beta"
    )


def test_wrap_pdf_lines_preserves_heading_and_width_contract() -> None:
    markdown = (
        "# explainability report\n\nvalue with a long value that should wrap\n**bold**"
    )
    lines = _wrap_pdf_lines(markdown, width=24)

    assert lines[0] == "EXPLAINABILITY REPORT"
    assert lines[1] == ""
    assert all(len(line) <= 24 for line in lines if line)
    assert lines[-1] == "**bold**"


def test_make_pdf_bytes_handles_wrapping_and_escapes() -> None:
    markdown_lines = _wrap_pdf_lines("## residual (line) test\nspecial\\line", width=20)
    pdf = _make_pdf_bytes(markdown_lines)

    assert b"%PDF-1.4" in pdf
    assert b"\\(" in pdf
    assert b"\\)" in pdf
    assert b"/Count 1" in pdf


def test_make_pdf_bytes_scales_to_multiple_pages() -> None:
    lines: list[str] = [f"line {idx} for residual PDF coverage" for idx in range(120)]
    pdf = _make_pdf_bytes(lines)

    assert b"/Type /Pages" in pdf
    assert b"/Count 3" in pdf
    assert pdf.count(b"/Type /Page /Parent") == 3


def test_write_helpers_create_parent_directories_for_nested_output_paths(
    tmp_path: Path,
) -> None:
    report = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "nominal",
                "stability": 0.5,
                "layers": [{"R": 0.2}],
            }
        ]
    )
    nested_dir = tmp_path / "reports" / "v1"
    markdown_path = nested_dir / "explain.md"
    pdf_path = nested_dir / "explain.pdf"

    assert write_markdown(report, markdown_path) == markdown_path
    assert write_pdf(report, pdf_path) == pdf_path
    assert markdown_path.is_file()
    assert pdf_path.is_file()
