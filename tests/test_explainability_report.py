# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Explainability report tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.reporting.explainability as explainability
from scpn_phase_orchestrator.reporting.explainability import (
    _make_pdf_bytes,
    _metric_summary,
    _regime_transitions,
    build_explainability_report,
    render_markdown,
    write_markdown,
    write_pdf,
)
from scpn_phase_orchestrator.runtime.cli import main


def _entries() -> list[dict[str, Any]]:
    return [
        {
            "step": 0,
            "regime": "NOMINAL",
            "stability": 0.91,
            "layers": [{"R": 0.82, "psi": 0.0}, {"R": 0.18, "psi": 0.0}],
            "actions": [],
        },
        {
            "step": 1,
            "regime": "CRITICAL",
            "stability": 0.21,
            "layers": [{"R": 0.34, "psi": 0.0}, {"R": 0.91, "psi": 0.0}],
            "actions": [
                {
                    "knob": "K",
                    "scope": "global",
                    "value": -0.1,
                    "ttl_s": 3.0,
                    "justification": "R_bad suppressed after Lyapunov spike",
                }
            ],
        },
        {
            "event": "boundary_violation",
            "step": 1,
            "detail": {"boundary": "r_bad_critical"},
        },
    ]


def _write_log(path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(entry) for entry in _entries()) + "\n",
        encoding="utf-8",
    )


def test_build_explainability_report_extracts_action_reason() -> None:
    report = build_explainability_report(_entries())

    assert report.steps == 2
    assert report.final_regime == "CRITICAL"
    assert report.regime_transitions == ("Step 1: NOMINAL -> CRITICAL",)
    assert len(report.action_explanations) == 1
    explanation = report.action_explanations[0]
    assert explanation.step == 1
    assert explanation.knob == "K"
    assert "Lyapunov spike" in explanation.reason
    assert any(item.startswith("L1 R=0.910") for item in explanation.evidence)


def test_render_markdown_contains_human_explanation() -> None:
    report = build_explainability_report(_entries())
    markdown = render_markdown(report)

    assert "Explainability Report" in markdown
    assert "Control Action Explanations" in markdown
    assert "R_bad suppressed" in markdown
    assert "boundary_violation" in markdown


def test_write_markdown_and_pdf_outputs(tmp_path: Path) -> None:
    report = build_explainability_report(_entries())
    md_path = tmp_path / "explain.md"
    pdf_path = tmp_path / "explain.pdf"

    assert write_markdown(report, md_path) == md_path
    assert write_pdf(report, pdf_path) == pdf_path

    assert md_path.read_text(encoding="utf-8").startswith("# SCPN")
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes


def test_explain_cli_prints_markdown(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    _write_log(log)

    result = CliRunner().invoke(main, ["explain", str(log)])

    assert result.exit_code == 0
    assert "Explainability Report" in result.output
    assert "R_bad suppressed" in result.output


def test_explain_cli_writes_files(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    md_path = tmp_path / "out.md"
    pdf_path = tmp_path / "out.pdf"
    _write_log(log)

    result = CliRunner().invoke(
        main,
        [
            "explain",
            str(log),
            "--markdown-out",
            str(md_path),
            "--pdf-out",
            str(pdf_path),
        ],
    )

    assert result.exit_code == 0
    assert "Markdown report written" in result.output
    assert "PDF report written" in result.output
    assert "Control Action Explanations" in md_path.read_text(encoding="utf-8")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_explain_cli_rejects_missing_log_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(main, ["explain", str(missing)])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_explain_cli_rejects_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"event": "only"}) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["explain", str(log)])

    assert result.exit_code != 0
    assert "no step records" in result.output


def test_report_marks_hash_chain_failure_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explainability.ReplayEngine,
        "verify_integrity",
        lambda entries: (False, 0),
    )

    report_with_failed_chain = build_explainability_report(
        [
            {
                "step": 0,
                "regime": "NOMINAL",
                "stability": 0.91,
                "layers": [{"R": 0.82}, {"R": 0.18}],
            }
        ]
    )
    markdown = render_markdown(report_with_failed_chain)

    assert report_with_failed_chain.hash_chain_ok is False
    assert report_with_failed_chain.hash_chain_verified == 0
    assert "- Hash chain: FAILED (0 records verified)" in markdown


def test_report_summarises_causal_transfer_hodge_and_safety_channels() -> None:
    entries = [
        {
            "step": 0,
            "regime": "NOMINAL",
            "stability": 0.74,
            "layers": [{"R": "not-a-number"}, {"R": 0.62}],
            "actions": [
                {
                    "knob": "K_nm",
                    "scope": "layer:0",
                    "value": 0.035,
                    "ttl_s": 1.5,
                    "justification": "causal parent K_nm selected after TE=0.42",
                },
                "malformed action is ignored",
            ],
        },
        {
            "step": 1,
            "regime": "NOMINAL",
            "stability": 0.69,
            "layers": [{"R": 0.58}, {"R": 0.61}],
            "actions": [
                {
                    "knob": "hodge_projection",
                    "scope": "cycle:3",
                    "value": -0.2,
                    "ttl_s": 2.0,
                    "justification": "",
                }
            ],
        },
        {
            "event": "safety_residual",
            "step": 1,
            "channel": "residual_norm",
            "value": 0.018,
        },
        {
            "event": "hodge_summary",
            "detail": {"gradient": 0.71, "curl": 0.08, "harmonic": 0.02},
        },
    ]

    report = build_explainability_report(entries)
    markdown = render_markdown(report)

    assert report.regime_transitions == ()
    assert report.metric_summary[0] == (
        "Layer 0: mean R=0.290, final R=0.580, min R=0.000, max R=0.580"
    )
    assert report.action_explanations[0].reason == (
        "causal parent K_nm selected after TE=0.42"
    )
    assert report.action_explanations[0].evidence == (
        "mean layer R=0.310",
        "stability proxy=0.740",
        "L0 R=0.000",
        "L1 R=0.620",
    )
    assert "hodge_projection=-0.2000" in markdown
    assert "NOMINAL regime with mean layer R=0.595" in markdown
    assert "safety_residual" in markdown
    assert "{'channel': 'residual_norm', 'value': 0.018}" in markdown
    assert "hodge_summary" in markdown
    assert "causal parent K_nm" in markdown


def test_report_handles_missing_actions_events_and_transitions() -> None:
    report = build_explainability_report(
        [
            {
                "step": 7,
                "regime": "STABLE",
                "stability": 1.0,
                "layers": [{"R": 0.4}],
                "actions": "not a list",
            }
        ]
    )

    markdown = render_markdown(report)

    assert report.action_explanations == ()
    assert report.events == ()
    assert report.regime_transitions == ()
    assert "- STABLE: 1 steps (100.0%)" in markdown
    assert "- No regime transitions recorded." in markdown
    assert "- No control actions recorded." in markdown
    assert "- No auxiliary events recorded." in markdown


def test_action_explanation_limit_preserves_first_actions_in_order() -> None:
    entries = [
        {
            "step": idx,
            "regime": "WATCH" if idx < 3 else "INTERVENE",
            "stability": 0.5 + idx / 100.0,
            "layers": [{"R": 0.2 + idx / 100.0}],
            "actions": [
                {
                    "knob": "gain",
                    "scope": f"layer:{idx}",
                    "value": float(idx),
                    "ttl_s": 1.0,
                    "justification": f"ranked action {idx}",
                }
            ],
        }
        for idx in range(5)
    ]

    report = build_explainability_report(entries, max_actions=3)

    assert [action.step for action in report.action_explanations] == [0, 1, 2]
    assert [action.reason for action in report.action_explanations] == [
        "ranked action 0",
        "ranked action 1",
        "ranked action 2",
    ]
    assert report.regime_transitions == ("Step 3: WATCH -> INTERVENE",)


def test_event_summary_limits_auxiliary_audit_lines() -> None:
    entries: list[dict[str, Any]] = [
        {
            "step": 0,
            "regime": "NOMINAL",
            "stability": 0.8,
            "layers": [{"R": 0.8}],
            "actions": [],
        }
    ]
    entries.extend({"event": f"audit_{idx}", "step": idx} for idx in range(15))

    report = build_explainability_report(entries)

    assert len(report.events) == 12
    assert report.events[0] == "Step 0: audit_0 — {}"
    assert report.events[-1] == "Step 11: audit_11 — {}"
    assert all("audit_12" not in event for event in report.events)


def test_private_aggregation_helpers_define_empty_input_contracts() -> None:
    assert _regime_transitions([]) == ()
    assert _metric_summary([]) == ()

    pdf = _make_pdf_bytes([])

    assert pdf.startswith(b"%PDF-1.4")
    assert b"/Type /Pages" in pdf
    assert b"/Count 1" in pdf
    assert pdf.endswith(b"%%EOF\n")
