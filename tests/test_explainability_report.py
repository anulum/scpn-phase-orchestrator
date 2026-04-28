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

from click.testing import CliRunner

from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.reporting.explainability import (
    build_explainability_report,
    render_markdown,
    write_markdown,
    write_pdf,
)


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


def test_explain_cli_rejects_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"event": "only"}) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["explain", str(log)])

    assert result.exit_code != 0
    assert "no step records" in result.output
