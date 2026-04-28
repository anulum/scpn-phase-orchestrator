# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit explainability reports

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.audit.replay import ReplayEngine

__all__ = [
    "ActionExplanation",
    "ExplainabilityReport",
    "build_explainability_report",
    "render_markdown",
    "write_markdown",
    "write_pdf",
]


@dataclass(frozen=True)
class ActionExplanation:
    """Human-readable explanation for one control action."""

    step: int
    regime: str
    knob: str
    scope: str
    value: float
    ttl_s: float
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ExplainabilityReport:
    """Structured explainability summary derived from an audit JSONL file."""

    steps: int
    layers: int
    hash_chain_ok: bool
    hash_chain_verified: int
    final_regime: str
    final_stability: float
    regime_counts: dict[str, int]
    regime_transitions: tuple[str, ...]
    action_explanations: tuple[ActionExplanation, ...]
    events: tuple[str, ...]
    metric_summary: tuple[str, ...]


def _step_records(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if "step" in entry and "layers" in entry]


def _layer_rs(step: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for layer in step.get("layers", []):
        try:
            values.append(float(layer.get("R", 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _regime_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        regime = str(step.get("regime", "unknown"))
        counts[regime] = counts.get(regime, 0) + 1
    return counts


def _regime_transitions(steps: list[dict[str, Any]]) -> tuple[str, ...]:
    if not steps:
        return ()
    transitions: list[str] = []
    previous = str(steps[0].get("regime", "unknown"))
    for step in steps[1:]:
        current = str(step.get("regime", "unknown"))
        if current != previous:
            transitions.append(
                f"Step {int(step.get('step', 0))}: {previous} -> {current}"
            )
            previous = current
    return tuple(transitions)


def _event_lines(entries: list[dict[str, Any]], limit: int = 12) -> tuple[str, ...]:
    lines: list[str] = []
    for entry in entries:
        event = entry.get("event")
        if event is None:
            continue
        step = entry.get("step")
        prefix = f"Step {step}: " if step is not None else ""
        detail = entry.get("detail")
        if detail is None:
            detail = {
                key: value
                for key, value in entry.items()
                if key not in {"event", "ts", "_hash", "step"}
            }
        lines.append(f"{prefix}{event} — {detail}")
        if len(lines) >= limit:
            break
    return tuple(lines)


def _metric_summary(steps: list[dict[str, Any]]) -> tuple[str, ...]:
    if not steps:
        return ()
    n_layers = len(steps[0].get("layers", []))
    lines: list[str] = []
    for idx in range(n_layers):
        series = [
            _layer_rs(step)[idx] for step in steps if idx < len(step.get("layers", []))
        ]
        if not series:
            continue
        lines.append(
            f"Layer {idx}: mean R={_mean(series):.3f}, "
            f"final R={series[-1]:.3f}, min R={min(series):.3f}, "
            f"max R={max(series):.3f}"
        )
    stability = [float(step.get("stability", 0.0)) for step in steps]
    lines.append(
        f"Stability proxy: mean={_mean(stability):.3f}, "
        f"final={stability[-1]:.3f}, min={min(stability):.3f}, "
        f"max={max(stability):.3f}"
    )
    return tuple(lines)


def _action_explanations(
    steps: list[dict[str, Any]],
    max_actions: int,
) -> tuple[ActionExplanation, ...]:
    explanations: list[ActionExplanation] = []
    for step in steps:
        actions = step.get("actions", [])
        if not isinstance(actions, list):
            continue
        step_no = int(step.get("step", 0))
        regime = str(step.get("regime", "unknown"))
        rs = _layer_rs(step)
        stability = float(step.get("stability", 0.0))
        evidence = [
            f"mean layer R={_mean(rs):.3f}" if rs else "no layer R values recorded",
            f"stability proxy={stability:.3f}",
        ]
        for idx, value in enumerate(rs):
            evidence.append(f"L{idx} R={value:.3f}")
        for action in actions:
            if not isinstance(action, dict):
                continue
            knob = str(action.get("knob", "?"))
            scope = str(action.get("scope", "?"))
            value = float(action.get("value", 0.0))
            ttl_s = float(action.get("ttl_s", 0.0))
            justification = str(action.get("justification", "")).strip()
            reason = (
                justification
                if justification
                else (
                    f"{regime} regime with {evidence[0]} and {evidence[1]} "
                    f"triggered {knob} adjustment"
                )
            )
            explanations.append(
                ActionExplanation(
                    step=step_no,
                    regime=regime,
                    knob=knob,
                    scope=scope,
                    value=value,
                    ttl_s=ttl_s,
                    reason=reason,
                    evidence=tuple(evidence),
                )
            )
            if len(explanations) >= max_actions:
                return tuple(explanations)
    return tuple(explanations)


def build_explainability_report(
    entries: list[dict[str, Any]],
    *,
    max_actions: int = 12,
) -> ExplainabilityReport:
    """Build a structured explanation from parsed audit records."""
    steps = _step_records(entries)
    if not steps:
        raise ValueError("no step records in audit log")
    integrity_ok, n_verified = ReplayEngine.verify_integrity(entries)
    last = steps[-1]
    return ExplainabilityReport(
        steps=len(steps),
        layers=len(steps[0].get("layers", [])),
        hash_chain_ok=integrity_ok,
        hash_chain_verified=n_verified,
        final_regime=str(last.get("regime", "unknown")),
        final_stability=float(last.get("stability", 0.0)),
        regime_counts=_regime_counts(steps),
        regime_transitions=_regime_transitions(steps),
        action_explanations=_action_explanations(steps, max_actions),
        events=_event_lines(entries),
        metric_summary=_metric_summary(steps),
    )


def render_markdown(report: ExplainabilityReport) -> str:
    """Render a structured report as Markdown."""
    lines = [
        "# SCPN Phase Orchestrator Explainability Report",
        "",
        "## Summary",
        f"- Steps analysed: {report.steps}",
        f"- Layers: {report.layers}",
        f"- Final regime: {report.final_regime}",
        f"- Final stability proxy: {report.final_stability:.4f}",
        "- Hash chain: "
        f"{'OK' if report.hash_chain_ok else 'FAILED'} "
        f"({report.hash_chain_verified} records verified)",
        "",
        "## Metric Evidence",
    ]
    lines.extend(f"- {item}" for item in report.metric_summary)
    lines.extend(["", "## Regime Distribution"])
    for regime, count in sorted(report.regime_counts.items()):
        pct = 100.0 * count / report.steps
        lines.append(f"- {regime}: {count} steps ({pct:.1f}%)")
    lines.extend(["", "## Regime Transitions"])
    if report.regime_transitions:
        lines.extend(f"- {transition}" for transition in report.regime_transitions)
    else:
        lines.append("- No regime transitions recorded.")
    lines.extend(["", "## Control Action Explanations"])
    if report.action_explanations:
        for explanation in report.action_explanations:
            lines.append(
                f"- Step {explanation.step}: {explanation.knob}="
                f"{explanation.value:.4f} ({explanation.scope}, "
                f"ttl={explanation.ttl_s:.1f}s) in {explanation.regime}. "
                f"Reason: {explanation.reason}. Evidence: "
                f"{'; '.join(explanation.evidence)}."
            )
    else:
        lines.append("- No control actions recorded.")
    lines.extend(["", "## Events"])
    if report.events:
        lines.extend(f"- {event}" for event in report.events)
    else:
        lines.append("- No auxiliary events recorded.")
    return "\n".join(lines) + "\n"


def write_markdown(report: ExplainabilityReport, output_path: str | Path) -> Path:
    """Write Markdown report and return the output path."""
    out = Path(output_path)
    out.write_text(render_markdown(report), encoding="utf-8")
    return out


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_lines(markdown: str, width: int = 92) -> list[str]:
    lines: list[str] = []
    for raw in markdown.splitlines():
        text = raw.strip()
        if not text:
            lines.append("")
            continue
        if text.startswith("#"):
            text = text.lstrip("#").strip().upper()
        wrapped = textwrap.wrap(text, width=width) or [""]
        lines.extend(wrapped)
    return lines


def _make_pdf_bytes(lines: list[str]) -> bytes:
    objects: list[bytes] = []
    page_ids: list[int] = []
    lines_per_page = 52
    pages = [
        lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)
    ]
    if not pages:
        pages = [[]]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    next_id = 4
    for page_lines in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode()
        )
        stream_lines = ["BT", "/F1 10 Tf", "14 TL", "50 800 Td"]
        for line in page_lines:
            stream_lines.append(f"({_pdf_escape(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode()
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(pdf)


def write_pdf(report: ExplainabilityReport, output_path: str | Path) -> Path:
    """Write a dependency-free text PDF report and return the output path."""
    out = Path(output_path)
    lines = _wrap_pdf_lines(render_markdown(report))
    out.write_bytes(_make_pdf_bytes(lines))
    return out
