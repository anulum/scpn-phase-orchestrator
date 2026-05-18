# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reporting subsystem

"""Public reporting helpers for audit-derived operator artefacts.

The reporting package converts parsed audit records into explainability
summaries, markdown/PDF reports, compact JSON-safe summaries, and optional
diagnostic plots. It reads supplied records only; report generation does not
rerun controllers or perform live actuation.
"""

from __future__ import annotations

from scpn_phase_orchestrator.reporting.explainability import (
    ActionExplanation,
    ExplainabilityReport,
    build_explainability_report,
    render_markdown,
    write_markdown,
    write_pdf,
)
from scpn_phase_orchestrator.reporting.plots import CoherencePlot
from scpn_phase_orchestrator.reporting.summary import build_audit_report_summary

__all__ = [
    "ActionExplanation",
    "CoherencePlot",
    "ExplainabilityReport",
    "build_audit_report_summary",
    "build_explainability_report",
    "render_markdown",
    "write_markdown",
    "write_pdf",
]
