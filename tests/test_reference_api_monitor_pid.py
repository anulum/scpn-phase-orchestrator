# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PID API reference documentation tests

"""Regression checks for the PID monitor reference page."""

from __future__ import annotations

from pathlib import Path

PID_REFERENCE = Path("docs/reference/api/monitor_pid.md")
TE_REFERENCE = Path("docs/reference/api/monitor_transfer_entropy.md")


def test_pid_api_reference_meets_depth_baseline() -> None:
    """The PID reference page keeps enough explanatory material to be useful."""
    doc = PID_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 140


def test_pid_api_reference_documents_math_and_polyglot_gate() -> None:
    """The PID page documents the estimator and release parity gate."""
    doc = PID_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "Williams-Beer",
        "I_min",
        "I_red",
        "I_syn",
        "redundancy",
        "synergy",
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
        "benchmark_pid_polyglot_parity_gate",
        "pid_polyglot",
        "local, non-isolated regression evidence",
        "scpn_phase_orchestrator.monitor.pid",
    )

    for phrase in required_phrases:
        assert phrase in doc


def test_transfer_entropy_reference_no_longer_marks_pid_broken() -> None:
    """Transfer-entropy docs must not retain the stale PID rewrite warning."""
    doc = TE_REFERENCE.read_text(encoding="utf-8")

    assert "algorithmically broken" not in doc
    assert "pending the PID rewrite" not in doc
    assert "pid_polyglot" in doc
