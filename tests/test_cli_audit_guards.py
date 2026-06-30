# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI replay/watch failure-path guard tests

"""Failure-path coverage for the ``spo replay`` and ``spo watch`` audit commands.

These tests drive the remaining uncovered branches of
:mod:`scpn_phase_orchestrator.runtime.cli.audit`: the replay determinism-FAILED
exit, the ``_watch_line`` fallback for non-step/non-header events, the
``--max-events`` guard, the bounded ``--from-start`` read path, and the
watch stream-integrity-FAILED exit. The determinism and integrity verifiers are
monkeypatched to their failing result so the exit paths run deterministically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.audit as cli_audit
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _state() -> UPDEState:
    """Return a minimal nominal UPDE state for a single layer."""
    return UPDEState(
        layers=[LayerState(R=0.82, psi=0.5)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.82,
        regime_id="nominal",
    )


def _write_step(logger: AuditLogger, step: int) -> None:
    """Append a deterministic two-oscillator step to an audit logger."""
    logger.log_step(
        step,
        _state(),
        [],
        phases=np.array([0.1, 0.2]),
        omegas=np.array([1.0, 1.0]),
        knm=np.array([[0.0, 0.3], [0.3, 0.0]]),
        alpha=np.zeros((2, 2)),
    )


def _write_replay_log(path: Path) -> None:
    """Write a valid, integrity-clean Kuramoto audit log with a header and a step."""
    with AuditLogger(str(path)) as logger:
        logger.log_header(n_oscillators=2, dt=0.01, seed=42)
        _write_step(logger, 0)


def _write_event_stream(path: Path, *, with_note: bool = False) -> Path:
    """Write a protobuf event stream; optionally append a non-step/header event."""
    stream_path = path.with_suffix(".spoa")
    with AuditLogger(str(path), event_stream=str(stream_path)) as logger:
        logger.log_header(n_oscillators=2, dt=0.01)
        _write_step(logger, 0)
        if with_note:
            logger.log_event("operator_note", {"step": 7, "note": "manual hold"})
    return stream_path


def test_replay_verify_reports_determinism_failure(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "audit.jsonl"
    _write_replay_log(log)
    monkeypatch.setattr(
        ReplayEngine,
        "verify_determinism_chained",
        lambda self, engine, entries: (False, 3),
    )

    result = runner.invoke(main, ["replay", str(log), "--verify"])

    assert result.exit_code == 1
    assert "Determinism FAILED at transition 3" in result.output


def test_watch_from_start_reads_bounded_stream_and_renders_event(
    runner: CliRunner, tmp_path: Path
) -> None:
    stream = _write_event_stream(tmp_path / "audit.jsonl", with_note=True)

    result = runner.invoke(main, ["watch", str(stream), "--from-start"])

    assert result.exit_code == 0, result.output
    # The bounded --from-start read path renders every event, including the
    # non-step/non-header operator_note via the _watch_line fallback.
    assert "operator_note step=7" in result.output
    assert "stream integrity: OK" in result.output


def test_watch_rejects_non_positive_max_events(
    runner: CliRunner, tmp_path: Path
) -> None:
    stream = _write_event_stream(tmp_path / "audit.jsonl")

    result = runner.invoke(main, ["watch", str(stream), "--max-events", "0"])

    assert result.exit_code == 1
    assert "--max-events must be >= 1" in result.output


def test_watch_reports_stream_integrity_failure(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _write_event_stream(tmp_path / "audit.jsonl")
    monkeypatch.setattr(
        cli_audit, "verify_event_stream_integrity", lambda events: (False, 2)
    )

    result = runner.invoke(main, ["watch", str(stream), "--from-start"])

    assert result.exit_code == 1
    assert "stream integrity: FAILED (2 events)" in result.output


def test_watch_line_renders_plain_event_without_step() -> None:
    # A non-step/non-header event with no integer step renders no step suffix —
    # the second outcome of the _watch_line fallback ternary.
    class _Event:
        sequence = 4
        event_type = "operator_note"
        event_hash = "abcdef0123456789"
        payload: dict[str, object] = {"note": "no step here"}

    line = cli_audit._watch_line(_Event())  # type: ignore[arg-type]

    assert line == "#4 operator_note hash=abcdef012345"
