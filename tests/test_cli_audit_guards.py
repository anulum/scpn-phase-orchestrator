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
exit, the replay hash-chain-integrity-FAILED exit (driven by a genuinely
tampered log, not a stub), the no-step replay summary, the ``_watch_line``
fallback for non-step/non-header events, the ``--max-events`` guard, the
bounded ``--from-start`` read path, the live-iterator path bounded by
``--max-events``, and the watch stream-integrity-FAILED exit. The determinism
and stream-integrity verifiers are monkeypatched to their failing result where
noted; the replay hash-chain check runs against real tampered bytes.
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


def test_replay_verify_reports_integrity_failure_on_tampered_log(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A genuinely tampered record must fail `--verify` before determinism runs.

    The step record's ``stability`` value is edited in place WITHOUT
    recomputing the record hash, so the real hash-chain verifier (no stub)
    must report the break and exit non-zero.
    """
    log = tmp_path / "audit.jsonl"
    _write_replay_log(log)
    lines = log.read_text(encoding="utf-8").splitlines()
    tampered = [
        (
            line.replace('"stability": 0.82', '"stability": 0.99')
            if '"step"' in line
            else line
        )
        for line in lines
    ]
    assert tampered != lines, "fixture must contain a tamperable step record"
    log.write_text("\n".join(tampered) + "\n", encoding="utf-8")

    result = runner.invoke(main, ["replay", str(log), "--verify"])

    assert result.exit_code == 1
    assert "audit integrity FAILED" in result.output


def test_replay_summarises_event_only_log_without_final_state_lines(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A log with no step records reports zero steps and omits final-state lines."""
    log = tmp_path / "audit.jsonl"
    with AuditLogger(str(log)) as logger:
        logger.log_header(n_oscillators=2, dt=0.01, seed=42)
        logger.log_event("operator_note", {"note": "no steps recorded"})

    result = runner.invoke(main, ["replay", str(log)])

    assert result.exit_code == 0, result.output
    assert "Steps logged: 0" in result.output
    assert "Events logged: 1" in result.output
    assert "Final regime" not in result.output


def test_watch_max_events_bounds_the_live_iterator_path(
    runner: CliRunner, tmp_path: Path
) -> None:
    """`--from-start --max-events N` takes the live-iterator path and stops at N.

    With ``--max-events`` set the command must use the tailing iterator (not
    the bounded one-shot read), echo exactly N events, and verify integrity
    over that bounded prefix.
    """
    stream = _write_event_stream(tmp_path / "audit.jsonl", with_note=True)

    result = runner.invoke(
        main,
        [
            "watch",
            str(stream),
            "--from-start",
            "--max-events",
            "2",
            "--poll-interval",
            "0.01",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stream integrity: OK (2 events)" in result.output
    # The third event (operator_note) is beyond the bound and must not render.
    assert "operator_note" not in result.output


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
