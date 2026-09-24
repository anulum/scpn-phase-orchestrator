# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spo watch tail integrity and spo replay --output

"""``spo watch`` verifies a tail correctly; ``spo replay --output`` writes.

A tail that started mid-stream began at sequence N + 1, which the verifier
(expecting sequence 1 and the zero hash) always refused: every valid live
stream ended "stream integrity: FAILED (0 events)" with exit 1. ``--output``
on ``spo replay`` was declared and documented but never used.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.audit_stream import EventStreamWriter
from scpn_phase_orchestrator.runtime.cli import main

_SPEC = str(
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)


@pytest.fixture(autouse=True)
def _unsigned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    monkeypatch.delenv("SPO_AUDIT_KEYRING", raising=False)


def _stream(path: Path, steps: range) -> None:
    writer = EventStreamWriter(path)
    try:
        for step in steps:
            writer.write({"step": step})
    finally:
        writer.close()


def _watch_while_appending(path: Path, tail: range) -> tuple[int, str]:
    def append_later() -> None:
        time.sleep(0.5)
        _stream(path, tail)

    thread = threading.Thread(target=append_later, daemon=True)
    thread.start()
    result = CliRunner().invoke(
        main,
        ["watch", str(path), "--max-events", str(len(tail)), "--poll-interval", "0.05"],
    )
    thread.join(timeout=10)
    return result.exit_code, result.output


def test_tail_of_a_valid_stream_verifies(tmp_path: Path) -> None:
    stream = tmp_path / "live.spoa"
    _stream(stream, range(3))
    code, output = _watch_while_appending(stream, range(3, 5))
    assert code == 0, output
    assert "#4 step" in output and "#5 step" in output
    assert "stream integrity: OK (5 events)" in output


def test_tail_still_refuses_a_broken_prefix(tmp_path: Path) -> None:
    stream = tmp_path / "live.spoa"
    _stream(stream, range(3))
    raw = bytearray(stream.read_bytes())
    marker = raw.index(b'{"step":1}')
    raw[marker : marker + 10] = b'{"step":7}'
    stream.write_bytes(bytes(raw))
    # the tail refuses the tampered prefix as soon as it reads the stream
    result = CliRunner().invoke(
        main, ["watch", str(stream), "--max-events", "1", "--poll-interval", "0.05"]
    )
    assert result.exit_code != 0
    assert "stream integrity: OK" not in result.output


def test_replay_output_writes_the_summary_file(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    run = CliRunner().invoke(main, ["run", _SPEC, "--steps", "5", "--audit", str(log)])
    assert run.exit_code == 0, run.output
    summary = tmp_path / "summary.txt"
    result = CliRunner().invoke(
        main, ["replay", str(log), "--verify", "--output", str(summary)]
    )
    assert result.exit_code == 0, result.output
    text = summary.read_text(encoding="utf-8")
    assert "Steps logged: 5" in text
    assert "Determinism verified:" in text
    assert "Steps logged" not in result.output
    assert f"Replay summary written: {summary}" in result.output
