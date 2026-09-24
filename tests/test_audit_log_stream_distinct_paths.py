# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — the JSONL log and the protobuf stream never share a file

"""An audit log and its event stream must be two different files.

``spo run --audit-stream audit.jsonl`` derived the JSONL path by swapping the
suffix to ``.jsonl``, which is the stream path itself: both writers appended
to one file, the run died on a protobuf ``DecodeError`` and neither
``spo replay`` nor the stream reader could parse what was left.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.audit_stream import (
    read_event_stream,
    verify_event_stream_integrity,
)
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


def _run(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(main, ["run", _SPEC, "--steps", "5", *args])
    return result.exit_code, result.output


def test_stream_named_jsonl_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    stream = tmp_path / "audit.jsonl"
    code, output = _run("--audit-stream", str(stream))
    assert code != 0
    assert "already ends in .jsonl" in output
    assert not stream.exists()


def test_same_path_for_log_and_stream_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "audit.bin"
    code, output = _run("--audit", str(shared), "--audit-stream", str(shared))
    assert code != 0
    assert "must be different files" in output
    assert not shared.exists()


def test_logger_refuses_the_same_file_under_another_spelling(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    alias = tmp_path / "sub" / ".." / "audit.jsonl"
    (tmp_path / "sub").mkdir()
    with pytest.raises(AuditError, match="must be different files"):
        AuditLogger(log, event_stream=alias)


def test_logger_refuses_a_symlink_to_the_log(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text("", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(log)
    except OSError:
        pytest.skip("symlinks need extra privileges on this platform")
    with pytest.raises(AuditError, match="must be different files"):
        AuditLogger(log, event_stream=link)


def test_distinct_stream_suffix_writes_two_readable_files(tmp_path: Path) -> None:
    stream = tmp_path / "audit.spoa"
    code, output = _run("--audit-stream", str(stream))
    assert code == 0, output
    jsonl = tmp_path / "audit.jsonl"
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    events = read_event_stream(stream)
    assert len(records) == len(events) > 0
    assert verify_event_stream_integrity(events) == (True, len(events))
