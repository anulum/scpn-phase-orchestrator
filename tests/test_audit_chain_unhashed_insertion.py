# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — unhashed records inside a hash-chained audit log

"""Prove a record without ``_hash`` cannot be slipped into a hash-chained log.

Without ``SPO_AUDIT_KEY`` the integrity check skipped every record lacking
``_hash``. A forged step appended to, or inserted into, a genuine hash-chained
log therefore verified as intact, and the explainability report presented the
forged final regime with "hash chain OK". Logs are written by the real
``AuditLogger``; the forgery is a plain JSON line, as an editor would add it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.reporting.explainability import (
    ExplainabilityReport,
    build_explainability_report,
)
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

FORGED = {
    "step": 9,
    "regime": "critical",
    "stability": 0.01,
    "layers": [{"R": 0.01, "psi": 0.0}],
    "actions": [],
}


@pytest.fixture(autouse=True)
def _unsigned_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as an unsigned deployment: no audit key in the environment."""
    for name in ("SPO_AUDIT_KEY", "SPO_AUDIT_KEYRING", "SPO_AUDIT_ENV", "SPO_ENV"):
        monkeypatch.delenv(name, raising=False)


def _state(r: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.1)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=r,
        regime_id="nominal",
    )


def _genuine_lines(path: Path) -> list[str]:
    with AuditLogger(path) as logger:
        logger.log_header(n_oscillators=2, dt=0.01)
        logger.log_step(0, _state(0.9), [])
        logger.log_step(1, _state(0.8), [])
    return path.read_text(encoding="utf-8").splitlines()


def _entries(path: Path, lines: list[str]) -> list[dict[str, object]]:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReplayEngine(path).load()


def test_genuine_chain_verifies(tmp_path: Path) -> None:
    lines = _genuine_lines(tmp_path / "genuine.jsonl")
    assert ReplayEngine.verify_integrity(_entries(tmp_path / "g.jsonl", lines)) == (
        True,
        3,
    )


@pytest.mark.parametrize("position", [3, 2, 1])
def test_unhashed_record_after_the_first_hashed_one_fails(
    tmp_path: Path, position: int
) -> None:
    lines = _genuine_lines(tmp_path / "genuine.jsonl")
    lines.insert(position, json.dumps(FORGED))
    entries = _entries(tmp_path / "tampered.jsonl", lines)

    ok, verified = ReplayEngine.verify_integrity(entries)

    assert ok is False
    assert verified == position
    assert build_explainability_report(entries).hash_chain_ok is False


def test_legacy_prefix_before_the_chain_still_verifies(tmp_path: Path) -> None:
    """The shape AuditLogger leaves when it appends to a legacy, unhashed log."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"step": 0, "layers": [{"R": 0.5}]}) + "\n")
    with AuditLogger(path) as logger:
        logger.log_step(1, _state(0.7), [])

    assert ReplayEngine.verify_integrity(ReplayEngine(path).load()) == (True, 1)


def test_fully_legacy_log_is_unverified_not_failed(tmp_path: Path) -> None:
    entries = _entries(tmp_path / "old.jsonl", [json.dumps(FORGED)])
    assert ReplayEngine.verify_integrity(entries) == (True, 0)


@pytest.mark.parametrize("junk", ["forged text", 7, ["step", 9], None])
@pytest.mark.parametrize("position", [0, 3])
def test_non_object_entry_fails_verification_instead_of_crashing(
    tmp_path: Path, junk: object, position: int
) -> None:
    """A caller-built entry list holding a non-object cannot verify as a chain.

    The file loader already refuses such lines; a caller passing parsed entries
    directly must get the same fail-closed verdict, not an AttributeError.
    """
    lines = _genuine_lines(tmp_path / "genuine.jsonl")
    entries: list[object] = list(_entries(tmp_path / "g.jsonl", lines))
    entries.insert(position, junk)
    # Alias as Callable[..., ...]: the typed signature admits only dict entries.
    verify: Callable[..., tuple[bool, int]] = ReplayEngine.verify_integrity

    assert verify(entries) == (False, position)


def test_explainability_report_skips_non_object_entries_and_flags_the_chain(
    tmp_path: Path,
) -> None:
    """Junk entries are left out of the report, which marks the chain broken."""
    lines = _genuine_lines(tmp_path / "genuine.jsonl")
    entries = _entries(tmp_path / "g.jsonl", lines)
    # Alias as Callable[..., ...]: the typed signature admits only dict entries.
    build: Callable[..., ExplainabilityReport] = build_explainability_report

    clean = build(entries)
    report = build([*entries, "forged text"])

    assert clean.hash_chain_ok is True
    assert report.hash_chain_ok is False
    assert report.steps == clean.steps
    assert report.events == clean.events
