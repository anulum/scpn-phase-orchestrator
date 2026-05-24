# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — deterministic replay tests

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.runtime.replay import ReplayEngine


def test_replay_load_rejects_non_finite_json_constants(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"event":"operator_note","value":NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="finite JSON"):
        ReplayEngine(path).load()


def test_replay_integrity_rejects_non_finite_hashed_entries() -> None:
    entry: dict[str, object] = {"event": "operator_note", "value": float("nan")}
    json_line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
    entry["_hash"] = hashlib.sha256((("0" * 64) + json_line).encode()).hexdigest()

    assert ReplayEngine.verify_integrity([entry]) == (False, 0)
