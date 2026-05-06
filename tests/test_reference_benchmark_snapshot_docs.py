# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference benchmark snapshot docs regression

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SNAPSHOT_JSON = Path("benchmarks/results/reference_suite.json")
SNAPSHOT_DOC = Path("docs/galleries/reference_benchmark_snapshot.md")


def test_reference_benchmark_snapshot_doc_matches_json_metadata() -> None:
    snapshot = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    metadata: dict[str, str] = snapshot["metadata"]
    doc = SNAPSHOT_DOC.read_text(encoding="utf-8")

    for key in (
        "snapshot_date",
        "suite_version",
        "command",
        "backend",
        "python_version",
        "numpy_version",
        "platform",
        "executable",
    ):
        assert metadata[key] in doc


def test_reference_benchmark_snapshot_doc_lists_all_benchmark_records() -> None:
    snapshot = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    benchmarks: dict[str, dict[str, Any]] = snapshot["benchmarks"]
    doc = SNAPSHOT_DOC.read_text(encoding="utf-8")

    for record in benchmarks.values():
        assert str(record["suite"]) in doc
        assert str(record["n_steps"]) in doc
        assert str(record["wall_time_s"]) in doc
        assert str(record["steps_per_second"]) in doc
