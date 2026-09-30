# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic diagnostic integration tests

"""Validate real public symbolic diagnostics and their command-line publication."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.bench_symbolic import benchmark_symbolic_extraction


def test_real_symbolic_diagnostic_outputs_and_dispatch() -> None:
    """Both real environments report correct phases and actual per-call owners."""
    report = benchmark_symbolic_extraction(repeats=2)
    assert (
        report["classification"]
        == "non-isolated functional and local timing diagnostic"
    )
    present = importlib.util.find_spec("spo_kernel") is not None
    assert report["kernel_present"] is present
    if present:
        import spo_kernel.spo_kernel as native_module

        assert native_module.__file__ is not None
        artifact = Path(native_module.__file__)
        assert report["native_artifact"] == {
            "filename": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    else:
        assert report["native_artifact"] is None
    rows = {row["name"]: row for row in report["cases"]}
    assert set(rows) == {
        "ring_1000",
        "graph_1000",
        "signed_full_span",
        "unsigned_full_span",
        "strided_graph",
        "vocabulary_beyond_native_capacity",
    }
    for name in ("signed_full_span", "unsigned_full_span"):
        row = rows[name]
        assert row["sampled_theta"] == pytest.approx([0.0, np.pi, 0.0])
        assert row["sampled_quality"] == pytest.approx([0.5, 0.1, 0.1])
    assert rows["strided_graph"]["sampled_theta"] == pytest.approx(
        [0.0, 10 * np.pi / 9, 0.0]
    )
    oversized = rows["vocabulary_beyond_native_capacity"]
    assert oversized["observed_backend"] == "python"
    assert oversized["native_calls"] == []
    expected_middle = 2 * np.pi / (int(np.iinfo(np.uintp).max) + 1)
    assert oversized["sampled_theta"][1] == pytest.approx(expected_middle, abs=0.0)
    for row in report["cases"]:
        assert 0.0 <= row["median_seconds"] <= row["p95_seconds"]
        if row is not oversized:
            assert row["observed_backend"] == ("native" if present else "python")
            expected_calls = (
                ["ring_phases_rust"]
                if row["mode"] == "ring"
                else ["graph_walk_phases_rust", "transition_qualities_rust"]
            )
            assert row["native_calls"] == (expected_calls if present else [])
    repository = Path(__file__).resolve().parents[1]
    for source, digest in report["source_sha256"].items():
        assert hashlib.sha256((repository / source).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("repeats", [True, 0, -1, 1.5, "2"])
def test_symbolic_diagnostic_rejects_invalid_repeat_counts(repeats: object) -> None:
    """Invalid timing counts refuse through the public diagnostic API."""
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_symbolic_extraction(repeats=cast(int, repeats))


@pytest.mark.parametrize("repeats", ["2", "0"])
def test_symbolic_diagnostic_cli(repeats: str) -> None:
    """The actual script emits finite JSON or a structured argparse refusal."""
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "benchmarks/bench_symbolic.py"),
            "--repeats",
            repeats,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if repeats == "0":
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "repeats must be a positive integer" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["format_version"] == 1
        assert report["repeats"] == 2
        assert len(report["cases"]) == 6
        assert (
            report["classification"]
            == "non-isolated functional and local timing diagnostic"
        )
