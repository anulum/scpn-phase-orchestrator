# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Projection benchmark public and CLI contracts

"""Verify real measurements and strict JSON through benchmark entry points."""

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

from benchmarks.geometry_projection_benchmark import measure_projection


def test_actual_projection_measurement() -> None:
    """The benchmark reports the actual installed runtime and source input hash."""
    record = measure_projection(4, calls=2, repeats=2)
    raw = ((np.arange(16) % 31) / 31.0 - 0.5).reshape(4, 4)
    assert (
        record["input_sha256"]
        == hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest()
    )
    assert record["isolated"] is False
    assert ("direct_native" in record) == (
        importlib.util.find_spec("spo_kernel") is not None
    )
    encoded = json.dumps(record, allow_nan=False)
    assert json.loads(encoded)["public_python"]["median_us"] > 0
    assert len(json.loads(encoded)["public_python"]["samples_us"]) == 2


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_measurement_refusal_and_recovery(position: int, value: object) -> None:
    """Bad controls refuse before work and permit a later valid measurement."""
    values: list[object] = [4, 2, 2]
    values[position] = value
    with pytest.raises(ValueError, match="positive non-boolean integers"):
        measure_projection(*cast(tuple[int, int, int], tuple(values)))
    assert measure_projection(4, 1, 1)["n"] == 4


def test_actual_benchmark_cli() -> None:
    """A real CLI subprocess binds measured JSON to current source bytes."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.geometry_projection_benchmark",
            "--sizes",
            "4",
            "--calls",
            "2",
            "--repeats",
            "2",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    record = json.loads(result.stdout)
    for name, digest in record["source_sha256"].items():
        assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert record["rows"][0]["n"] == 4
    assert len(record["rows"][0]["public_python"]["samples_us"]) == 2
    assert ("direct_native" in record["rows"][0]) == (
        importlib.util.find_spec("spo_kernel") is not None
    )


@pytest.mark.parametrize("flag", ["--sizes", "--calls", "--repeats"])
def test_cli_refuses_zero_counts(flag: str) -> None:
    """Real malformed CLI requests return authored usage errors without a traceback."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.geometry_projection_benchmark", flag, "0"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 2
    assert "sizes and repetition counts must be positive integers" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
