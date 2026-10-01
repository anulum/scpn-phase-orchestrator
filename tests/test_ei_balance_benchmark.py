# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real E/I diagnostic CLI contracts

"""Exercise the actual benchmark callable and separately launched CLI."""

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

from benchmarks.ei_balance_benchmark import (
    benchmark_ei_balance,
    native_binary_provenance,
    validate_ei_measurement,
)
from scpn_phase_orchestrator.coupling import adjust_ei_ratio, compute_ei_balance

ROOT = Path(__file__).resolve().parents[1]


def test_real_standalone_extension_provenance() -> None:
    """Hash NumPy's actual standalone extension file without claiming EI use."""
    spec = importlib.util.find_spec("numpy.linalg._umath_linalg")
    assert spec is not None and spec.origin is not None
    binary, digest = native_binary_provenance(spec.name)
    assert binary == spec.origin
    assert digest == hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "module_name", ["scpn_phase_orchestrator", "benchmarks.ei_balance_benchmark", "sys"]
)
def test_real_source_layout_refuses_native_attribution(module_name: str) -> None:
    """Refuse real source and built-in layouts as disk extension binaries."""
    with pytest.raises(RuntimeError, match="no resolvable native binary"):
        native_binary_provenance(module_name)


@pytest.mark.parametrize(
    "mismatch",
    ["summary_source", "summary_inhibitory_source", "target", "adjustment_source"],
)
def test_real_measurement_source_and_target_mismatch_refuses(mismatch: str) -> None:
    """Reject genuine successful outputs joined to the wrong input or target."""
    matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
    source = matrix.copy()
    exc, inh = [0], [1]
    summary_input = matrix.copy()
    if mismatch == "summary_source":
        summary_input *= 2.0
    elif mismatch == "summary_inhibitory_source":
        summary_input[inh] *= 2.0
    summary = compute_ei_balance(summary_input, exc, inh)
    adjusted = adjust_ei_ratio(
        matrix * 2.0 if mismatch == "adjustment_source" else matrix,
        exc,
        inh,
        2.0 if mismatch == "target" else 1.5,
    )
    original_adjusted = adjusted.copy()
    message = (
        "public E/I means"
        if mismatch.startswith("summary")
        else "declared target"
        if mismatch == "target"
        else "Not equal to tolerance"
    )
    with pytest.raises(AssertionError, match=message):
        validate_ei_measurement(matrix, exc, inh, summary, adjusted, 1.5)
    np.testing.assert_array_equal(matrix, source)
    np.testing.assert_array_equal(adjusted, original_adjusted)
    validate_ei_measurement(
        matrix,
        exc,
        inh,
        compute_ei_balance(matrix, exc, inh),
        adjust_ei_ratio(matrix, exc, inh, 1.5),
        1.5,
    )


def test_real_runtime_diagnostics_preserve_values_and_provenance() -> None:
    """Use actual dense matrices and public calls with retained timing samples."""
    result = benchmark_ei_balance([3, 6], calls=2, repeats=2)
    assert result["actual_native"] is (
        importlib.util.find_spec("spo_kernel") is not None
    )
    assert result["production_performance_claim"] is False
    assert result["speedup_claim"] is False
    rows = cast(list[dict[str, object]], result["rows"])
    assert [(r["n"], r["operation"]) for r in rows] == [
        (3, "compute"),
        (3, "adjust"),
        (6, "compute"),
        (6, "adjust"),
    ]
    for row in rows:
        samples = cast(list[float], row["ns_per_call"])
        assert len(samples) == 2
        assert all(v > 0 for v in samples)
        assert row["adjusted_ratio"] == pytest.approx(1.5)
        assert len(cast(str, row["input_sha256"])) == 64
    assert rows[0]["input_sha256"] == rows[1]["input_sha256"]
    assert rows[0]["summary"] == rows[1]["summary"]


@pytest.mark.parametrize(
    "sizes,calls,repeats",
    [
        ([], 1, 1),
        ([1], 1, 1),
        ([True], 1, 1),
        ([2], 0, 1),
        ([2], True, 1),
        ([2], 1, 0),
        ([2], 1, True),
        ([cast(int, 2.5)], 1, 1),
        ([2], cast(int, 1.5), 1),
        ([2], 1, cast(int, 1.5)),
    ],
)
def test_bad_measurement_configuration_refuses(
    sizes: list[int],
    calls: int,
    repeats: int,
) -> None:
    """Reject invalid measurement dimensions before starting any timing loop."""
    with pytest.raises(ValueError, match="sizes must be integers"):
        benchmark_ei_balance(sizes, calls=calls, repeats=repeats)


@pytest.mark.parametrize("valid", [False, True])
def test_actual_cli_reports_measurements_or_refuses(valid: bool) -> None:
    """Launch the maintained CLI and observe its real output and exit status."""
    command = [
        sys.executable,
        str(ROOT / "benchmarks/ei_balance_benchmark.py"),
        "--sizes",
        "4" if valid else "0",
        "--calls",
        "2",
        "--repeats",
        "2",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    if valid:
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["command"] == command
        assert result["actual_native"] is (
            importlib.util.find_spec("spo_kernel") is not None
        )
        assert [row["operation"] for row in result["rows"]] == ["compute", "adjust"]
        assert all(
            row["adjusted_ratio"] == pytest.approx(1.5) for row in result["rows"]
        )
    else:
        assert completed.returncode != 0
        assert "sizes must be integers" in completed.stderr
        assert completed.stdout == ""
