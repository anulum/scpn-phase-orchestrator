# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Executed phase benchmark CLI contract

"""Exercise the actual comparison CLI with installed compiled and JAX runtimes.

The explicit WebGPU redirect inside the CPU comparator requires WebGPU to be in
its actual available-backend list. No host dispatch adapter is configured
in this CPython runtime; the separate browser CLI executes its real counterpart.
No availability flag or producer is replaced. POSIX stdout recovery is exercised
through captured CLI output; the Windows branch requires a Windows host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.upde_phase_wrapping_benchmark import benchmark_phase_wrapping

pytestmark = [pytest.mark.native_runtime, pytest.mark.performance]


def test_actual_cli_can_write_to_a_memory_text_sink() -> None:
    """Embed the real module CLI using Python's standard text-stream adapter."""
    script = """
import contextlib
import io
import json
import os
import runpy
import sys

output = io.StringIO()
with contextlib.redirect_stdout(output):
    try:
        runpy.run_module(
            'benchmarks.upde_phase_wrapping_benchmark', run_name='__main__'
        )
    except SystemExit as exc:
        assert exc.code == 0
payload = json.loads(output.getvalue())
if os.name == 'posix':
    os.set_blocking(sys.stdout.fileno(), True)
print(json.dumps(payload))
"""
    execution = subprocess.run(
        [sys.executable, "-c", script, "--calls", "1", "--backends", "python"],
        check=False,
        shell=False,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    payload = cast("dict[str, object]", json.loads(execution.stdout))
    rows = cast("list[dict[str, object]]", payload["measurements"])
    assert len(rows) == 15
    assert {row["backend"] for row in rows} == {
        "python",
        "jax",
        "nn-dense",
        "nn-masked",
    }
    assert all(row["canonical_torus_passed"] is True for row in rows)


@pytest.mark.parametrize("invalid", ["count", "backend"])
def test_invalid_measurement_request_is_refused(invalid: str) -> None:
    """Refuse measurements that cannot execute a real selected consumer.

    Parameters
    ----------
    invalid : str
        Select zero repetitions or an unrecognised backend request.
    """
    with pytest.raises(ValueError):
        if invalid == "count":
            benchmark_phase_wrapping(calls=0, backends=["python"])
        else:
            benchmark_phase_wrapping(calls=1, backends=["unknown-backend"])


def test_actual_comparison_cli_writes_canonical_outputs(tmp_path: Path) -> None:
    """Run every actual backend and inspect the resulting numerical artefact.

    Parameters
    ----------
    tmp_path : Path
        Owned directory for the actual CLI JSON artefact.
    """
    destination = tmp_path / "phase-comparison.json"
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    execution = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.upde_phase_wrapping_benchmark",
            "--calls",
            "1",
            "--backends",
            "rust",
            "mojo",
            "julia",
            "go",
            "python",
            "--output",
            str(destination),
        ],
        check=False,
        shell=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    payload = cast("dict[str, object]", json.loads(destination.read_text()))
    assert json.loads(execution.stdout) == payload
    rows = cast("list[dict[str, object]]", payload["measurements"])
    observed = {str(row["backend"]) for row in rows}
    assert observed == {
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
        "jax",
        "nn-dense",
        "nn-masked",
    }
    for row in rows:
        dtype = np.float32 if row["precision"] == "float32" else np.float64
        output: NDArray[np.float32 | np.float64] = np.asarray(
            row["output"], dtype=dtype
        )
        period = dtype(2.0 * np.pi)
        np.testing.assert_array_equal(output[:5], np.zeros(5))
        assert not np.any(np.signbit(output[:5]))
        assert output[5] == np.nextafter(period, dtype(0.0))
        assert abs(float(output[6]) - 0.27) <= (3e-8 if dtype == np.float32 else 2e-16)
        assert np.all(np.isfinite(output) & (output >= 0.0) & (output < period))
