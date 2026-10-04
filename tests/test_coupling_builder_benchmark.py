# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Construction measurement and CLI contracts

"""Verify actual construction measurements, runtime attribution and CLI output."""

from __future__ import annotations

import cProfile
import hashlib
import importlib.util
import json
import profile
import struct
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from benchmarks.coupling_builder_benchmark import measure_construction


@pytest.mark.parametrize("n", [4, 16])
def test_actual_construction_measurement(n: int) -> None:
    """Same scalar bits and actual backend observations accompany measured samples.

    Parameters
    ----------
    n : int
        Positive matrix dimension; sixteen also measures the SCPN model.

    Notes
    -----
    CPython suspends tracing inside sys.setprofile callbacks; see
    https://docs.python.org/3.12/library/sys.html#sys.call_tracing
    Thus statements inside the benchmark's observe_native_build remain untraced
    in the recorded CPython 3.11-3.13 coverage runs. This public test covers
    the nearest real behaviour by
    checking actual compiled calls in native and genuinely absent interpreters.
    The caller-profiler test below also exercises public success, refusal and
    recovery under both standard profiling implementations. The exact reporting
    limit before portable host metadata was 108/110 statements and 26/28
    branches, with zero exclusions. Current OS metadata also requires genuine
    platform qualification; those historical counts and executed callbacks do
    not establish a current unqualified 100% coverage claim.
    """
    record = json.loads(json.dumps(measure_construction(n, 2, 2), allow_nan=False))
    assert record["input_hex"] == struct.pack("<Qdd", n, 0.45, 0.3).hex()
    assert record["isolated"] is False
    assert (record["native_build_calls_observed"] > 0) == (
        importlib.util.find_spec("spo_kernel") is not None
    )
    assert ("scpn_physics" in record) == (n == 16)
    for key in ("generic", "amplitude"):
        assert len(record[key]["samples_us"]) == 2
        assert record[key]["median_us"] > 0


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_bad_measurement_controls_refuse_and_recover(
    position: int, value: object
) -> None:
    """Invalid original controls cannot start work and a valid request recovers.

    Parameters
    ----------
    position : int
        Index of the invalid dimension, call-count or repeat-count control.
    value : object
        Original case value supplied unchanged at the public boundary.
    """
    values: list[object] = [4, 2, 2]
    values[position] = value
    with pytest.raises(ValueError, match="positive non-boolean integers"):
        measure_construction(*cast(tuple[int, int, int], tuple(values)))
    assert measure_construction(4, 1, 1)["n"] == 4


def test_actual_construction_benchmark_cli() -> None:
    """The real CLI subprocess reports measurements bound to current source bytes."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.coupling_builder_benchmark",
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
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert record["rows"][0]["n"] == 4
    assert len(record["rows"][0]["generic"]["samples_us"]) == 2


@pytest.mark.parametrize("flag", ["--sizes", "--calls", "--repeats"])
def test_actual_cli_refuses_zero_controls(flag: str) -> None:
    """Malformed public CLI requests refuse with an authored usage diagnostic.

    Parameters
    ----------
    flag : str
        Public benchmark CLI control given a zero value.
    """
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.coupling_builder_benchmark", flag, "0"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 2
    assert "controls must be positive integers" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("profiler_type", [cProfile.Profile, profile.Profile])
@pytest.mark.parametrize("n", [4, 1 << (struct.calcsize("P") * 4)])
def test_actual_caller_profiler_restored_after_measurement_or_refusal(
    profiler_type: type[cProfile.Profile] | type[profile.Profile], n: int
) -> None:
    """Actual native/Python profilers resume after public success and capacity refusal.

    Parameters
    ----------
    profiler_type : type[cProfile.Profile] or type[profile.Profile]
        Real standard-library profiler that owns the caller's profiling callback.
    n : int
        Valid dimension or guaranteed platform-capacity overflow dimension.
    """
    profiler = profiler_type()

    def exercise() -> None:
        """Exercise public measurement and recovery inside the real profiler."""
        previous = sys.getprofile()
        monitoring = getattr(sys, "monitoring", None)
        tool = monitoring.get_tool(monitoring.PROFILER_ID) if monitoring else None
        events = monitoring.get_events(monitoring.PROFILER_ID) if monitoring else 0
        assert previous is not None or (tool == "cProfile" and events != 0)
        if n == 4:
            assert measure_construction(n, 1, 1)["n"] == n
        else:
            with pytest.raises(ValueError, match="capacity"):
                measure_construction(n, 1, 1)
        assert sys.getprofile() is previous
        if monitoring is not None:
            assert monitoring.get_tool(monitoring.PROFILER_ID) == tool
            assert monitoring.get_events(monitoring.PROFILER_ID) == events
        recovered = measure_construction(4, 1, 1)
        assert recovered["n"] == 4
        assert sys.getprofile() is previous
        if monitoring is not None:
            assert monitoring.get_tool(monitoring.PROFILER_ID) == tool
            assert monitoring.get_events(monitoring.PROFILER_ID) == events

    profiler.runcall(exercise)
