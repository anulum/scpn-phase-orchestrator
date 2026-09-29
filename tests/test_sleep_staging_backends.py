# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep staging backend contracts

"""Exercise backend selection, real native parity and optional-kernel absence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Literal, cast

import numpy as np
import pytest

from benchmarks.sleep_staging_dispatch import measure
from scpn_phase_orchestrator.monitor.sleep_staging import (
    classify_sleep_stage,
    ultradian_phase,
)

Backend = Literal["python", "rust"]
HAS_KERNEL = importlib.util.find_spec("spo_kernel") is not None
BACKENDS: list[Backend] = ["python", "rust"] if HAS_KERNEL else ["python"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("desync", [False, True])
def test_backend_threshold_neighbours(backend: Backend, desync: bool) -> None:
    """Both engines preserve the inclusive boundaries and REM flag precedence."""
    bands = [
        (0.2, "Wake", "REM" if desync else "Wake"),
        (0.3, "REM" if desync else "Wake", "REM" if desync else "N1"),
        (0.4, "REM" if desync else "N1", "N2"),
        (0.7, "N2", "N3"),
    ]
    for threshold, below, above in bands:
        probes = [
            (float(np.nextafter(threshold, 0.0)), below),
            (threshold, above),
            (float(np.nextafter(threshold, 1.0)), above),
        ]
        for order_parameter, expected in probes:
            assert (
                classify_sleep_stage(order_parameter, desync, backend=backend)
                == expected
            )
            assert classify_sleep_stage(order_parameter, desync) == expected
    assert classify_sleep_stage(0.0, desync, backend=backend) == "Wake"
    assert classify_sleep_stage(1.0, desync, backend=backend) == "N3"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("epochs", [0, 1, 2, 1000, 10000])
@pytest.mark.parametrize("position", ["none", "first", "last", "multiple"])
def test_backend_history_phase(backend: Backend, epochs: int, position: str) -> None:
    """Real FFI preserves latest-N3 selection, wrapped phases and input arrays."""
    timestamps = (np.arange(epochs * 2, dtype=np.float64) * 30.0 - 900.0)[::2]
    timestamps.setflags(write=False)
    stages = ["N2"] * epochs
    expected = 0.0
    if epochs and position != "none":
        last_n3 = 0 if position == "first" else epochs - 1
        if position == "multiple":
            stages[0] = "N3"
            last_n3 = epochs // 2
        stages[last_n3] = "N3"
        expected = float((timestamps[-1] - timestamps[last_n3]) % 5400.0 / 5400.0)
    original_timestamps = timestamps.copy()
    original_stages = stages.copy()
    assert ultradian_phase(timestamps, stages, backend=backend) == expected
    assert ultradian_phase(timestamps, stages) == expected
    np.testing.assert_array_equal(timestamps, original_timestamps)
    assert stages == original_stages


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_repeated_epochs(backend: Backend) -> None:
    """Equal timestamps remain valid and the last N3 determines the origin."""
    timestamps = np.array([0.0, 2700.0, 2700.0, 5400.0, 13500.0])
    assert (
        ultradian_phase(timestamps, ["N3", "N2", "N3", "REM", "N2"], backend=backend)
        == 0.0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_validates_before_execution(backend: Backend) -> None:
    """Native selection cannot bypass scalar, time or label validation."""
    with pytest.raises(ValueError, match="R"):
        classify_sleep_stage(float("nan"), backend=backend)
    with pytest.raises(TypeError, match="functional_desync"):
        classify_sleep_stage(0.3, cast(bool, "yes"), backend=backend)
    with pytest.raises(ValueError, match="monotonic"):
        ultradian_phase(np.array([1.0, 0.0]), ["N3", "REM"], backend=backend)
    with pytest.raises(ValueError, match="unknown sleep stage"):
        ultradian_phase(np.array([0.0]), ["invalid"], backend=backend)


@pytest.mark.parametrize("backend", ["auto", "Rust", "", None])
def test_unknown_backend_refused_even_for_empty_history(backend: object) -> None:
    """Misspelled or implicit engine selection is not silently substituted."""
    selection = cast(Backend, backend)
    with pytest.raises(ValueError, match="backend must be"):
        classify_sleep_stage(0.5, backend=selection)
    with pytest.raises(ValueError, match="backend must be"):
        ultradian_phase(np.array([]), [], backend=selection)


def test_optional_kernel_contract() -> None:
    """Installed kernels run; a genuinely absent kernel is explicitly refused."""
    timestamps = np.array([0.0, 2700.0])
    assert classify_sleep_stage(0.5) == "N2"
    assert ultradian_phase(timestamps, ["N3", "REM"]) == 0.5
    if HAS_KERNEL:
        assert classify_sleep_stage(0.5, backend="rust") == "N2"
        assert ultradian_phase(timestamps, ["N3", "REM"], backend="rust") == 0.5
    else:
        with pytest.raises(RuntimeError, match="optional spo_kernel"):
            classify_sleep_stage(0.5, backend="rust")
        with pytest.raises(RuntimeError, match="optional spo_kernel"):
            ultradian_phase(timestamps, ["N3", "REM"], backend="rust")
        with pytest.raises(RuntimeError, match="optional spo_kernel"):
            ultradian_phase(np.array([]), [], backend="rust")


def test_dispatch_invokes_only_the_requested_native_functions() -> None:
    """Observe real C calls without replacing imports or backend functions."""
    native_calls: list[str] = []

    def record_native_call(frame: FrameType, event: str, argument: object) -> None:
        """Collect only the two PyO3 sleep functions from the running interpreter."""
        if event == "c_call":
            name = getattr(argument, "__name__", "")
            if name in ("classify_sleep_stage_rust", "ultradian_phase_rust"):
                native_calls.append(name)

    previous_profiler = sys.getprofile()
    sys.setprofile(record_native_call)
    try:
        for explicit_python in (False, True):
            if explicit_python:
                assert classify_sleep_stage(0.8, backend="python") == "N3"
                assert (
                    ultradian_phase(
                        np.array([0.0, 2700.0]), ["N3", "REM"], backend="python"
                    )
                    == 0.5
                )
            else:
                assert classify_sleep_stage(0.8) == "N3"
                assert ultradian_phase(np.array([0.0, 2700.0]), ["N3", "REM"]) == 0.5
        assert native_calls == []
        if HAS_KERNEL:
            assert classify_sleep_stage(0.8, backend="rust") == "N3"
            assert (
                ultradian_phase(np.array([0.0, 2700.0]), ["N3", "REM"], backend="rust")
                == 0.5
            )
            assert native_calls == ["classify_sleep_stage_rust", "ultradian_phase_rust"]
    finally:
        sys.setprofile(previous_profiler)


def test_benchmark_cli_measures_public_dispatch() -> None:
    """The standalone benchmark runs the actual available interpreter and kernel."""
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/sleep_staging_dispatch.py",
            "--number",
            "1",
            "--repeat",
            "2",
            "--epochs",
            "0",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if not HAS_KERNEL:
        assert result.returncode != 0
        assert "release spo_kernel extension is required" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["isolation"].startswith("non-isolated")
    assert len(evidence["extension_sha256"]) == 64
    assert len(evidence["results"]) == 11
    for measurement in evidence["results"]:
        assert set(measurement["outputs"]) == {"default", "python", "rust"}
        assert len(set(measurement["outputs"].values())) == 1
        assert all(len(samples) == 2 for samples in measurement["samples_us"].values())
        assert all(timing > 0 for timing in measurement["median_us"].values())


def test_benchmark_refuses_incomparable_outputs() -> None:
    """Do not report timing comparisons for invocations with different results."""
    with pytest.raises(ValueError, match="sleep backend output mismatch"):
        measure(
            {
                "wake": partial(classify_sleep_stage, 0.1),
                "deep": partial(classify_sleep_stage, 0.8),
            },
            number=1,
            repeat=1,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--number", "0"],
        ["--repeat", "0"],
        ["--epochs", "-1"],
    ],
)
def test_benchmark_cli_rejects_invalid_measurement_counts(arguments: list[str]) -> None:
    """Invalid sampling requests fail before any timing or kernel execution."""
    result = subprocess.run(
        [sys.executable, "benchmarks/sleep_staging_dispatch.py", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "number and repeat must be positive" in result.stderr
