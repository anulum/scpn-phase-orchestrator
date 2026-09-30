# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical extraction diagnostics

"""Measure real public physical extraction, with observed native-call provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor


class PhysicalMeasurement(TypedDict):
    """One timed workload and its actual native-call and numerical observations."""

    name: str
    band: list[float]
    edge_trim: int | None
    samples: int
    sample_rate_hz: float
    signal_sha256: str
    observed_backend: str
    native_calls: list[str]
    median_seconds: float
    p95_seconds: float
    fields: list[float]
    reference_fields: list[float]
    analytic_sha256: str


class PhysicalBenchmarkReport(TypedDict):
    """Environment-bound public-extraction diagnostics, not isolated speed claims."""

    format_version: int
    recorded_at: str
    classification: str
    python: str
    numpy: str
    scipy: str
    kernel_present: bool
    native_artifact: dict[str, str] | None
    repeats: int
    host: dict[str, object]
    source_sha256: dict[str, str]
    cases: list[PhysicalMeasurement]


PhysicalCase = tuple[
    str, NDArray[np.float64], float, tuple[float, float] | None, int | None
]


def _cases() -> list[PhysicalCase]:
    """Return periodic, modulated, large, filtered and degenerate public workloads."""
    phase = 2 * np.pi * np.arange(1000, dtype=np.float64) / 1000
    cases: list[PhysicalCase] = [
        ("sinusoid_1000", np.cos(10 * phase), 1000.0, None, None),
        (
            "bandpass_1000",
            np.cos(10 * phase) + np.cos(60 * phase),
            1000.0,
            (8.0, 12.0),
            17,
        ),
        ("zero_envelope", np.zeros(128, dtype=np.float64), 128.0, None, 0),
        ("tiny_envelope", np.full(128, 1e-17), 128.0, None, None),
        ("trim_to_two", np.cos(10 * phase), 1000.0, None, 10000),
    ]
    for length in (127, 128):
        phase = 2 * np.pi * np.arange(length, dtype=np.float64) / length
        waveform = (1 + 0.6 * np.cos(phase)) * np.cos(8 * phase)
        cases.append((f"modulated_{length}", waveform, float(length), None, None))
        cases.append(
            (f"large_modulated_{length}", 1e290 * waveform, float(length), None, None)
        )
    return cases


def _reference(case: PhysicalCase) -> tuple[list[float], str]:
    """Evaluate the documented equations separately using this runtime's SciPy."""
    _name, signal, sample_rate, band, edge_trim = case
    filtered = signal
    if band is not None:
        b, a = butter(4, np.asarray(band) / (0.5 * sample_rate), btype="band")
        filtered = np.asarray(filtfilt(b, a, signal), dtype=np.float64)
    analytic = hilbert(filtered)
    if edge_trim is not None:
        trim = min(edge_trim, max(0, (len(signal) - 2) // 2))
        if trim:
            analytic = analytic[trim:-trim]
    phase = np.angle(analytic)
    envelope = np.abs(analytic)
    scale = float(np.max(envelope))
    normalized = envelope / scale if scale > 0 else envelope
    mean = float(np.mean(normalized))
    amplitude = scale * mean
    quality = (
        float(np.clip(1 - np.std(normalized) / mean, 0, 1))
        if amplitude >= 1e-15
        else 0.0
    )
    return [
        float(phase[-1] % (2 * np.pi)),
        float(np.median(np.gradient(np.unwrap(phase)))) * sample_rate,
        amplitude,
        quality,
    ], hashlib.sha256(analytic.tobytes()).hexdigest()


def benchmark_physical_extraction(*, repeats: int = 20) -> PhysicalBenchmarkReport:
    """Time public extraction and record real C-call identities outside timed loops.

    Parameters
    ----------
    repeats : int
        Positive number of timed calls per workload, following one warm-up.

    Returns
    -------
    PhysicalBenchmarkReport
        Non-isolated timings in seconds, observed implementation, numerical
        samples, source hashes and host/runtime context for each workload.

    Raises
    ------
    ValueError
        If repeats is boolean, non-integral or less than one.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    kernel_present = importlib.util.find_spec("spo_kernel") is not None
    owners: dict[str, object] = {}
    native_artifact: dict[str, str] | None = None
    if kernel_present:
        kernel = importlib.import_module("spo_kernel")
        native_module = importlib.import_module("spo_kernel.spo_kernel")
        artifact = Path(inspect.getfile(native_module))
        native_artifact = {
            "filename": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        owners = {"physical_extract": kernel.physical_extract}
    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    measurements: list[PhysicalMeasurement] = []
    for name, signal, sample_rate, band, edge_trim in _cases():
        extractor = PhysicalExtractor(band=band, edge_trim=edge_trim)
        executed: set[str] = set()

        def observe_call(
            frame: FrameType,
            event: str,
            argument: object,
            observed: set[str] = executed,
        ) -> None:
            """Identify calls to installed physical owners without replacing them."""
            if event == "c_call":
                for owner_name, owner in owners.items():
                    if argument is owner:
                        observed.add(owner_name)

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(observe_call)
            states = extractor.extract(signal, sample_rate)
        finally:
            sys.setprofile(previous_profile)
        durations = []
        for _repeat in range(repeats):
            started = time.perf_counter()
            extractor.extract(signal, sample_rate)
            durations.append(time.perf_counter() - started)
        state = states[0]
        fields = [state.theta, state.omega, state.amplitude, state.quality]
        reference, analytic_hash = _reference(
            (name, signal, sample_rate, band, edge_trim)
        )
        np.testing.assert_allclose(fields, reference, rtol=1e-10, atol=1e-10)
        measurements.append(
            {
                "name": name,
                "band": list(band) if band is not None else [],
                "edge_trim": edge_trim,
                "samples": len(signal),
                "sample_rate_hz": sample_rate,
                "signal_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
                "observed_backend": "native" if executed else "python",
                "native_calls": sorted(executed),
                "median_seconds": float(np.median(durations)),
                "p95_seconds": float(np.percentile(durations, 95)),
                "fields": fields,
                "reference_fields": reference,
                "analytic_sha256": analytic_hash,
            }
        )
    repository = Path(__file__).resolve().parents[1]
    sources = (
        "benchmarks/physical_extraction.py",
        "src/scpn_phase_orchestrator/oscillators/physical.py",
        "spo-kernel/crates/spo-oscillators/src/physical.rs",
        "spo-kernel/crates/spo-ffi/src/lib.rs",
    )
    return {
        "format_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "classification": "non-isolated functional and local timing diagnostic",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "kernel_present": kernel_present,
        "native_artifact": native_artifact,
        "repeats": repeats,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported",
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "load_before": load_before,
            "load_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "isolation": "none; shared workstation",
        },
        "source_sha256": {
            source: hashlib.sha256((repository / source).read_bytes()).hexdigest()
            for source in sources
        },
        "cases": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    """Emit the public extraction diagnostic as JSON from the command line.

    Parameters
    ----------
    argv : list[str] or None
        Command arguments without the program name; None reads process arguments.

    Returns
    -------
    int
        Zero after printing one finite JSON report to standard output.

    Raises
    ------
    SystemExit
        If argument parsing fails or the repeat count is invalid; the exit code
        is two and the usage error goes to standard error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        report = benchmark_physical_extraction(repeats=args.repeats)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
