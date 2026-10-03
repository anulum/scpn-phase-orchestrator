# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual coupling construction benchmark

"""Measure public construction and the actual installed native counterpart."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import statistics
import struct
import sys
import time
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from types import FrameType

import numpy as np

from scpn_phase_orchestrator.coupling.knm import (
    SCPN_CALIBRATION_ANCHORS,
    CouplingBuilder,
    CouplingState,
)


def measure_construction(
    n: int, calls: int = 10, repeats: int = 3
) -> dict[str, object]:
    """Measure identical public and native inputs, checking results outside timing.

    Parameters
    ----------
    n : int
        Positive matrix dimension. SCPN construction is also measured when 16.
    calls : int
        Positive number of calls per uninstrumented sample.
    repeats : int
        Positive number of samples following two warm-up calls.

    Returns
    -------
    dict[str, object]
        Scalar input bits, runtime identity, native dispatch observation, matrix
        hashes and individual elapsed samples. Public amplitude includes phase
        construction; direct native measures generic phase construction only.

    Raises
    ------
    ValueError
        If controls are not positive non-boolean integers or n is unrepresentable.
    AssertionError
        If a measured result violates its declared numerical contract.

    Notes
    -----
    Measurements share a loaded host and establish reproducible local regression
    evidence. They do not establish isolated latency, speed-up or physical truth.
    Existing native and Python caller profilers are suspended for the complete
    measurement and restored through their original APIs on success or refusal.
    """
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or v < 1
        for v in (n, calls, repeats)
    ):
        raise ValueError("controls must be positive non-boolean integers")
    n, calls, repeats = int(n), int(calls), int(repeats)
    previous_profile = sys.getprofile()
    monitoring = getattr(sys, "monitoring", None)
    profile_events = monitoring.get_events(monitoring.PROFILER_ID) if monitoring else 0
    if isinstance(previous_profile, cProfile.Profile):
        previous_profile.disable()
    else:
        sys.setprofile(None)
    if monitoring is not None and profile_events:
        monitoring.set_events(monitoring.PROFILER_ID, 0)
    try:
        return _measure_construction(n, calls, repeats)
    finally:
        if isinstance(previous_profile, cProfile.Profile):
            previous_profile.enable()
        else:
            sys.setprofile(previous_profile)
        if monitoring is not None and profile_events:
            monitoring.set_events(monitoring.PROFILER_ID, profile_events)


def _measure_construction(n: int, calls: int, repeats: int) -> dict[str, object]:
    """Measure admitted inputs with caller profiling suspended."""
    builder = CouplingBuilder()
    # Construction refuses impossible dimensions before this benchmark's oracle
    # can allocate its own buffers.
    builder.build(n, 0.45, 0.3)
    expected = np.array(
        [
            [0.0 if i == j else 0.45 * math.exp(-0.3 * abs(i - j)) for j in range(n)]
            for i in range(n)
        ]
    )
    amplitude = np.array(
        [
            [0.0 if i == j else 0.2 * math.exp(-0.1 * abs(i - j)) for j in range(n)]
            for i in range(n)
        ]
    )
    native = (
        importlib.import_module("spo_kernel")
        if importlib.util.find_spec("spo_kernel")
        else None
    )
    calls_seen = 0

    def observe_native_build(_frame: FrameType, event: str, argument: object) -> None:
        """Count actual compiled build calls outside the measured intervals."""
        nonlocal calls_seen
        if event == "c_call" and getattr(argument, "__name__", None) == "build":
            calls_seen += 1

    try:
        sys.setprofile(observe_native_build)
        builder.build(n, 0.45, 0.3)
    finally:
        sys.setprofile(None)
    assert (calls_seen > 0) == (native is not None)

    def verify(state: CouplingState, kind: str) -> None:
        """Check numerical output, dimensions and fresh buffer ownership."""
        assert state.knm.shape == (n, n)
        assert np.all(np.isfinite(state.knm))
        np.testing.assert_array_equal(state.knm, state.knm.T)
        np.testing.assert_array_equal(np.diag(state.knm), 0.0)
        np.testing.assert_array_equal(state.alpha, 0.0)
        if kind == "scpn_physics":
            for (i, j), strength in SCPN_CALIBRATION_ANCHORS.items():
                assert state.knm[i - 1, j - 1] == strength
            assert state.knm[0, 15] >= 0.05
            assert state.knm[4, 6] >= 0.15
        else:
            np.testing.assert_allclose(state.knm, expected, rtol=3e-15, atol=0.0)
        if kind == "amplitude":
            assert state.knm_r is not None
            np.testing.assert_allclose(state.knm_r, amplitude, rtol=3e-15, atol=0.0)

    def measure(fn: Callable[[], CouplingState], kind: str) -> dict[str, object]:
        """Verify warm-ups and each sample without timing assertions or profiling."""
        first, second = fn(), fn()
        verify(first, kind)
        verify(second, kind)
        assert not np.shares_memory(first.knm, second.knm)
        assert not np.shares_memory(first.alpha, second.alpha)
        samples: list[float] = []
        result = second
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls):
                result = fn()
            samples.append((time.perf_counter() - start) * 1e6 / calls)
            verify(result, kind)
        return {
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "knm_sha256": hashlib.sha256(
                result.knm.astype("<f8").tobytes()
            ).hexdigest(),
        }

    row: dict[str, object] = {
        "n": n,
        "base_strength": 0.45,
        "decay_alpha": 0.3,
        "calls": calls,
        "repeats": repeats,
        "warmups": 2,
        "input_encoding": "little-endian uint64 n then IEEE754 binary64 base and decay",
        "input_hex": struct.pack("<Qdd", n, 0.45, 0.3).hex(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "load_before": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "isolated": False,
        "native_build_calls_observed": calls_seen,
        "native_installation": native.__file__ if native else None,
        "generic": measure(lambda: builder.build(n, 0.45, 0.3), "generic"),
        "amplitude": measure(
            lambda: builder.build_with_amplitude(n, 0.45, 0.3, 0.2, 0.1), "amplitude"
        ),
    }
    if n == 16:
        row["scpn_physics"] = measure(builder.build_scpn_physics, "scpn_physics")
    if native:

        def direct() -> CouplingState:
            """Build actual PyO3 output and convert its arrays inside timing."""
            data = native.PyCouplingBuilder().build(n, 0.45, 0.3)
            return CouplingState(
                np.asarray(data["knm"], dtype=np.float64).reshape(n, n),
                np.asarray(data["alpha"], dtype=np.float64).reshape(n, n),
                "default",
            )

        row["direct_native_with_array_conversion"] = measure(direct, "generic")
        assert native.__file__ is not None
        directory = Path(native.__file__).parent
        row["native_binary_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.so"))
        }
    row["load_after"] = os.getloadavg()
    return row


def main() -> None:
    """Print strict JSON bound to the actual interpreter and current source bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 100])
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(v < 1 for v in [*args.sizes, args.calls, args.repeats]):
        parser.error("controls must be positive integers")
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/scpn_phase_orchestrator/coupling/knm.py",
        "spo-kernel/crates/spo-engine/src/coupling.rs",
        "spo-kernel/crates/spo-ffi/src/coupling_builder.rs",
        "benchmarks/coupling_builder_benchmark.py",
    ]
    print(
        json.dumps(
            {
                "source_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in paths
                },
                "rows": [
                    measure_construction(n, args.calls, args.repeats)
                    for n in args.sizes
                ],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
