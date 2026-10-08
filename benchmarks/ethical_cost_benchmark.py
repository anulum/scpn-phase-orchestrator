# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Installed ethical-cost backend comparison

"""Measure original public owners in separate native and kernel-absent installs.

Provide --python-profile and --rust-profile as qualified installed interpreter
paths. Every worker checks the actual owner, source location and source hash;
no resolution flag or numerical callable is replaced. Batch means and separately
observed individual-call latency percentiles are labeled distinctly. Timings
exclude interpreter startup.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import subprocess
import sys
from numbers import Integral
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import cast

import numpy as np

from scpn_phase_orchestrator.ssgf import ethical


def validate_profile_python(executable: Path) -> Path:
    """Require a profile executable identical to this trusted Python binary.

    Parameters
    ----------
    executable : pathlib.Path
        Explicit operator-selected interpreter in a qualified installation.

    Returns
    -------
    pathlib.Path
        Absolute invocation path retaining the virtual environment identity.

    Raises
    ------
    ValueError
        If the path is missing or differs from the actual running interpreter.
    """
    target = executable.absolute()
    if (
        not target.is_file()
        or hashlib.sha256(target.read_bytes()).digest()
        != hashlib.sha256(Path(sys.executable).read_bytes()).digest()
    ):
        raise ValueError("profile executable must match the trusted running Python")
    return target


def measure_current(n: int, calls: int, batches: int, owner: str) -> dict[str, object]:
    """Measure the real default public function in the current installed runtime.

    Parameters
    ----------
    n : int
        Positive oscillator count for the fixed seed-42 fixture.
    calls : int
        Positive number of calls per independently timed batch.
    batches : int
        Positive number of batch means to retain.
    owner : str
        Required original ``rust`` builtin or actually kernel-absent ``python``.

    Returns
    -------
    dict[str, object]
        Actual numerical fields, raw batch timings and source/binary provenance.

    Raises
    ------
    ValueError
        If controls or the required owner name are invalid.
    RuntimeError
        If the installed runtime does not have the required original owner.
    """
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
        for v in (n, calls, batches)
    ) or owner not in {"python", "rust"}:
        raise ValueError(
            "positive sizes, calls, batches and a valid owner are required"
        )
    native = ethical._rust_ethical_cost
    if owner == "rust":
        if native is None or not inspect.isbuiltin(native):
            raise RuntimeError("the original compiled ethical-cost owner is required")
    elif native is not None or importlib.util.find_spec("spo_kernel") is not None:
        raise RuntimeError("Python timing requires a genuinely kernel-absent install")
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    profile = cProfile.Profile()
    profile.enable()
    result = ethical.compute_ethical_cost(phases, knm)
    profile.disable()
    native_calls = sum(
        e.callcount
        for e in profile.getstats()
        if isinstance(e.code, str) and "compute_ethical_cost_rust" in e.code
    )
    if native_calls != (1 if owner == "rust" else 0):
        raise RuntimeError(
            "observed public compute path does not match the required owner"
        )
    samples: list[float] = []
    for _ in range(batches):
        start = perf_counter_ns()
        for _ in range(calls):
            ethical.compute_ethical_cost(phases, knm)
        samples.append((perf_counter_ns() - start) / calls / 1000.0)
    latencies: list[float] = []
    for _ in range(calls * batches):
        start = perf_counter_ns()
        ethical.compute_ethical_cost(phases, knm)
        latencies.append((perf_counter_ns() - start) / 1000.0)
    module = Path(ethical.__file__)
    row: dict[str, object] = {
        "n": n,
        "owner": owner,
        "module": str(module),
        "source_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(phases.tobytes() + knm.tobytes()).hexdigest(),
        "cost": [result.J_sec, result.phi_ethics, result.c15_sec],
        "violations": result.constraints_violated,
        "native_calls": native_calls,
        "batch_mean_us": samples,
        "median_batch_mean_us": median(samples),
        "latency_us": latencies,
        "p50_us": float(np.percentile(latencies, 50)),
        "p95_us": float(np.percentile(latencies, 95)),
        "p99_us": float(np.percentile(latencies, 99)),
        "calls_per_batch": calls,
        "batches": batches,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if native is not None:
        native_module = importlib.import_module(native.__module__)
        binary_path = native_module.__file__
        if binary_path is None:
            raise RuntimeError("the compiled owner has no backing file")
        binary = Path(binary_path)
        row["binary"] = str(binary)
        row["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return row


def benchmark_size(
    n: int,
    calls: int,
    batches: int,
    *,
    python_profile: Path,
    rust_profile: Path,
) -> dict[str, object]:
    """Compare two source-matching installed owners on identical real fixtures.

    Parameters
    ----------
    n : int
        Positive oscillator count.
    calls : int
        Calls per batch.
    batches : int
        Number of retained batch means.
    python_profile : pathlib.Path
        Interpreter in a genuine kernel-absent project installation.
    rust_profile : pathlib.Path
        Interpreter in the same-source installation with the compiled kernel.

    Returns
    -------
    dict[str, object]
        Both full worker records and their median batch-mean ratio.

    Raises
    ------
    RuntimeError
        If a worker refuses, its installed source differs, or parity fails.
    """
    rows: list[dict[str, object]] = []
    for owner, executable in (("python", python_profile), ("rust", rust_profile)):
        executable = validate_profile_python(executable)
        result = subprocess.run(  # noqa: S603 - verified original Python binary; explicit profile, no shell.
            [
                str(executable),
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                owner,
                "--sizes",
                str(n),
                "--calls",
                str(calls),
                "--batches",
                str(batches),
            ],
            cwd=executable.parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        row = cast("dict[str, object]", json.loads(result.stdout))
        expected = hashlib.sha256(Path(ethical.__file__).read_bytes()).hexdigest()
        if row["source_sha256"] != expected:
            raise RuntimeError(
                "installed ethical-cost source does not match this candidate"
            )
        rows.append(row)
    python, rust = rows
    from benchmarks.ethical_cost_reference import reference_cost

    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    expected_cost = reference_cost(phases, knm)
    if python["input_sha256"] != rust["input_sha256"] or any(
        not np.allclose(
            cast("list[float]", row["cost"]), expected_cost[:3], rtol=1e-10, atol=1e-10
        )
        or row["violations"] != expected_cost[3]
        for row in rows
    ):
        raise RuntimeError(f"ethical backend parity failed for N={n}")
    return {
        "n": n,
        "python": python,
        "rust": rust,
        "parity_passed": True,
        "python_over_rust": float(cast("float", python["median_batch_mean_us"]))
        / float(cast("float", rust["median_batch_mean_us"])),
    }


def main() -> None:
    """Print measurements only after every required installed owner completes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--python-profile", type=Path)
    parser.add_argument("--rust-profile", type=Path)
    parser.add_argument("--worker", choices=["python", "rust"])
    args = parser.parse_args()
    if args.worker:
        if len(args.sizes) != 1:
            parser.error("one size is required per worker")
        result = measure_current(args.sizes[0], args.calls, args.batches, args.worker)
    else:
        if args.python_profile is None or args.rust_profile is None:
            parser.error("both genuine installed profile interpreters are required")
        result = {
            "results": [
                benchmark_size(
                    n,
                    args.calls,
                    args.batches,
                    python_profile=args.python_profile,
                    rust_profile=args.rust_profile,
                )
                for n in args.sizes
            ]
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
