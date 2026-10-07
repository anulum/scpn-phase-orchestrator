# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual phase attention integration diagnostics

"""Measure original named runtimes on paired, independently checked UPDE loops.

Shared-host samples are diagnostics, not isolated speedup or production-budget
acceptance. Named runtimes cannot silently fall back. Loader discovery is
recorded separately from actual observed calls and artifact identities.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.attnres_reference import FloatArray, phase_attention_oracle
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine


class Artifact(TypedDict):
    """Actual source or runtime file identity used by this process."""

    path: str
    sha256: str


class Timing(TypedDict):
    """Paired wall-time samples after successful numerical qualification."""

    durations_seconds: list[float]
    median_ms_per_step: float
    observed_calls: list[str]
    oracle_max_abs_error: float
    final_phases: list[float]


class BenchmarkRow(TypedDict):
    """One explicit graph size with the same inputs for every named owner."""

    n: int
    n_steps: int
    repetitions: int
    available: list[str]
    baseline: Timing
    backends: dict[str, Timing]


def _artifact(path: Path) -> Artifact:
    """Hash an actual regular file rather than infer identity from a version."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _euler_reference(
    phases: FloatArray,
    omegas: FloatArray,
    coupling: FloatArray,
    alpha: FloatArray,
    dt: float,
) -> FloatArray:
    """Evaluate target-row/source-column Sakaguchi dynamics in rad/s."""
    result = phases.copy()
    for target in range(len(phases)):
        derivative = float(omegas[target]) + sum(
            float(coupling[target, source])
            * float(np.sin(phases[source] - phases[target] - alpha[target, source]))
            for source in range(len(phases))
        )
        result[target] = (phases[target] + dt * derivative) % (2.0 * np.pi)
    return result


def _run_loop(
    backend: str | None,
    phases: FloatArray,
    omegas: FloatArray,
    coupling: FloatArray,
    alpha: FloatArray,
    steps: int,
    dt: float,
) -> FloatArray:
    """Feed actual modulated matrices into the real Euler integrator."""
    engine = UPDEEngine(n_oscillators=len(phases), dt=dt, method="euler")
    result = phases.copy()
    for _ in range(steps):
        matrix = (
            coupling
            if backend is None
            else attnres_modulate(
                coupling, result, block_size=4, lambda_=0.5, backend=backend
            )
        )
        result = engine.step(result, omegas, matrix, 0.0, 0.0, alpha)
    return result


def _measure(
    backend: str | None,
    phases: FloatArray,
    omegas: FloatArray,
    coupling: FloatArray,
    alpha: FloatArray,
    expected: FloatArray,
    steps: int,
    dt: float,
    repeats: int,
) -> Timing:
    """Check original owner calls and numerical results before paired timing."""
    with cProfile.Profile() as profile:
        final = _run_loop(backend, phases, omegas, coupling, alpha, steps, dt)
    profile.create_stats()
    calls = sorted({entry[2] for entry in profile.stats})
    if backend is not None:
        owner = {
            "python": "_python_fallback",
            "rust": "<built-in method spo_kernel.spo_kernel.attnres_modulate_rust>",
            "go": "attnres_modulate_go",
            "julia": "attnres_modulate_julia",
            "mojo": "attnres_modulate_mojo",
        }[backend]
        if owner not in calls or (backend != "python" and "_python_fallback" in calls):
            raise RuntimeError(f"named owner {backend} did not execute: {calls}")
    np.testing.assert_allclose(final, expected, rtol=0.0, atol=2e-11)
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        final = _run_loop(backend, phases, omegas, coupling, alpha, steps, dt)
        durations.append(time.perf_counter() - started)
        np.testing.assert_allclose(final, expected, rtol=0.0, atol=2e-11)
    return {
        "durations_seconds": durations,
        "median_ms_per_step": 1000.0 * statistics.median(durations) / steps,
        "observed_calls": calls,
        "oracle_max_abs_error": float(np.max(np.abs(final - expected))),
        "final_phases": final.tolist(),
    }


def bench_one(
    n: int,
    n_steps: int = 20,
    dt: float = 0.01,
    *,
    repeats: int = 20,
    backends: tuple[str, ...] | None = None,
) -> BenchmarkRow:
    """Compare actual named coupling owners on identical complete trajectories.

    Parameters
    ----------
    n : int
        Positive oscillator count.
    n_steps : int
        Positive steps per trajectory.
    dt : float
        Finite positive Euler interval in seconds.
    repeats : int
        Positive paired repetitions after warm-up and independent checks.
    backends : tuple of str or None
        Required owners; defaults to currently discovered owners.

    Returns
    -------
    BenchmarkRow
        Raw durations, observed calls, independent oracle errors and phases.

    Raises
    ------
    ValueError
        On invalid counts, interval, duplicate or unknown owners.
    ImportError
        If a required optional runtime is missing.
    RuntimeError
        If a named owner was not observed.
    AssertionError
        If real output disagrees with the independent numerical reference.
    """
    for name, value in (("n", n), ("n_steps", n_steps), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    owners = tuple(AVAILABLE_BACKENDS) if backends is None else backends
    if (
        not owners
        or len(set(owners)) != len(owners)
        or any(
            owner not in ("python", "rust", "go", "julia", "mojo") for owner in owners
        )
    ):
        raise ValueError("backends must be distinct supported owner names")
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float64)
    omegas = (rng.standard_normal(n) * 1.5).astype(np.float64)
    half = rng.uniform(0.0, 1.0 / n, size=(n, n))
    coupling = 0.5 * (half + half.T)
    np.fill_diagonal(coupling, 0.0)
    alpha = np.full((n, n), 0.07)
    np.fill_diagonal(alpha, 0.0)
    weights = default_projections()
    expected_base = phases.copy()
    expected_modulated = phases.copy()
    for _ in range(n_steps):
        expected_base = _euler_reference(expected_base, omegas, coupling, alpha, dt)
        matrix = phase_attention_oracle(coupling, expected_modulated, weights, radius=4)
        expected_modulated = _euler_reference(
            expected_modulated, omegas, matrix, alpha, dt
        )
    baseline = _measure(
        None, phases, omegas, coupling, alpha, expected_base, n_steps, dt, repeats
    )
    measured = {
        owner: _measure(
            owner,
            phases,
            omegas,
            coupling,
            alpha,
            expected_modulated,
            n_steps,
            dt,
            repeats,
        )
        for owner in owners
    }
    return {
        "n": n,
        "n_steps": n_steps,
        "repetitions": repeats,
        "available": list(AVAILABLE_BACKENDS),
        "baseline": baseline,
        "backends": measured,
    }


def main() -> int:
    """Run comparison diagnostics and emit source-bound JSON through the real CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--backends", nargs="+", choices=("python", "rust", "go", "julia", "mojo")
    )
    args = parser.parse_args()
    owners = None if args.backends is None else tuple(args.backends)
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    started = time.time()
    results = [
        bench_one(n, n_steps=args.steps, repeats=args.repeats, backends=owners)
        for n in args.sizes
    ]
    root = Path(__file__).resolve().parents[1]
    sources = [
        _artifact(root / name)
        for name in (
            "src/scpn_phase_orchestrator/coupling/attention_residuals.py",
            "src/scpn_phase_orchestrator/coupling/_attnres_validation.py",
            "src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_go.py",
            "src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_julia.py",
            "src/scpn_phase_orchestrator/experimental/accelerators/coupling/_attnres_mojo.py",
            "spo-kernel/crates/spo-engine/src/attnres.rs",
            "go/attnres.go",
            "julia/attnres.jl",
            "mojo/attnres.mojo",
            "benchmarks/attnres_modulation_benchmark.py",
            "benchmarks/attnres_reference.py",
        )
    ]
    artifacts: list[Artifact] = []
    if any("rust" in row["backends"] for row in results):
        kernel = importlib.import_module("spo_kernel.spo_kernel")
        artifacts.append(_artifact(Path(cast(str, kernel.__file__))))
    for name, owner in (("go/libattnres.so", "go"), ("mojo/attnres_mojo", "mojo")):
        if any(owner in row["backends"] for row in results):
            artifacts.append(_artifact(root / name))
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    report: dict[str, object] = {
        "schema": "spo.phase-attention-diagnostics.v2",
        "claim_boundary": (
            "shared-host diagnostics; no isolated speedup "
            "or production-budget acceptance"
        ),
        "host": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "command": sys.argv,
        "started_unix": started,
        "finished_unix": time.time(),
        "isolation": "none; shared workstation",
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "frequency_governor": (
            governor_path.read_text().strip() if governor_path.is_file() else None
        ),
        "load_average_before": load_before,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "sources": sources,
        "artifacts": artifacts,
        "results": results,
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if os.name == "posix":
        os.set_blocking(sys.stdout.fileno(), True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
