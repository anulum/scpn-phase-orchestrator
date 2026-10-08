# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Current chimera CPU and JAX comparison evidence

"""Collect source-bound POSIX native comparisons through original public owners."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from benchmarks.chimera_local_order_reference import scalar_local_order
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _chimera_go,
    _chimera_julia,
    _chimera_mojo,
)
from scpn_phase_orchestrator.monitor import chimera

FloatArray = NDArray[np.float64]
OWNERS = ("rust", "mojo", "julia", "go", "python")
REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/scpn_phase_orchestrator/monitor/chimera.py",
    "src/scpn_phase_orchestrator/nn/chimera.py",
    "src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_go.py",
    "src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_julia.py",
    "src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_mojo.py",
    "src/scpn_phase_orchestrator/experimental/accelerators/monitor/_chimera_validation.py",
    "spo-kernel/crates/spo-engine/src/chimera.rs",
    "spo-kernel/crates/spo-ffi/src/chimera_boundary.rs",
    "go/chimera.go",
    "julia/chimera.jl",
    "mojo/chimera.mojo",
    "benchmarks/chimera_benchmark.py",
    "benchmarks/chimera_comparison.py",
    "benchmarks/chimera_local_order_reference.py",
)


def _count(value: object, name: str, minimum: int) -> int:
    """Admit a plain integral comparison control at its physical minimum."""
    if isinstance(
        value, (bool, np.bool_, np.timedelta64, np.datetime64)
    ) or not isinstance(value, Integral):
        raise ValueError(name + " must be a plain integer")
    count = int(value)
    if count < minimum:
        raise ValueError(name + " must be at least " + str(minimum))
    return count


def require_owners(required: Sequence[str]) -> None:
    """Refuse unsupported or actually unavailable owners before measuring.

    Parameters
    ----------
    required : collections.abc.Sequence[str]
        Exact CPU owner names required by the caller's qualification lane.

    Raises
    ------
    ValueError
        For unsupported names or a missing actual resolved owner.
    """
    unknown = [name for name in required if name not in OWNERS]
    if unknown:
        raise ValueError("Unknown chimera benchmark owner: " + ", ".join(unknown))
    missing = [name for name in required if name not in chimera.AVAILABLE_BACKENDS]
    if missing:
        raise ValueError(
            "Required chimera benchmark owners unavailable: " + ", ".join(missing)
        )


def _hashes() -> dict[str, str]:
    """Hash current source and actual native artifacts admitted for this comparison."""
    kernel = importlib.import_module("spo_kernel.spo_kernel")
    binary = Path(cast(str, vars(kernel)["__file__"]))
    paths = {name: REPO / name for name in SOURCE_FILES}
    paths.update(
        rust_binary=binary,
        go_binary=_chimera_go._LIB_PATH,
        mojo_binary=_chimera_mojo._EXE_PATH,
        julia_loaded_source=_chimera_julia._JULIA_FILE,
    )
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def _measure(operation: Callable[[], object], calls: int) -> tuple[list[float], object]:
    """Warm a real public call and retain one duration per completed invocation."""
    output = operation()
    samples = []
    for _ in range(calls):
        started = time.perf_counter_ns()
        output = operation()
        samples.append((time.perf_counter_ns() - started) / 1e9)
    return samples, output


def _statistics(samples: list[float]) -> dict[str, object]:
    """Retain raw durations and their population distribution in seconds."""
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_seconds": samples,
        "sample_count": len(samples),
        "warmup_calls": 2,
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values)),
        "min_seconds": float(np.min(values)),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "max_seconds": float(np.max(values)),
    }


def benchmark_comparison(
    sizes: Sequence[int] = (16, 64, 256),
    *,
    calls: int = 20,
    density: float = 0.3,
    seed: int = 2026,
) -> dict[str, object]:
    """Compare every CPU owner and separately labelled actual JAX local-order model.

    Parameters
    ----------
    sizes : collections.abc.Sequence[int]
        Nonempty population sizes, each at least three.
    calls : int
        At least twenty retained completed timings, after two untimed warm-ups.
    density : float
        Finite adjacency-generation density in [0,1].
    seed : int
        Nonnegative NumPy workload seed.

    Returns
    -------
    dict[str, object]
        Raw timings/distributions, scalar-oracle parity, workload/source/native
        hashes and observed POSIX host metadata. JAX is labelled as its distinct
        nonzero-adjacency model on the positive zero-diagonal common domain.

    Raises
    ------
    ValueError
        For invalid controls or any missing required CPU owner.
    RuntimeError
        For numerical disagreement or source/artifact drift during measurement.
    ImportError
        If the actual JAX runtime is absent. POSIX host APIs and all five native
        installations are required for this native collection interface.
    """
    if not sizes:
        raise ValueError("sizes must not be empty")
    populations = [_count(n, "size", 3) for n in sizes]
    calls = _count(calls, "calls", 20)
    seed = _count(seed, "seed", 0)
    if isinstance(density, (bool, np.bool_)) or not isinstance(density, Real):
        raise ValueError("density must be a finite real number in [0,1]")
    density = float(density)
    if not np.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("density must be a finite real number in [0,1]")
    require_owners(OWNERS)
    import jax
    import jax.numpy as jnp

    from scpn_phase_orchestrator.nn.chimera import local_order_parameter as nn_local

    sources_before = _hashes()
    started = time.time()
    load_before = os.getloadavg()
    rows: list[dict[str, object]] = []
    for n in populations:
        rng = np.random.default_rng(seed + n)
        phases = np.asarray(rng.uniform(-np.pi, np.pi, n), dtype=np.float64)
        coupling = np.where(rng.random((n, n)) < density, 1.0, 0.0)
        np.fill_diagonal(coupling, 0.0)
        reference = scalar_local_order(phases, coupling)
        workload_sha256 = hashlib.sha256(
            phases.tobytes() + coupling.tobytes()
        ).hexdigest()
        for owner in OWNERS:
            tolerance = 1e-9 if owner == "mojo" else 1e-12
            operation = partial(
                chimera.local_order_parameter, phases, coupling, backend=owner
            )
            warm = operation()
            if not np.allclose(warm, reference, atol=tolerance, rtol=0):
                raise RuntimeError("Chimera comparison parity failed for " + owner)
            samples, output = _measure(operation, calls)
            actual = np.asarray(output)
            error = float(np.max(np.abs(actual - reference)))
            if error > tolerance:
                raise RuntimeError("Chimera measured output drift for " + owner)
            rows.append(
                dict(
                    owner=owner,
                    operation="monitor.local_order_parameter",
                    n=n,
                    model="positive non-self unweighted adjacency",
                    precision="float64",
                    max_abs_error=error,
                    tolerance=tolerance,
                    workload_sha256=workload_sha256,
                    **_statistics(samples),
                )
            )
        with jax.enable_x64():
            phases_jax = jnp.asarray(phases, dtype=jnp.float64)
            coupling_jax = jnp.asarray(coupling, dtype=jnp.float64)
            compiled = jax.jit(nn_local)

            def jax_operation(
                kernel: Callable[[jax.Array, jax.Array], jax.Array] = compiled,
                p: jax.Array = phases_jax,
                k: jax.Array = coupling_jax,
            ) -> object:
                """Complete the bound genuine JAX call before stopping its timer."""
                return kernel(p, k).block_until_ready()

            warm = np.asarray(jax_operation())
            if not np.allclose(warm, reference, atol=1e-12, rtol=0):
                raise RuntimeError("JAX common-domain scalar parity failed")
            samples, output = _measure(jax_operation, calls)
            error = float(np.max(np.abs(np.asarray(output) - reference)))
            if error > 1e-12:
                raise RuntimeError("JAX measured output drift")
            rows.append(
                dict(
                    owner="jax_nn",
                    operation="nn.local_order_parameter_jit",
                    n=n,
                    model=(
                        "nonzero signed/self adjacency; shared "
                        "positive zero-diagonal graph"
                    ),
                    precision="float64",
                    max_abs_error=error,
                    tolerance=1e-12,
                    workload_sha256=workload_sha256,
                    **_statistics(samples),
                )
            )
    sources_after = _hashes()
    if sources_before != sources_after:
        raise RuntimeError(
            "Chimera comparison source/artifact changed during measurement"
        )
    return {
        "schema": "chimera-public-comparison-v1",
        "source_hashes": sources_before,
        "sample_scope": "one public call; completed JAX JIT call excludes compilation",
        "calls": calls,
        "density": density,
        "seed": seed,
        "results": rows,
        "started_unix": started,
        "finished_unix": time.time(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "numpy": np.__version__,
            "jax": jax.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "load_before": load_before,
            "load_after": os.getloadavg(),
            "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution,
            "thread_controls": {
                name: os.getenv(name)
                for name in (
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "JULIA_NUM_THREADS",
                )
            },
        },
        "performance_scope": "shared-host diagnostics; controlled performance pending",
    }
