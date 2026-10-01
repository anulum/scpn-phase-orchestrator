# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Canonical phase projection runtime benchmark

"""Measure actual public UPDE and JAX consumers on identical torus cut cases.

Compiled float64 backends use their installed runtimes. JAX records its actual
device and precision, with compilation separated from repeated host readback.
The separate WebGPU benchmark records its real browser and adapter identity.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.nn.functional import (
    kuramoto_rk4_step,
    kuramoto_rk4_step_masked,
    kuramoto_step,
    kuramoto_step_masked,
)
from scpn_phase_orchestrator.upde import engine
from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

FloatArray = NDArray[np.float64]
ResultArray = NDArray[np.float32 | np.float64]


class PhaseMeasurement(TypedDict):
    """One executed public consumer, actual precision, and unisolated timings."""

    backend: str
    method: str
    precision: str
    period: float
    first_call_ms: float
    repeated_call_ms: list[float]
    output: list[float]
    input_sha256: str
    canonical_torus_passed: bool


def _problem(x64: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build mixed boundary/interior inputs in the consumer's target precision.

    Parameters
    ----------
    x64 : bool
        Select the binary64 period; otherwise use its binary32 representation.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Seven initial phases, frequencies, and a zero dense coupling/lag matrix.
    """
    dtype = np.float64 if x64 else np.float32
    period = dtype(2.0 * np.pi)
    interior = np.nextafter(period, dtype(0.0))
    phases = np.array([0.0, -period, -2.0 * period, -0.0, period, interior, 0.25])
    phases = phases.astype(np.float64)
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    return phases, omega, np.zeros((phases.size, phases.size))


def _measure(
    backend: str,
    method: str,
    phases: FloatArray,
    omega: FloatArray,
    matrix: FloatArray,
    execute: Callable[[], ResultArray],
    calls: int,
) -> PhaseMeasurement:
    """Time a real consumer and enforce its returned torus and buffer contract.

    Parameters
    ----------
    backend : str
        Actual runtime and public surface identifier.
    method : str
        Executed integration method.
    phases, omega, matrix : FloatArray
        Actual input arrays, whose bytes must remain unchanged.
    execute : Callable[[], ResultArray]
        Bound real public consumer; no producer is replaced.
    calls : int
        Positive repeated-call count after the first compilation/setup call.

    Returns
    -------
    PhaseMeasurement
        Raw durations, output, precision, and input fingerprint.

    Raises
    ------
    AssertionError
        Actual output violates projection, interior, evolution, or buffer rules.
    """
    initial = b"".join(array.tobytes() for array in (phases, omega, matrix))
    started = time.perf_counter()
    result = execute()
    first_ms = (time.perf_counter() - started) * 1000.0
    samples: list[float] = []
    for _ in range(calls):
        started = time.perf_counter()
        result = execute()
        samples.append((time.perf_counter() - started) * 1000.0)
    period = np.asarray(2.0 * np.pi, dtype=result.dtype)
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == np.nextafter(period, np.asarray(0.0, dtype=result.dtype))
    assert abs(float(result[6]) - 0.27) <= (
        3e-8 if result.dtype == np.float32 else 2e-16
    )
    assert np.all(np.isfinite(result) & (result >= 0.0) & (result < period))
    assert initial == b"".join(array.tobytes() for array in (phases, omega, matrix))
    return {
        "backend": backend,
        "method": method,
        "precision": str(result.dtype),
        "period": float(period),
        "first_call_ms": first_ms,
        "repeated_call_ms": samples,
        "output": result.tolist(),
        "input_sha256": sha256(initial).hexdigest(),
        "canonical_torus_passed": True,
    }


def _execute_functional(
    compiled: Callable[..., jax.Array],
    phases: jax.Array,
    omega: jax.Array,
    matrix: jax.Array,
    mask: jax.Array,
    masked: bool,
) -> ResultArray:
    """Read back one actual compiled dense or masked Kuramoto primitive.

    Parameters
    ----------
    compiled : Callable[..., jax.Array]
        Actual JIT-compiled public Kuramoto primitive.
    phases, omega, matrix, mask : jax.Array
        Actual device phase, frequency, coupling, and coupling-mask arrays.
    masked : bool
        Include the real coupling mask when invoking a masked primitive.

    Returns
    -------
    ResultArray
        Synchronous host readback in the actual configured JAX precision.
    """
    if masked:
        result = compiled(phases, omega, matrix, mask, 0.01)
    else:
        result = compiled(phases, omega, matrix, 0.01)
    output: ResultArray = np.asarray(result)
    return output


def benchmark_phase_wrapping(
    calls: int = 3, backends: Sequence[str] | None = None
) -> dict[str, object]:
    """Execute installed float64 backends and real dense/masked JAX primitives.

    Parameters
    ----------
    calls : int, default 3
        Positive repeated-call count per consumer after its first call.
    backends : Sequence[str] or None, default None
        Explicit installed stateless backends, or all actually available ones.
        JAX engines and functional primitives run in both actual precisions.

    Returns
    -------
    dict[str, object]
        Runtime versions/devices, actual installed backends, and measurements.

    Raises
    ------
    ValueError
        Counts are invalid or an explicitly requested backend is unavailable.
    AssertionError
        A real consumer violates the canonical torus or caller-buffer contract.

    Notes
    -----
    Each measured call integrates one 0.01-second outer step. Backend setup,
    JAX compilation, and repeated synchronous host readback are separate fields.
    Local shared-host measurements do not establish production speedup.
    """
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    selected = tuple(engine.AVAILABLE_BACKENDS if backends is None else backends)
    if not selected or any(name not in engine.AVAILABLE_BACKENDS for name in selected):
        raise ValueError("Every requested stateless backend must actually be available")
    if "webgpu" in selected:
        raise ValueError("Use the explicit real-browser WebGPU phase benchmark")
    phases, omega, matrix = _problem(True)
    rows: list[PhaseMeasurement] = []
    previous = engine.ACTIVE_BACKEND
    try:
        for backend in selected:
            engine.ACTIVE_BACKEND = backend
            for method in ("euler", "rk4", "rk45"):
                run = partial(
                    engine.upde_run,
                    phases,
                    omega,
                    matrix,
                    matrix,
                    0.0,
                    0.0,
                    0.01,
                    1,
                    method=method,
                )
                rows.append(
                    _measure(backend, method, phases, omega, matrix, run, calls)
                )
    finally:
        engine.ACTIVE_BACKEND = previous
    primitives: tuple[tuple[str, str, Callable[..., jax.Array]], ...] = (
        ("nn-dense", "euler", kuramoto_step),
        ("nn-dense", "rk4", kuramoto_rk4_step),
        ("nn-masked", "euler", kuramoto_step_masked),
        ("nn-masked", "rk4", kuramoto_rk4_step_masked),
    )
    for x64 in (False, True):
        with jax.enable_x64(x64):
            phases, omega, matrix = _problem(x64)
            for method in ("euler", "rk4"):
                public_engine = JaxUPDEEngine(phases.size, method=method)

                run_jax = partial(
                    public_engine.step, phases, omega, matrix, 0.0, 0.0, matrix
                )
                rows.append(
                    _measure("jax", method, phases, omega, matrix, run_jax, calls)
                )
            jp, jo, jm = jnp.asarray(phases), jnp.asarray(omega), jnp.asarray(matrix)
            mask = jnp.ones_like(jm)
            for name, method, primitive in primitives:
                compiled = jax.jit(primitive)

                run_functional = partial(
                    _execute_functional,
                    compiled,
                    jp,
                    jo,
                    jm,
                    mask,
                    name == "nn-masked",
                )

                rows.append(
                    _measure(name, method, phases, omega, matrix, run_functional, calls)
                )
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "installed_backends": list(engine.AVAILABLE_BACKENDS),
        "evidence_kind": "local_regression_non_isolated",
        "production_speedup_claim": False,
        "measurements": rows,
    }


def main() -> int:
    """Print or save fresh phase projection measurements from actual runtimes.

    Returns
    -------
    int
        Zero after every selected real consumer passes its semantic assertions.

    Notes
    -----
    Julia can mark the process's POSIX stdout descriptor nonblocking. Restore
    blocking output before emitting the complete JSON document, including when
    a caller captures stdout through a pipe. A real text sink without a file
    descriptor, such as ``io.StringIO``, receives the same complete document.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = benchmark_phase_wrapping(args.calls, args.backends)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    if os.name == "posix":
        try:
            stdout_fd = sys.stdout.fileno()
        except (OSError, ValueError):
            stdout_fd = None
        if stdout_fd is not None:
            os.set_blocking(stdout_fd, True)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
