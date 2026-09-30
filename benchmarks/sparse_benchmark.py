# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sparse UPDE engine benchmark

"""Compare real sparse Euler trajectories with an independent CSR equation reference.

Default workloads retain the original 1,000/10,000-node, 100-step cases. Timings
are local observations; the CLI records runtime provenance and raw repetitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from typing import TypedDict

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


class SparseMeasurement(TypedDict):
    """One actual runtime workload, equation residual, and raw timing samples."""

    n: int
    density: float
    edges: int
    steps: int
    repeats: int
    input_sha256: str
    kernel_available: bool
    python: str
    numpy: str
    scipy: str
    seconds: list[float]
    median_step_us: float
    order_parameter: float
    max_phase_error: float


def run_sparse_bench(
    n: int = 1000,
    density: float = 0.01,
    n_steps: int = 100,
    repeats: int = 3,
) -> SparseMeasurement:
    """Measure public sparse Euler calls and verify their final phase equation.

    The reference uses SciPy CSR matrix-vector products and the sine difference
    identity; the engine's CSR loop and native implementation are not called by
    the reference. Every repetition starts from the identical seeded input.

    Parameters
    ----------
    n : int
        Positive oscillator count.
    density : float
        Requested random entry count divided by ``n**2``; duplicate edges merge.
    n_steps : int
        Number of Euler steps of 0.01 seconds per repetition.
    repeats : int
        Number of freshly initialised timing repetitions.

    Returns
    -------
    SparseMeasurement
        Input fingerprint, runtime versions, raw timings and reference residual.

    Raises
    ------
    ValueError
        Counts are not positive or density is outside the finite unit interval.
    AssertionError
        An actual trajectory differs from the independent CSR equation or the
        supposedly fixed coupling values change.
    """
    if isinstance(n, bool) or n < 1:
        raise ValueError("n must be positive")
    if not np.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("density must be finite and inside [0, 1]")
    if isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be positive")
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(42)
    initial = rng.uniform(0.0, 2.0 * np.pi, n)
    omegas = np.ones(n, dtype=np.float64)
    num_entries = int(n * n * density)
    rows = rng.integers(0, n, num_entries)
    cols = rng.integers(0, n, num_entries)
    data = rng.uniform(0.0, 0.5 / n, num_entries)
    coupling = csr_matrix((data, (rows, cols)), shape=(n, n))
    coupling.eliminate_zeros()
    row_ptr = np.asarray(coupling.indptr, dtype=np.int64)
    indices = np.asarray(coupling.indices, dtype=np.int64)
    values = np.asarray(coupling.data, dtype=np.float64)
    original_values = values.copy()
    alpha = np.zeros(values.size, dtype=np.float64)
    digest = hashlib.sha256()
    for array in (initial, omegas, row_ptr, indices, values, alpha):
        digest.update(array.tobytes())
    reference = initial.copy()
    for _ in range(n_steps):
        sine = np.sin(reference)
        cosine = np.cos(reference)
        derivative = omegas + cosine * (coupling @ sine) - sine * (coupling @ cosine)
        reference = (reference + 0.01 * derivative) % (2.0 * np.pi)
    samples: list[float] = []
    max_error = 0.0
    result = initial.copy()
    for _ in range(repeats):
        engine = SparseUPDEEngine(n, dt=0.01, method="euler")
        result = initial.copy()
        started = time.perf_counter()
        for _ in range(n_steps):
            result = engine.step(
                result, omegas, row_ptr, indices, values, 0.0, 0.0, alpha
            )
        samples.append(time.perf_counter() - started)
        residual = np.angle(np.exp(1j * (result - reference)))
        np.testing.assert_allclose(residual, 0.0, atol=1e-11, rtol=0.0)
        np.testing.assert_array_equal(values, original_values)
        max_error = max(max_error, float(np.max(np.abs(residual))))
    try:
        from spo_kernel import PySparseUPDEStepper
    except ImportError:
        kernel_available = False
    else:
        kernel_available = callable(PySparseUPDEStepper)

    return {
        "n": n,
        "density": density,
        "edges": int(values.size),
        "steps": n_steps,
        "repeats": repeats,
        "input_sha256": digest.hexdigest(),
        "kernel_available": kernel_available,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seconds": samples,
        "median_step_us": float(np.median(samples)) * 1e6 / n_steps,
        "order_parameter": float(abs(np.mean(np.exp(1j * result)))),
        "max_phase_error": max_error,
    }


def main() -> None:
    """Print JSON measurements for the original workloads or one requested case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cases = (
        [(1000, 0.01), (10000, 0.001)] if args.n is None else [(args.n, args.density)]
    )
    measurements = [
        run_sparse_bench(n, density, args.steps, args.repeats) for n, density in cases
    ]
    print(json.dumps(measurements, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
