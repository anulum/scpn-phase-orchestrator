# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sheaf runtime diagnostics

"""Measure real sheaf trajectories against an independent vector ODE reference.

Timings are shared-host local diagnostics, without an acceleration or latency claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from numbers import Integral
from typing import TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from scpn_phase_orchestrator import SheafUPDEEngine


class SheafMeasurement(TypedDict):
    """One seeded actual-runtime trajectory and repeated timing observations."""

    n: int
    d: int
    steps: int
    repeats: int
    method: str
    dt: float
    input_sha256: str
    kernel_available: bool
    python: str
    numpy: str
    scipy: str
    seconds: list[float]
    median_step_us: float
    max_phase_error: float
    last_dt: float
    diagnostic_scope: str


def run_sheaf_bench(
    n: int = 8,
    d: int = 3,
    n_steps: int = 20,
    repeats: int = 3,
    method: str = "rk45",
) -> SheafMeasurement:
    """Run a public sheaf batch and check its anisotropic forced ODE trajectory.

    Parameters
    ----------
    n, d : int
        Positive oscillator count and dimension per phase vector.
    n_steps : int
        Positive number of complete intervals of 0.01 seconds.
    repeats : int
        Positive number of fresh solver timing repetitions.
    method : str
        Euler, RK4 or adaptive Dormand-Prince RK45.

    Returns
    -------
    SheafMeasurement
        Runtime, input fingerprint, equation residual, proposal and raw timings.

    Raises
    ------
    ValueError
        A count or solver method is invalid.
    AssertionError
        A real trajectory violates its method's accuracy budget or mutates input.
    """
    for name, count in (("n", n), ("d", d), ("n_steps", n_steps), ("repeats", repeats)):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"{name} must be a positive non-boolean integer")
    if method not in ("euler", "rk4", "rk45"):
        raise ValueError("method must be euler, rk4 or rk45")
    dt = 0.01
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.1, 1.2, (n, d))
    omegas = rng.uniform(-0.2, 0.4, (n, d))
    maps = rng.uniform(-0.1, 0.2, (n, n, d, d)) / (n * d)
    psi = rng.uniform(0.4, 0.8, d)
    zeta = 0.2
    digest = hashlib.sha256()
    inputs = (phases, omegas, maps, psi)
    snapshots = tuple(value.copy() for value in inputs)
    for value in inputs:
        digest.update(value.tobytes())

    def derivative(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the documented tensor equation independently of either solver."""
        theta = state.reshape(n, d)
        differences = theta[None, :, None, :] - theta[:, None, :, None]
        coupling = np.einsum("ijdk,ijdk->id", maps, np.sin(differences))
        return np.asarray(omegas + coupling + zeta * np.sin(psi - theta)).ravel()

    reference = solve_ivp(
        derivative,
        (0.0, dt * n_steps),
        phases.ravel(),
        method="DOP853",
        atol=1e-13,
        rtol=1e-13,
    )
    if not reference.success:
        raise ValueError(
            f"independent reference integration failed: {reference.message}"
        )
    expected = reference.y[:, -1].reshape(n, d)
    samples: list[float] = []
    max_error = 0.0
    budget = {"euler": 0.003, "rk4": 1e-9, "rk45": 1e-9}[method]
    for _ in range(repeats):
        engine = SheafUPDEEngine(n, d, dt, method=method, atol=1e-10, rtol=1e-10)
        started = time.perf_counter()
        result = engine.run(phases, omegas, maps, zeta, psi, n_steps)
        samples.append(time.perf_counter() - started)
        residual = np.angle(np.exp(1j * (result - expected)))
        np.testing.assert_allclose(residual, 0.0, atol=budget, rtol=0)
        max_error = max(max_error, float(np.max(np.abs(residual))))
        for value, saved in zip(inputs, snapshots, strict=True):
            np.testing.assert_array_equal(value, saved)
    # This reports the actual instance selected by the public constructor.
    kernel_available = engine._rust is not None
    return {
        "n": n,
        "d": d,
        "steps": n_steps,
        "repeats": repeats,
        "method": method,
        "dt": dt,
        "input_sha256": digest.hexdigest(),
        "kernel_available": kernel_available,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seconds": samples,
        "median_step_us": float(np.median(samples)) * 1e6 / n_steps,
        "max_phase_error": max_error,
        "last_dt": engine.last_dt,
        "diagnostic_scope": (
            "shared-host functional and timing diagnostic; no production latency claim"
        ),
    }


def main() -> None:
    """Print strict JSON diagnostics for all maintained sheaf solver methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--method", choices=("euler", "rk4", "rk45"))
    args = parser.parse_args()
    methods = [args.method] if args.method else ["euler", "rk4", "rk45"]
    records = [
        run_sheaf_bench(args.n, args.d, args.steps, args.repeats, method)
        for method in methods
    ]
    print(json.dumps(records, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
