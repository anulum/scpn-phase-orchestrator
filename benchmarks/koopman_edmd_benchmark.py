# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD multi-backend benchmark + parity gate

"""Timing and cross-backend parity gate for the Koopman EDMD-with-control solve.

Each declared language backend solves the same lifted snapshot least squares;
the gate records its wall-clock time and the maximum absolute deviation of its
``(A, B, C)`` from the NumPy reference, and fails if any present backend exceeds
the parity tolerance.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import koopman_edmd as ke

FloatArray = NDArray[np.float64]

_PARITY_TOLERANCE = 1.0e-9


def _make_snapshots(
    *, samples: int, lift_dim: int, input_dim: int, state_dim: int, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    x_lift = rng.standard_normal((samples, lift_dim))
    inputs = rng.standard_normal((samples, input_dim))
    y_lift = rng.standard_normal((samples, lift_dim))
    states = rng.standard_normal((samples, state_dim))
    return x_lift, inputs, y_lift, states


def _backend_solver(backend: str) -> object | None:
    if backend == "python":
        return ke._edmd_solve_reference
    try:
        return ke._load_backend(backend)["edmd_solve"]
    except (ImportError, RuntimeError, OSError, KeyError):
        return None


def _max_deviation(
    produced: tuple[FloatArray, FloatArray, FloatArray],
    reference: tuple[FloatArray, FloatArray, FloatArray],
) -> float:
    return max(
        float(np.max(np.abs(got - expected)))
        for got, expected in zip(produced, reference, strict=True)
    )


def benchmark_koopman_edmd_polyglot_parity_gate(
    *,
    samples: int = 256,
    lift_dim: int = 8,
    input_dim: int = 2,
    state_dim: int = 3,
    regularisation: float = 1.0e-8,
    calls: int = 5,
    seed: int = 2026,
) -> dict[str, object]:
    """Record EDMD solve timing and parity across every language backend slot."""
    snapshots = _make_snapshots(
        samples=samples,
        lift_dim=lift_dim,
        input_dim=input_dim,
        state_dim=state_dim,
        seed=seed,
    )
    reference = ke._edmd_solve_reference(*snapshots, regularisation)

    records: list[dict[str, object]] = []
    for backend in ke._BACKEND_NAMES:
        solver = _backend_solver(backend)
        if solver is None:
            records.append({"backend": backend, "available": False})
            continue
        start = time.perf_counter()
        produced: tuple[FloatArray, FloatArray, FloatArray] = (np.empty(0),) * 3
        for _ in range(calls):
            produced = solver(*snapshots, regularisation)  # type: ignore[operator]
        elapsed = (time.perf_counter() - start) / calls
        deviation = _max_deviation(produced, reference)
        records.append(
            {
                "backend": backend,
                "available": True,
                "seconds_per_call": elapsed,
                "max_abs_deviation": deviation,
                "within_tolerance": deviation <= _PARITY_TOLERANCE,
            }
        )

    present = [record for record in records if record.get("available")]
    parity_ok = all(record["within_tolerance"] for record in present)
    return {
        "suite": "koopman_edmd_polyglot_parity_gate",
        "boundary_contract": "exact_numpy_reference_validated",
        "parity_tolerance": _PARITY_TOLERANCE,
        "parity_ok": parity_ok,
        "backend_records": records,
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()
    result = benchmark_koopman_edmd_polyglot_parity_gate(calls=args.calls)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["parity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
