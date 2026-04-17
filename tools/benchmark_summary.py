# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Python benchmark runner summary

"""Run the Python benchmark scripts under ``benchmarks/`` and print a
compact summary of their stdout and wall-clock time.

This is a convenience wrapper; the primary benchmark surface is the
criterion bench family in ``spo-kernel/crates/spo-engine/benches/``
which is run via ``cargo bench``. The Python scripts here are smaller
scaling probes that complement the Rust microbenchmarks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run_bench(args: list[str], env: dict[str, str]) -> tuple[str, float]:
    """Run a benchmark subprocess without a shell; return (stdout, seconds)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - start
        return f"Error (exit {exc.returncode}): {exc.stderr.strip()}", elapsed
    except FileNotFoundError as exc:
        elapsed = time.perf_counter() - start
        return f"Error: benchmark script not found: {exc}", elapsed
    return result.stdout.strip(), time.perf_counter() - start


def main() -> int:
    print("SCPN Phase Orchestrator — Optimisation Summary")
    print("=" * 50)

    benchmarks: list[tuple[str, list[str]]] = [
        ("UPDE (Dense, N=1000)", ["python", "benchmarks/scaling_benchmark.py"]),
        ("Sparse UPDE (N=1000, d=0.01)", ["python", "benchmarks/sparse_benchmark.py"]),
        ("Stuart-Landau (N=1000)", ["python", "benchmarks/sl_benchmark.py"]),
        ("Simplicial (N=1000)", ["python", "benchmarks/simplicial_benchmark.py"]),
        ("Hypergraph (N=1000)", ["python", "benchmarks/hypergraph_benchmark.py"]),
        ("Inertial (N=1000)", ["python", "benchmarks/inertial_benchmark.py"]),
        ("Delayed (N=1000, tau=10)", ["python", "benchmarks/delayed_benchmark.py"]),
        ("Swarmalator (N=500)", ["python", "benchmarks/swarmalator_benchmark.py"]),
        ("Recurrence (T=2000, d=10)", ["python", "benchmarks/recurrence_benchmark.py"]),
    ]

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else src_path
    )

    exit_code = 0
    for name, args in benchmarks:
        print(f"Running {name}...")
        output, _elapsed = run_bench(args, env)
        print(output)
        if output.startswith("Error"):
            exit_code = 1
        print("-" * 30)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
