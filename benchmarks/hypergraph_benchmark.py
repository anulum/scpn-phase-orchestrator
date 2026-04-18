# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hypergraph multi-backend benchmark

"""Per-backend wall-clock benchmark for
``upde.hypergraph.HypergraphEngine.run`` (mixed pairwise + k-body
coupling)."""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import hypergraph as h_mod
from scpn_phase_orchestrator.upde.hypergraph import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    HypergraphEngine,
)


def _make_problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    omega = rng.normal(1.0, 0.2, n)
    knm = rng.uniform(0, 0.3 / n, (n, n))
    np.fill_diagonal(knm, 0.0)
    all_triples = list(combinations(range(n), 3))
    max_edges = min(2 * n, len(all_triples))
    idx = rng.choice(len(all_triples), size=max_edges, replace=False)
    edges = [all_triples[i] for i in idx]
    return theta, omega, knm, edges


def _bench(backend: str, theta, omega, knm, edges, n: int,
           n_steps: int, calls: int) -> float:
    saved = h_mod.ACTIVE_BACKEND
    try:
        h_mod.ACTIVE_BACKEND = backend
        eng = HypergraphEngine(n, 0.01)
        for nodes in edges:
            eng.add_edge(nodes, strength=0.1)
        eng.run(theta, omega, n_steps=1, pairwise_knm=knm)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            eng.run(theta, omega, n_steps=n_steps, pairwise_knm=knm)
        return time.perf_counter() - t0
    finally:
        h_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, n_steps: int, calls: int) -> dict:
    theta, omega, knm, edges = _make_problem(n)
    row: dict = {
        "N": n, "n_steps": n_steps, "calls": calls,
        "n_edges": len(edges), "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, theta, omega, knm, edges, n, n_steps, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'nsteps':>7} {'E':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.n_steps, args.calls)
        results.append(row)
        line = (f"{n:>4} {args.n_steps:>7} {row['n_edges']:>4} "
                f"{args.calls:>6}")
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_ms_per_call']:>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
