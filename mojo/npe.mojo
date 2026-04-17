# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Normalised Persistent Entropy (Mojo port)

"""NPE + phase distance matrix as a Mojo executable.

Stdin line layout (whitespace separated):

* ``PDM n phases[0..n]`` — prints ``n*n`` f64, one per line.
* ``NPE n max_radius phases[0..n]`` — prints single NPE value.

Build with::

    mojo build mojo/npe.mojo -o mojo/npe_mojo -Xlinker -lm
"""

from std.math import sin, cos, atan2, log, abs
from std.collections import List


fn phase_distance_matrix(
    phases: List[Float64],
    mut out: List[Float64],
    n: Int,
) -> None:
    for i in range(n):
        for j in range(n):
            var d = phases[i] - phases[j]
            var v = atan2(sin(d), cos(d))
            if v < 0:
                v = -v
            out[i * n + j] = v


fn find_root(parent: List[Int], x: Int) -> Int:
    # No path compression — Mojo 0.26 aliasing rules forbid reading and
    # writing the same list in one statement. Correctness identical to
    # the other backends, just slower asymptotically (still O(α(N))
    # amortised with union-by-rank alone).
    var cur = x
    while parent[cur] != cur:
        cur = parent[cur]
    return cur


fn compute_npe(
    phases: List[Float64], n: Int, max_radius: Float64
) -> Float64:
    if n < 2:
        return 0.0
    var pi_val = 3.141592653589793
    var radius = pi_val if max_radius < 0.0 else max_radius

    var dist = List[Float64](capacity=n * n)
    for _ in range(n * n):
        dist.append(0.0)
    phase_distance_matrix(phases, dist, n)

    # Build upper-triangle edge list with (d, i, j) tuples as parallel arrays.
    var n_edges = n * (n - 1) // 2
    var ed = List[Float64](capacity=n_edges)
    for _ in range(n_edges):
        ed.append(0.0)
    var ei = List[Int](capacity=n_edges)
    for _ in range(n_edges):
        ei.append(0)
    var ej = List[Int](capacity=n_edges)
    for _ in range(n_edges):
        ej.append(0)
    var k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            ed[k] = dist[i * n + j]
            ei[k] = i
            ej[k] = j
            k += 1

    # Simple insertion sort on edge distance (N small in practice; the
    # Rust / Go / Julia paths sort too — correctness matches).
    for a in range(1, n_edges):
        var key_d = ed[a]
        var key_i = ei[a]
        var key_j = ej[a]
        var b = a - 1
        while b >= 0 and ed[b] > key_d:
            ed[b + 1] = ed[b]
            ei[b + 1] = ei[b]
            ej[b + 1] = ej[b]
            b -= 1
        ed[b + 1] = key_d
        ei[b + 1] = key_i
        ej[b + 1] = key_j

    var parent = List[Int](capacity=n)
    for i in range(n):
        parent.append(i)
    var rank = List[Int](capacity=n)
    for _ in range(n):
        rank.append(0)

    var lifetimes = List[Float64]()
    for p in range(n_edges):
        if ed[p] > radius:
            break
        var ri = find_root(parent, ei[p])
        var rj = find_root(parent, ej[p])
        if ri != rj:
            lifetimes.append(ed[p])
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] = rank[ri] + 1

    if len(lifetimes) == 0:
        return 0.0
    var total: Float64 = 0.0
    for lt in lifetimes:
        total += lt
    if total < 1e-15:
        return 0.0
    var entropy: Float64 = 0.0
    var n_probs = 0
    for lt in lifetimes:
        var p = lt / total
        if p > 0.0:
            entropy -= p * log(p)
            n_probs += 1
    var max_entropy: Float64 = 1.0
    if n_probs > 1:
        max_entropy = log(Float64(n_probs))
    if max_entropy < 1e-15:
        return 0.0
    return entropy / max_entropy


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "PDM":
        var n = Int(atol(tokens[idx])); idx += 1
        var phases = List[Float64](capacity=n)
        for _ in range(n):
            phases.append(atof(tokens[idx])); idx += 1
        var out = List[Float64](capacity=n * n)
        for _ in range(n * n):
            out.append(0.0)
        phase_distance_matrix(phases, out, n)
        for v in out:
            print(v)
    elif op == "NPE":
        var n = Int(atol(tokens[idx])); idx += 1
        var max_radius = atof(tokens[idx]); idx += 1
        var phases = List[Float64](capacity=n)
        for _ in range(n):
            phases.append(atof(tokens[idx])); idx += 1
        print(compute_npe(phases, n, max_radius))
    else:
        print(-1)
