# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hypergraph k-body Kuramoto (Mojo port)

"""Generalised k-body hypergraph Kuramoto as a Mojo executable.

Stdin:

    HGRUN n n_edge_nodes n_edges knm_len alpha_len zeta psi dt n_steps
          phases[0..n] omegas[0..n]
          edge_nodes[0..n_edge_nodes]
          edge_offsets[0..n_edges]
          edge_strengths[0..n_edges]
          knm[0..knm_len]
          alpha[0..alpha_len]

Prints ``n`` final-phase floats on stdout (one per line).

Build with::

    mojo build mojo/hypergraph.mojo -o mojo/hypergraph_mojo -Xlinker -lm
"""

from std.math import sin, cos
from std.collections import List


fn _compute_derivative(
    theta: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    edge_nodes: List[Int],
    edge_offsets: List[Int],
    edge_strengths: List[Float64],
    zeta: Float64,
    psi: Float64,
    mut deriv: List[Float64],
) -> None:
    var sin_th = List[Float64](capacity=n)
    var cos_th = List[Float64](capacity=n)
    for i in range(n):
        sin_th.append(sin(theta[i]))
        cos_th.append(cos(theta[i]))
    var has_pairwise = (len(knm) == n * n)
    var alpha_zero = True
    for a in alpha:
        if a != 0.0:
            alpha_zero = False
            break
    var zs_psi: Float64 = 0.0
    var zc_psi: Float64 = 0.0
    if zeta != 0.0:
        zs_psi = zeta * sin(psi)
        zc_psi = zeta * cos(psi)

    for i in range(n):
        var pw: Float64 = 0.0
        if has_pairwise:
            var offset = i * n
            var ci = cos_th[i]
            var si = sin_th[i]
            if alpha_zero:
                for j in range(n):
                    pw += knm[offset + j] * (sin_th[j] * ci - cos_th[j] * si)
            else:
                for j in range(n):
                    pw += knm[offset + j] * \
                        sin(theta[j] - theta[i] - alpha[offset + j])
        deriv[i] = omegas[i] + pw
        if zeta != 0.0:
            deriv[i] += zs_psi * cos_th[i] - zc_psi * sin_th[i]

    var n_edges = len(edge_offsets)
    for e in range(n_edges):
        var start = edge_offsets[e]
        var stop = len(edge_nodes)
        if e + 1 < n_edges:
            stop = edge_offsets[e + 1]
        var k = stop - start
        var phase_sum: Float64 = 0.0
        for p in range(start, stop):
            phase_sum += theta[edge_nodes[p]]
        var sigma = edge_strengths[e]
        for p in range(start, stop):
            var m = edge_nodes[p]
            deriv[m] += sigma * sin(phase_sum - Float64(k) * theta[m])


fn hypergraph_run(
    phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    edge_nodes: List[Int],
    edge_offsets: List[Int],
    edge_strengths: List[Float64],
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    n_steps: Int,
    mut out: List[Float64],
) -> None:
    var two_pi = 6.283185307179586
    for i in range(n):
        out[i] = phases[i]
    var deriv = List[Float64](capacity=n)
    for _ in range(n):
        deriv.append(0.0)
    for _ in range(n_steps):
        _compute_derivative(out, omegas, knm, alpha, n,
                            edge_nodes, edge_offsets, edge_strengths,
                            zeta, psi, deriv)
        for i in range(n):
            var raw = out[i] + dt * deriv[i]
            var v = raw - Float64(Int(raw / two_pi)) * two_pi
            if v < 0.0:
                v += two_pi
            out[i] = v


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "HGRUN":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var n_edge_nodes = Int(atol(tokens[idx])); idx += 1
    var n_edges = Int(atol(tokens[idx])); idx += 1
    var knm_len = Int(atol(tokens[idx])); idx += 1
    var alpha_len = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var edge_nodes = List[Int](capacity=n_edge_nodes)
    for _ in range(n_edge_nodes):
        edge_nodes.append(Int(atol(tokens[idx]))); idx += 1
    var edge_offsets = List[Int](capacity=n_edges)
    for _ in range(n_edges):
        edge_offsets.append(Int(atol(tokens[idx]))); idx += 1
    var edge_strengths = List[Float64](capacity=n_edges)
    for _ in range(n_edges):
        edge_strengths.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=knm_len)
    for _ in range(knm_len):
        knm.append(atof(tokens[idx])); idx += 1
    var alpha = List[Float64](capacity=alpha_len)
    for _ in range(alpha_len):
        alpha.append(atof(tokens[idx])); idx += 1

    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(0.0)
    hypergraph_run(phases, omegas, knm, alpha, n,
                   edge_nodes, edge_offsets, edge_strengths,
                   zeta, psi, dt, n_steps, out)
    for i in range(n):
        print(out[i])
