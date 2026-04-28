# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Steady-state R (Mojo port)

"""One-trial Kuramoto steady-state R as a Mojo executable.

Stdin:

    STEADY n k_scale dt n_transient n_measure
           phases[0..n] omegas[0..n]
           knm[0..n*n] alpha[0..n*n]

Prints a single Float64 ``R`` on stdout.

Build with::

    mojo build mojo/basin_stability.mojo -o mojo/basin_stability_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn _kuramoto_step(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm_flat: List[Float64],
    alpha_flat: List[Float64],
    n: Int,
    k_scale: Float64,
    dt: Float64,
) -> None:
    var old = List[Float64](capacity=n)
    for i in range(n):
        old.append(phases[i])
    for i in range(n):
        var coupling: Float64 = 0.0
        var base = i * n
        var theta_i = old[i]
        for j in range(n):
            var k_ij = knm_flat[base + j] * k_scale
            var abs_k = k_ij
            if abs_k < 0.0:
                abs_k = -abs_k
            if abs_k < 1e-30:
                continue
            var a_ij = alpha_flat[base + j]
            coupling += k_ij * sin(old[j] - theta_i - a_ij)
        phases[i] = theta_i + dt * (omegas[i] + coupling)


fn _order_parameter(phases: List[Float64], n: Int) -> Float64:
    if n == 0:
        return 0.0
    var nn = Float64(n)
    var sum_cos: Float64 = 0.0
    var sum_sin: Float64 = 0.0
    for i in range(n):
        sum_cos += cos(phases[i])
        sum_sin += sin(phases[i])
    var c = sum_cos / nn
    var s = sum_sin / nn
    return sqrt(c * c + s * s)


fn steady_state_r(
    phases_init: List[Float64],
    omegas: List[Float64],
    knm_flat: List[Float64],
    alpha_flat: List[Float64],
    n: Int,
    k_scale: Float64,
    dt: Float64,
    n_transient: Int,
    n_measure: Int,
) -> Float64:
    var phases = List[Float64](capacity=n)
    for i in range(n):
        phases.append(phases_init[i])
    for _ in range(n_transient):
        _kuramoto_step(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
    var r_sum: Float64 = 0.0
    for _ in range(n_measure):
        _kuramoto_step(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
        r_sum += _order_parameter(phases, n)
    if n_measure == 0:
        return 0.0
    return r_sum / Float64(n_measure)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "STEADY":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var k_scale = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_transient = Int(atol(tokens[idx])); idx += 1
    var n_measure = Int(atol(tokens[idx])); idx += 1

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var alpha = List[Float64](capacity=n * n)
    for _ in range(n * n):
        alpha.append(atof(tokens[idx])); idx += 1

    var r = steady_state_r(
        phases, omegas, knm, alpha,
        n, k_scale, dt, n_transient, n_measure,
    )
    print(r)
