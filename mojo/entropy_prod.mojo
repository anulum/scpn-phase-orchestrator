# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entropy production rate (Mojo port)

"""Overdamped-Kuramoto dissipation rate as a Mojo executable.

Stdin layout (single line):

    EP n alpha dt phases[0..n] omegas[0..n] knm_flat[0..n*n]

Prints the single f64 dissipation scalar.

Build with::

    mojo build mojo/entropy_prod.mojo -o mojo/entropy_prod_mojo -Xlinker -lm
"""

from std.math import sin
from std.collections import List


fn entropy_production_rate(
    phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    n: Int,
    alpha: Float64,
    dt: Float64,
) -> Float64:
    if n == 0 or dt <= 0.0:
        return 0.0
    var inv_n = alpha / Float64(n)
    var acc: Float64 = 0.0
    for i in range(n):
        var s: Float64 = 0.0
        var offset = i * n
        for j in range(n):
            s += knm[offset + j] * sin(phases[j] - phases[i])
        var d = omegas[i] + inv_n * s
        acc += d * d
    return acc * dt


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "EP":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var alpha = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1

    var result = entropy_production_rate(phases, omegas, knm, n, alpha, dt)
    print(result)
