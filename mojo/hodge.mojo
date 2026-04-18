# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition (Mojo port)

"""Hodge decomposition as a Mojo executable.

Stdin:

    HODGE n knm_flat[0..n*n] phases[0..n]

Prints ``3 * n`` f64 values: first ``n`` gradient, then ``n`` curl,
then ``n`` harmonic.

Build with::

    mojo build mojo/hodge.mojo -o mojo/hodge_mojo -Xlinker -lm
"""

from std.math import cos
from std.collections import List


fn hodge_decomposition(
    knm: List[Float64],
    phases: List[Float64],
    n: Int,
    mut gradient: List[Float64],
    mut curl: List[Float64],
    mut harmonic: List[Float64],
) -> None:
    for i in range(n):
        var g: Float64 = 0.0
        var c: Float64 = 0.0
        var t: Float64 = 0.0
        var theta_i = phases[i]
        var base_i = i * n
        for j in range(n):
            var kij = knm[base_i + j]
            var kji = knm[j * n + i]
            var cd = cos(phases[j] - theta_i)
            var sym = 0.5 * (kij + kji)
            var anti = 0.5 * (kij - kji)
            g += sym * cd
            c += anti * cd
            t += kij * cd
        gradient[i] = g
        curl[i] = c
        harmonic[i] = t - g - c


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "HODGE":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var gradient = List[Float64](capacity=n)
    var curl = List[Float64](capacity=n)
    var harmonic = List[Float64](capacity=n)
    for _ in range(n):
        gradient.append(0.0)
        curl.append(0.0)
        harmonic.append(0.0)
    hodge_decomposition(knm, phases, n, gradient, curl, harmonic)
    for k in range(n):
        print(gradient[k])
    for k in range(n):
        print(curl[k])
    for k in range(n):
        print(harmonic[k])
