# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spatial coupling modulation (Mojo port)

"""Spatial coupling modulation as a Mojo executable.

Stdin:

    SPATIAL_MODULATE n dim form k_base exponent length epsilon knm_flat positions_flat

Prints n*n f64 values in row-major order.
"""

from std.math import exp, pow, sqrt
from std.collections import List


fn _weight(distance: Float64, k_base: Float64, form: Int, exponent: Float64, length: Float64, epsilon: Float64) -> Float64:
    if form == 0:
        return k_base / (1.0 + distance)
    if form == 1:
        return k_base * exp(-distance / length)
    if form == 2:
        return k_base * pow(1.0 + distance / length, -exponent)
    return k_base / sqrt(distance * distance + epsilon)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "SPATIAL_MODULATE":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var dim = Int(atol(tokens[idx])); idx += 1
    var form = Int(atol(tokens[idx])); idx += 1
    var k_base = atof(tokens[idx]); idx += 1
    var exponent = atof(tokens[idx]); idx += 1
    var length = atof(tokens[idx]); idx += 1
    var epsilon = atof(tokens[idx]); idx += 1

    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var positions = List[Float64](capacity=n * dim)
    for _ in range(n * dim):
        positions.append(atof(tokens[idx])); idx += 1

    for i in range(n):
        for j in range(n):
            if i == j:
                print(0.0)
                continue
            var d2: Float64 = 0.0
            for d in range(dim):
                var delta = positions[i * dim + d] - positions[j * dim + d]
                d2 += delta * delta
            var distance = sqrt(d2)
            var out = knm[i * n + j] * _weight(distance, k_base, form, exponent, length, epsilon)
            print(out)
