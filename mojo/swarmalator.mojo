# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator stepper (Mojo port)

"""One swarmalator step as a Mojo executable.

Stdin:

    STEP n dim a b j k dt
         pos[0..n*dim] phases[0..n] omegas[0..n]

Prints ``n*dim`` new_pos floats, then ``n`` new_phases floats
(one per line).

Build with::

    mojo build mojo/swarmalator.mojo -o mojo/swarmalator_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn swarmalator_step(
    pos: List[Float64],
    phases: List[Float64],
    omegas: List[Float64],
    n: Int,
    dim: Int,
    a: Float64, b: Float64, j: Float64, k: Float64, dt: Float64,
    mut new_pos: List[Float64],
    mut new_phases: List[Float64],
) -> None:
    var two_pi = 6.283185307179586
    var inv_n = 1.0 / Float64(n)
    for i in range(n * dim):
        new_pos[i] = pos[i]
    for i in range(n):
        var vel = List[Float64](capacity=dim)
        for _ in range(dim):
            vel.append(0.0)
        var phase_acc: Float64 = 0.0
        var base_i = i * dim
        var theta_i = phases[i]
        for m in range(n):
            var base_m = m * dim
            var s: Float64 = 0.0
            for d in range(dim):
                var delta = pos[base_m + d] - pos[base_i + d]
                s += delta * delta
            var dist = sqrt(s + 1e-6)
            var cos_d = cos(phases[m] - theta_i)
            var sin_d = sin(phases[m] - theta_i)
            var attract = (a + j * cos_d) / dist
            # Rust canonical: b / (dist * d2 + eps).
            var repulse = b / (dist * s + 1e-6)
            var factor = attract - repulse
            for d in range(dim):
                var delta = pos[base_m + d] - pos[base_i + d]
                vel[d] += delta * factor
            phase_acc += sin_d / dist
        for d in range(dim):
            new_pos[base_i + d] = pos[base_i + d] + dt * vel[d] * inv_n
        var dth = omegas[i] + k * phase_acc * inv_n
        var x = theta_i + dt * dth
        var v = x - Float64(Int(x / two_pi)) * two_pi
        if v < 0.0:
            v += two_pi
        new_phases[i] = v


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "STEP":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var dim = Int(atol(tokens[idx])); idx += 1
    var a = atof(tokens[idx]); idx += 1
    var b = atof(tokens[idx]); idx += 1
    var j = atof(tokens[idx]); idx += 1
    var k = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1

    var pos = List[Float64](capacity=n * dim)
    for _ in range(n * dim):
        pos.append(atof(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1

    var new_pos = List[Float64](capacity=n * dim)
    var new_phases = List[Float64](capacity=n)
    for _ in range(n * dim):
        new_pos.append(0.0)
    for _ in range(n):
        new_phases.append(0.0)
    swarmalator_step(
        pos, phases, omegas, n, dim, a, b, j, k, dt,
        new_pos, new_phases,
    )
    for i in range(n * dim):
        print(new_pos[i])
    for i in range(n):
        print(new_phases[i])
