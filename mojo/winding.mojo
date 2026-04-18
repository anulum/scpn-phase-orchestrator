# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding numbers (Mojo port)

"""Cumulative winding number per oscillator as a Mojo executable.

Stdin layout (single line):

    WIND T N phases_flat[0..T*N]

Prints ``N`` integer winding numbers, one per line.

Build with::

    mojo build mojo/winding.mojo -o mojo/winding_mojo -Xlinker -lm
"""

from std.math import floor
from std.collections import List


fn wrap(delta: Float64, two_pi: Float64, pi_val: Float64) -> Float64:
    var r = delta + pi_val
    r = r - Float64(Int(r / two_pi)) * two_pi
    if r < 0.0:
        r += two_pi
    return r - pi_val


fn winding_numbers(
    phases: List[Float64],
    t: Int,
    n: Int,
    mut out: List[Int],
) -> None:
    if t < 2:
        for i in range(n):
            out[i] = 0
        return
    var two_pi = 6.283185307179586
    var pi_val = 3.141592653589793
    var cumulative = List[Float64](capacity=n)
    for _ in range(n):
        cumulative.append(0.0)
    for step in range(1, t):
        var base_now = step * n
        var base_prev = (step - 1) * n
        for i in range(n):
            var delta = phases[base_now + i] - phases[base_prev + i]
            cumulative[i] = cumulative[i] + wrap(delta, two_pi, pi_val)
    for i in range(n):
        out[i] = Int(floor(cumulative[i] / two_pi))


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "WIND":
        print(-1)
        return

    var t = Int(atol(tokens[idx])); idx += 1
    var n = Int(atol(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=t * n)
    for _ in range(t * n):
        phases.append(atof(tokens[idx])); idx += 1
    var out = List[Int](capacity=n)
    for _ in range(n):
        out.append(0)
    winding_numbers(phases, t, n, out)
    for i in range(n):
        print(out[i])
