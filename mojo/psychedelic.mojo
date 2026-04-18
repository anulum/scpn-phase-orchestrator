# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic observables (Mojo port)

"""Circular-phase Shannon entropy as a Mojo executable.

Stdin layout:

    ENT n n_bins phases[0..n]

Prints one f64 entropy value in nats.

Build with::

    mojo build mojo/psychedelic.mojo -o mojo/psychedelic_mojo -Xlinker -lm
"""

from std.math import log, floor
from std.collections import List


fn entropy_from_phases(
    phases: List[Float64],
    n_bins: Int,
) -> Float64:
    var t = len(phases)
    if t == 0:
        return 0.0
    var two_pi = 6.283185307179586
    var counts = List[Int](capacity=n_bins)
    for _ in range(n_bins):
        counts.append(0)
    var bin_width = two_pi / Float64(n_bins)
    for i in range(t):
        var x = phases[i]
        var v = x - Float64(Int(x / two_pi)) * two_pi
        if v < 0.0:
            v += two_pi
        var bx = Int(floor(v / bin_width))
        if bx >= n_bins:
            bx = n_bins - 1
        counts[bx] = counts[bx] + 1
    var total = Float64(t)
    var h: Float64 = 0.0
    for k in range(n_bins):
        var c = counts[k]
        if c > 0:
            var p = Float64(c) / total
            h = h - p * log(p)
    return h


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "ENT":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var n_bins = Int(atol(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    print(entropy_from_phases(phases, n_bins))
