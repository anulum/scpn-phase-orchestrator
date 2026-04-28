# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera local order-parameter (Mojo port)

"""Kuramoto local order parameter per oscillator as a Mojo executable.

Stdin layout (single whitespace-separated line):

    CHI n phases[0..n] knm_flat[0..n*n]

Prints ``n`` f64 R_local values, one per line.

Build with::

    mojo build mojo/chimera.mojo -o mojo/chimera_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn local_order_parameter(
    phases: List[Float64],
    knm: List[Float64],
    n: Int,
    mut out: List[Float64],
) -> None:
    for i in range(n):
        var sr: Float64 = 0.0
        var si: Float64 = 0.0
        var cnt: Int = 0
        var theta_i = phases[i]
        var base = i * n
        for j in range(n):
            if knm[base + j] > 0.0:
                var delta = phases[j] - theta_i
                sr += cos(delta)
                si += sin(delta)
                cnt += 1
        if cnt == 0:
            out[i] = 0.0
        else:
            var inv = 1.0 / Float64(cnt)
            sr = sr * inv
            si = si * inv
            out[i] = sqrt(sr * sr + si * si)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "CHI":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(0.0)
    local_order_parameter(phases, knm, n, out)
    for i in range(n):
        print(out[i])
