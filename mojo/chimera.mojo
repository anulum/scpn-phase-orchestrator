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
) raises -> None:
    """Admit finite exact buffers, then measure positive non-self neighbours."""
    if n < 0 or len(phases) != n:
        raise Error("phases length must match nonnegative n")
    if n > 0 and n > len(knm) // n:
        raise Error("matrix dimensions exceed actual buffer")
    if len(knm) != n * n or len(out) != n:
        raise Error("matrix/output lengths must match n")
    for value in phases:
        require_finite(value)
    for value in knm:
        require_finite(value)
    for i in range(n):
        if abs(knm[i * n + i]) > 1e-15:
            raise Error("knm self-coupling diagonal must be zero")
    for i in range(n):
        var sr: Float64 = 0.0
        var si: Float64 = 0.0
        var cnt: Int = 0
        var base = i * n
        for j in range(n):
            if j != i and knm[base + j] > 0.0:
                sr += cos(phases[j])
                si += sin(phases[j])
                cnt += 1
        if cnt == 0:
            out[i] = 0.0
        else:
            var inv = 1.0 / Float64(cnt)
            sr = sr * inv
            si = si * inv
            out[i] = min(sqrt(sr * sr + si * si), Float64(1.0))


fn main() raises:
    """Accept one exact CHI request and emit n finite scalar rows."""
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    if len(tokens) < 2:
        raise Error("incomplete CHI header")
    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "CHI":
        raise Error("unknown operation; expected CHI")

    var n = Int(atol(tokens[idx])); idx += 1
    if n < 0 or String(n) != tokens[1]:
        raise Error("n must be a canonical nonnegative integer")
    var remaining = len(tokens) - idx
    if n > remaining or (n > 0 and n > remaining // n):
        raise Error("dimensions exceed request length")
    if remaining != n + n * n:
        raise Error("CHI buffer cardinality mismatch")
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


fn require_finite(value: Float64) raises:
    """Refuse nonfinite measurements before calculation or output."""
    if not (value <= Float64.MAX_FINITE and value >= Float64.MIN_FINITE):
        raise Error("CHI measurements must be finite")
