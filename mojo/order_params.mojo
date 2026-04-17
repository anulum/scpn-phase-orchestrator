# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameters (Mojo port)

"""Kuramoto order parameter + phase-locking value + layer coherence
as a Mojo executable. Text stdin / stdout protocol matches the
AttnRes Mojo bridge.

Stdin line layout (whitespace separated):

    op phases_len [payload...]

Where ``op`` is one of:

* ``R``   — compute ``(R, psi)``. Payload: ``phases[0..n]``.
          Output: two floats, one per line (``R`` then ``psi``).
* ``PLV`` — compute phase-locking value. Payload:
          ``phases_a[0..n] phases_b[0..n]``. Output: one float.
* ``LC``  — compute layer coherence. Payload:
          ``phases[0..n] n_idx indices[0..n_idx]``. Output: one float.

Build with::

    mojo build mojo/order_params.mojo -o mojo/order_params_mojo \\
        -Xlinker -lm
"""

from std.math import cos, sin, atan2, sqrt
from std.collections import List


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    var n = Int(atol(tokens[idx])); idx += 1
    var two_pi = 2.0 * 3.141592653589793

    if op == "R":
        if n == 0:
            print(0.0)
            print(0.0)
            return
        var sx: Float64 = 0.0
        var sy: Float64 = 0.0
        for _ in range(n):
            var p = atof(tokens[idx]); idx += 1
            sx += cos(p)
            sy += sin(p)
        sx /= Float64(n)
        sy /= Float64(n)
        var r = sqrt(sx * sx + sy * sy)
        var psi = atan2(sy, sx)
        if psi < 0:
            psi += two_pi
        print(r)
        print(psi)
    elif op == "PLV":
        if n == 0:
            print(0.0)
            return
        var pa = List[Float64](capacity=n)
        for _ in range(n):
            pa.append(atof(tokens[idx])); idx += 1
        var pb = List[Float64](capacity=n)
        for _ in range(n):
            pb.append(atof(tokens[idx])); idx += 1
        var sx: Float64 = 0.0
        var sy: Float64 = 0.0
        for i in range(n):
            var diff = pa[i] - pb[i]
            sx += cos(diff)
            sy += sin(diff)
        sx /= Float64(n)
        sy /= Float64(n)
        print(sqrt(sx * sx + sy * sy))
    elif op == "LC":
        # n here is phases_len.
        var phases = List[Float64](capacity=n)
        for _ in range(n):
            phases.append(atof(tokens[idx])); idx += 1
        var ni = Int(atol(tokens[idx])); idx += 1
        if ni == 0:
            print(0.0)
            return
        var sx: Float64 = 0.0
        var sy: Float64 = 0.0
        for _ in range(ni):
            var i = Int(atol(tokens[idx])); idx += 1
            sx += cos(phases[i])
            sy += sin(phases[i])
        sx /= Float64(ni)
        sy /= Float64(ni)
        print(sqrt(sx * sx + sy * sy))
    else:
        print(-1)
