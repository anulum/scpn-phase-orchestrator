# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence matrix kernels (Mojo port)

"""Recurrence and cross-recurrence matrices as a Mojo executable.

Stdin verbs:

* ``REC T d angular epsilon traj[0..T*d]`` — single-trajectory
  recurrence matrix.
* ``CROSS T d angular epsilon traj_a[0..T*d] traj_b[0..T*d]`` —
  cross-recurrence matrix.

Prints ``T*T`` int entries (0 / 1) row-major, one per line.

Build with::

    mojo build mojo/recurrence.mojo -o mojo/recurrence_mojo -Xlinker -lm
"""

from std.math import sin
from std.collections import List


fn squared_distance(
    a: List[Float64],
    b: List[Float64],
    ia: Int,
    ib: Int,
    d: Int,
    angular: Bool,
) -> Float64:
    var s: Float64 = 0.0
    if angular:
        for k in range(d):
            var delta = a[ia * d + k] - b[ib * d + k]
            var c = 2.0 * sin(delta / 2.0)
            s += c * c
    else:
        for k in range(d):
            var delta = a[ia * d + k] - b[ib * d + k]
            s += delta * delta
    return s


fn fill_recurrence(
    a: List[Float64],
    b: List[Float64],
    t: Int,
    d: Int,
    epsilon: Float64,
    angular: Bool,
    mut out: List[Int],
) -> None:
    var eps_sq = epsilon * epsilon
    for i in range(t):
        for j in range(t):
            if squared_distance(a, b, i, j, d, angular) <= eps_sq:
                out[i * t + j] = 1
            else:
                out[i * t + j] = 0


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "REC":
        var t = Int(atol(tokens[idx])); idx += 1
        var d = Int(atol(tokens[idx])); idx += 1
        var ang = Int(atol(tokens[idx])); idx += 1
        var epsilon = atof(tokens[idx]); idx += 1
        var traj = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj.append(atof(tokens[idx])); idx += 1
        var out = List[Int](capacity=t * t)
        for _ in range(t * t):
            out.append(0)
        fill_recurrence(traj, traj, t, d, epsilon, ang != 0, out)
        for k in range(t * t):
            print(out[k])

    elif op == "CROSS":
        var t = Int(atol(tokens[idx])); idx += 1
        var d = Int(atol(tokens[idx])); idx += 1
        var ang = Int(atol(tokens[idx])); idx += 1
        var epsilon = atof(tokens[idx]); idx += 1
        var traj_a = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj_a.append(atof(tokens[idx])); idx += 1
        var traj_b = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj_b.append(atof(tokens[idx])); idx += 1
        var out = List[Int](capacity=t * t)
        for _ in range(t * t):
            out.append(0)
        fill_recurrence(traj_a, traj_b, t, d, epsilon, ang != 0, out)
        for k in range(t * t):
            print(out[k])

    else:
        print(-1)
