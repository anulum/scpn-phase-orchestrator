# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension kernels (Mojo port)

"""Grassberger-Procaccia correlation integral + Kaplan-Yorke
dimension as a Mojo executable.

Stdin verbs:

* ``CI T d nP nK idx_i[0..nP] idx_j[0..nP] epsilons[0..nK]
     traj_flat[0..T*d]`` — prints ``nK`` correlation fractions.
* ``KY n lambda[0..n]`` — prints the Kaplan-Yorke scalar.

Build with::

    mojo build mojo/dimension.mojo -o mojo/dimension_mojo -Xlinker -lm
"""

from std.math import sqrt, abs
from std.collections import List


fn correlation_integral(
    traj: List[Float64],
    t: Int,
    d: Int,
    idx_i: List[Int],
    idx_j: List[Int],
    epsilons: List[Float64],
    mut out: List[Float64],
) -> None:
    var n_p = len(idx_i)
    var n_k = len(epsilons)
    if n_p == 0:
        for k in range(n_k):
            out[k] = 0.0
        return
    var dists = List[Float64](capacity=n_p)
    for _ in range(n_p):
        dists.append(0.0)
    for p in range(n_p):
        var i = idx_i[p]
        var j = idx_j[p]
        var s: Float64 = 0.0
        var base_i = i * d
        var base_j = j * d
        for k in range(d):
            var delta = traj[base_i + k] - traj[base_j + k]
            s += delta * delta
        dists[p] = sqrt(s)
    var inv_p = 1.0 / Float64(n_p)
    for k in range(n_k):
        var cnt = 0
        var eps = epsilons[k]
        for p in range(n_p):
            if dists[p] < eps:
                cnt += 1
        out[k] = Float64(cnt) * inv_p


fn sort_descending(mut xs: List[Float64], n: Int) -> None:
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] < key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


fn kaplan_yorke_dimension(lyap: List[Float64]) -> Float64:
    var n = len(lyap)
    if n == 0:
        return 0.0
    var le = List[Float64](capacity=n)
    for i in range(n):
        le.append(lyap[i])
    sort_descending(le, n)
    var cumsum: Float64 = 0.0
    var j = -1
    for i in range(n):
        cumsum += le[i]
        if cumsum >= 0.0:
            j = i
        else:
            break
    if j == -1:
        return 0.0
    if j >= n - 1:
        return Float64(n)
    var denom = abs(le[j + 1])
    if denom == 0.0:
        return Float64(j + 1)
    var s_j: Float64 = 0.0
    for i in range(j + 1):
        s_j += le[i]
    return Float64(j + 1) + s_j / denom


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "CI":
        var t = Int(atol(tokens[idx])); idx += 1
        var d = Int(atol(tokens[idx])); idx += 1
        var n_p = Int(atol(tokens[idx])); idx += 1
        var n_k = Int(atol(tokens[idx])); idx += 1
        var idx_i = List[Int](capacity=n_p)
        for _ in range(n_p):
            idx_i.append(Int(atol(tokens[idx]))); idx += 1
        var idx_j = List[Int](capacity=n_p)
        for _ in range(n_p):
            idx_j.append(Int(atol(tokens[idx]))); idx += 1
        var epsilons = List[Float64](capacity=n_k)
        for _ in range(n_k):
            epsilons.append(atof(tokens[idx])); idx += 1
        var traj = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj.append(atof(tokens[idx])); idx += 1
        var out = List[Float64](capacity=n_k)
        for _ in range(n_k):
            out.append(0.0)
        correlation_integral(traj, t, d, idx_i, idx_j, epsilons, out)
        for k in range(n_k):
            print(out[k])

    elif op == "KY":
        var n = Int(atol(tokens[idx])); idx += 1
        var lyap = List[Float64](capacity=n)
        for _ in range(n):
            lyap.append(atof(tokens[idx])); idx += 1
        var result = kaplan_yorke_dimension(lyap)
        print(result)

    else:
        print(-1)
