# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Partial Information Decomposition (Mojo port)

"""Time-series Williams & Beer PID as a Mojo executable.

Stdin:

    PID t n n_bins n_a n_b history[0..t*n] group_a[0..n_a] group_b[0..n_b]

Prints two f64 lines: redundancy then synergy. Group indices are 0-based.

Build with::

    mojo build mojo/pid.mojo -o mojo/pid_mojo -Xlinker -lm
"""

from std.math import sin, cos, atan2, log, floor
from std.collections import List

alias TAU = 6.283185307179586


fn bin_angle(angle: Float64, n_bins: Int) -> Int:
    var w = angle - floor(angle / TAU) * TAU
    var b = Int(floor(w / (TAU / Float64(n_bins))))
    if b > n_bins - 1:
        b = n_bins - 1
    return b


fn group_phase(history: List[Float64], n: Int, row: Int, members: List[Int]) -> Float64:
    var sin_sum: Float64 = 0.0
    var cos_sum: Float64 = 0.0
    for k in range(len(members)):
        var theta = history[row * n + members[k]]
        sin_sum += sin(theta)
        cos_sum += cos(theta)
    var count = Float64(len(members))
    return atan2(sin_sum / count, cos_sum / count)


fn global_phase(history: List[Float64], n: Int, row: Int) -> Float64:
    var sin_sum: Float64 = 0.0
    var cos_sum: Float64 = 0.0
    for j in range(n):
        var theta = history[row * n + j]
        sin_sum += sin(theta)
        cos_sum += cos(theta)
    var count = Float64(n)
    return atan2(sin_sum / count, cos_sum / count)


fn mutual_information(
    joint: List[Float64],
    marg_x: List[Float64],
    marg_y: List[Float64],
    n_x: Int,
    n_bins: Int,
    total: Float64,
) -> Float64:
    if total <= 0.0:
        return 0.0
    var mi: Float64 = 0.0
    for x in range(n_x):
        if marg_x[x] <= 0.0:
            continue
        for y in range(n_bins):
            var cxy = joint[x * n_bins + y]
            if cxy <= 0.0 or marg_y[y] <= 0.0:
                continue
            var p_xy = cxy / total
            mi += p_xy * log(p_xy / ((marg_x[x] / total) * (marg_y[y] / total)))
    if mi < 0.0:
        return 0.0
    return mi


fn i_min_redundancy(
    cay: List[Float64],
    cby: List[Float64],
    ca: List[Float64],
    cb: List[Float64],
    cy: List[Float64],
    n_bins: Int,
    total: Float64,
) -> Float64:
    if total <= 0.0:
        return 0.0
    var i_red: Float64 = 0.0
    for y in range(n_bins):
        if cy[y] <= 0.0:
            continue
        var p_y = cy[y] / total
        var ispec_a: Float64 = 0.0
        for x in range(n_bins):
            if cay[x * n_bins + y] <= 0.0 or ca[x] <= 0.0:
                continue
            var p_a_given_y = cay[x * n_bins + y] / cy[y]
            var p_y_given_a = cay[x * n_bins + y] / ca[x]
            ispec_a += p_a_given_y * log(p_y_given_a / p_y)
        var ispec_b: Float64 = 0.0
        for x in range(n_bins):
            if cby[x * n_bins + y] <= 0.0 or cb[x] <= 0.0:
                continue
            var p_b_given_y = cby[x * n_bins + y] / cy[y]
            var p_y_given_b = cby[x * n_bins + y] / cb[x]
            ispec_b += p_b_given_y * log(p_y_given_b / p_y)
        var min_spec = ispec_a
        if ispec_b < min_spec:
            min_spec = ispec_b
        i_red += p_y * min_spec
    if i_red < 0.0:
        return 0.0
    return i_red


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "PID":
        print(-1)
        return

    var t = Int(atol(tokens[idx])); idx += 1
    var n = Int(atol(tokens[idx])); idx += 1
    var n_bins = Int(atol(tokens[idx])); idx += 1
    var n_a = Int(atol(tokens[idx])); idx += 1
    var n_b = Int(atol(tokens[idx])); idx += 1

    var history = List[Float64](capacity=t * n)
    for _ in range(t * n):
        history.append(atof(tokens[idx])); idx += 1
    var group_a = List[Int](capacity=n_a)
    for _ in range(n_a):
        group_a.append(Int(atol(tokens[idx]))); idx += 1
    var group_b = List[Int](capacity=n_b)
    for _ in range(n_b):
        group_b.append(Int(atol(tokens[idx]))); idx += 1

    if t == 0 or n == 0 or n_a == 0 or n_b == 0 or n_bins == 0:
        print(0.0)
        print(0.0)
        return

    var cy = List[Float64](capacity=n_bins)
    var ca = List[Float64](capacity=n_bins)
    var cb = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        cy.append(0.0); ca.append(0.0); cb.append(0.0)
    var cay = List[Float64](capacity=n_bins * n_bins)
    var cby = List[Float64](capacity=n_bins * n_bins)
    var cab = List[Float64](capacity=n_bins * n_bins)
    for _ in range(n_bins * n_bins):
        cay.append(0.0); cby.append(0.0); cab.append(0.0)
    var caby = List[Float64](capacity=n_bins * n_bins * n_bins)
    for _ in range(n_bins * n_bins * n_bins):
        caby.append(0.0)

    for row in range(t):
        var y = bin_angle(global_phase(history, n, row), n_bins)
        var a = bin_angle(group_phase(history, n, row, group_a), n_bins)
        var b = bin_angle(group_phase(history, n, row, group_b), n_bins)
        cy[y] += 1.0
        ca[a] += 1.0
        cb[b] += 1.0
        cay[a * n_bins + y] += 1.0
        cby[b * n_bins + y] += 1.0
        var ab = a * n_bins + b
        cab[ab] += 1.0
        caby[ab * n_bins + y] += 1.0

    var total = Float64(t)
    var mi_a = mutual_information(cay, ca, cy, n_bins, n_bins, total)
    var mi_b = mutual_information(cby, cb, cy, n_bins, n_bins, total)
    var mi_ab = mutual_information(caby, cab, cy, n_bins * n_bins, n_bins, total)
    var i_red = i_min_redundancy(cay, cby, ca, cb, cy, n_bins, total)
    var syn = mi_ab - mi_a - mi_b + i_red
    if syn < 0.0:
        syn = 0.0
    print(i_red)
    print(syn)
