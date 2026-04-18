# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Financial market PLV / R(t) (Mojo port)

"""Market order parameter and windowed PLV as a Mojo executable.

Stdin dispatch:

    ORDER t n phases[0..t*n]
        → prints ``t`` R(t) floats on stdout

    PLV t n window phases[0..t*n]
        → prints ``(t − window + 1) × n × n`` PLV floats

Build with::

    mojo build mojo/market.mojo -o mojo/market_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn order_parameter(
    phases_flat: List[Float64],
    t: Int,
    n: Int,
    mut out: List[Float64],
) -> None:
    var inv_n = 1.0 / Float64(n)
    for row in range(t):
        var sum_cos: Float64 = 0.0
        var sum_sin: Float64 = 0.0
        var base = row * n
        for i in range(n):
            var theta = phases_flat[base + i]
            sum_cos += cos(theta)
            sum_sin += sin(theta)
        var mc = sum_cos * inv_n
        var ms = sum_sin * inv_n
        out[row] = sqrt(mc * mc + ms * ms)


fn plv_matrix(
    phases_flat: List[Float64],
    t: Int,
    n: Int,
    window: Int,
    mut out: List[Float64],
) -> None:
    var n_windows = t - window + 1
    var inv_w = 1.0 / Float64(window)
    var window_s = List[Float64](capacity=window * n)
    var window_c = List[Float64](capacity=window * n)
    for _ in range(window * n):
        window_s.append(0.0)
        window_c.append(0.0)
    for w in range(n_windows):
        for k in range(window):
            var base_step = (w + k) * n
            for i in range(n):
                var theta = phases_flat[base_step + i]
                window_s[k * n + i] = sin(theta)
                window_c[k * n + i] = cos(theta)
        var mat_offset = w * n * n
        for i in range(n):
            for j in range(n):
                var sum_cos: Float64 = 0.0
                var sum_sin: Float64 = 0.0
                for k in range(window):
                    var si = window_s[k * n + i]
                    var ci = window_c[k * n + i]
                    var sj = window_s[k * n + j]
                    var cj = window_c[k * n + j]
                    sum_cos += cj * ci + sj * si
                    sum_sin += sj * ci - cj * si
                var mc = sum_cos * inv_w
                var ms = sum_sin * inv_w
                out[mat_offset + i * n + j] = sqrt(mc * mc + ms * ms)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op == "ORDER":
        var t = Int(atol(tokens[idx])); idx += 1
        var n = Int(atol(tokens[idx])); idx += 1
        var phases = List[Float64](capacity=t * n)
        for _ in range(t * n):
            phases.append(atof(tokens[idx])); idx += 1
        var out = List[Float64](capacity=t)
        for _ in range(t):
            out.append(0.0)
        order_parameter(phases, t, n, out)
        for i in range(t):
            print(out[i])
        return
    if op == "PLV":
        var t = Int(atol(tokens[idx])); idx += 1
        var n = Int(atol(tokens[idx])); idx += 1
        var window = Int(atol(tokens[idx])); idx += 1
        var phases = List[Float64](capacity=t * n)
        for _ in range(t * n):
            phases.append(atof(tokens[idx])); idx += 1
        var n_windows = t - window + 1
        var out = List[Float64](capacity=n_windows * n * n)
        for _ in range(n_windows * n * n):
            out.append(0.0)
        plv_matrix(phases, t, n, window, out)
        for i in range(n_windows * n * n):
            print(out[i])
        return
    print(-1)
