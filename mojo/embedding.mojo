# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Embedding primitives (Mojo port)

"""Delay-embedding primitives as a Mojo executable.

Stdin verbs:

* ``DE T delay dim signal[0..T]`` — delay_embed; prints T_eff*dim floats.
* ``MI T lag n_bins signal[0..T]`` — mutual information; prints one float.
* ``NN T m embedded[0..T*m]`` — nearest_neighbor_distances; prints
  T distances then T int indices.

Build with::

    mojo build mojo/embedding.mojo -o mojo/embedding_mojo -Xlinker -lm
"""

from std.math import log, sqrt, floor
from std.collections import List


fn delay_embed(
    signal: List[Float64],
    delay: Int,
    dim: Int,
    t_eff: Int,
    mut out: List[Float64],
) -> None:
    for i in range(t_eff):
        for d in range(dim):
            out[i * dim + d] = signal[i + d * delay]


fn mutual_information(
    signal: List[Float64],
    lag: Int,
    n_bins: Int,
) -> Float64:
    var t_total = len(signal) - lag
    if t_total <= 0:
        return 0.0
    var x_min = signal[0]
    var x_max = signal[0]
    for i in range(1, t_total):
        var v = signal[i]
        if v < x_min:
            x_min = v
        if v > x_max:
            x_max = v
    var y_min = signal[lag]
    var y_max = signal[lag]
    for i in range(lag + 1, lag + t_total):
        var v = signal[i]
        if v < y_min:
            y_min = v
        if v > y_max:
            y_max = v
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    var dx = (x_max - x_min) / Float64(n_bins)
    var dy = (y_max - y_min) / Float64(n_bins)
    var hist = List[Float64](capacity=n_bins * n_bins)
    for _ in range(n_bins * n_bins):
        hist.append(0.0)
    for i in range(t_total):
        var x = signal[i]
        var y = signal[i + lag]
        var bx = Int(floor((x - x_min) / dx))
        var by = Int(floor((y - y_min) / dy))
        if bx >= n_bins:
            bx = n_bins - 1
        if by >= n_bins:
            by = n_bins - 1
        hist[bx * n_bins + by] = hist[bx * n_bins + by] + 1.0
    var total = Float64(t_total)
    var hx = List[Float64](capacity=n_bins)
    var hy = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        hx.append(0.0)
        hy.append(0.0)
    for i in range(n_bins):
        for j in range(n_bins):
            var h = hist[i * n_bins + j]
            hx[i] = hx[i] + h
            hy[j] = hy[j] + h
    var mi: Float64 = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            var h = hist[i * n_bins + j]
            if h > 0.0 and hx[i] > 0.0 and hy[j] > 0.0:
                var p_xy = h / total
                var p_x = hx[i] / total
                var p_y = hy[j] / total
                mi += p_xy * log(p_xy / (p_x * p_y))
    return mi


fn nearest_neighbor_distances(
    embedded: List[Float64],
    t: Int,
    m: Int,
    mut dist: List[Float64],
    mut idx: List[Int],
) -> None:
    var inf_val: Float64 = 1.0e300
    for i in range(t):
        var best = inf_val
        var best_j = 0
        var base_i = i * m
        for j in range(t):
            if j == i:
                continue
            var base_j = j * m
            var d: Float64 = 0.0
            for k in range(m):
                var dd = embedded[base_i + k] - embedded[base_j + k]
                d += dd * dd
            if d < best:
                best = d
                best_j = j
        dist[i] = sqrt(best)
        idx[i] = best_j


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "DE":
        var t = Int(atol(tokens[idx])); idx += 1
        var delay = Int(atol(tokens[idx])); idx += 1
        var dim = Int(atol(tokens[idx])); idx += 1
        var signal = List[Float64](capacity=t)
        for _ in range(t):
            signal.append(atof(tokens[idx])); idx += 1
        var t_eff = t - (dim - 1) * delay
        if t_eff <= 0:
            print(-1)
            return
        var out = List[Float64](capacity=t_eff * dim)
        for _ in range(t_eff * dim):
            out.append(0.0)
        delay_embed(signal, delay, dim, t_eff, out)
        for k in range(t_eff * dim):
            print(out[k])

    elif op == "MI":
        var t = Int(atol(tokens[idx])); idx += 1
        var lag = Int(atol(tokens[idx])); idx += 1
        var n_bins = Int(atol(tokens[idx])); idx += 1
        var signal = List[Float64](capacity=t)
        for _ in range(t):
            signal.append(atof(tokens[idx])); idx += 1
        print(mutual_information(signal, lag, n_bins))

    elif op == "NN":
        var t = Int(atol(tokens[idx])); idx += 1
        var m = Int(atol(tokens[idx])); idx += 1
        var embedded = List[Float64](capacity=t * m)
        for _ in range(t * m):
            embedded.append(atof(tokens[idx])); idx += 1
        var dist = List[Float64](capacity=t)
        var nn_idx = List[Int](capacity=t)
        for _ in range(t):
            dist.append(0.0)
            nn_idx.append(0)
        nearest_neighbor_distances(embedded, t, m, dist, nn_idx)
        for i in range(t):
            print(dist[i])
        for i in range(t):
            print(nn_idx[i])

    else:
        print(-1)
