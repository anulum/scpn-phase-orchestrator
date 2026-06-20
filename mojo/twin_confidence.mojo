# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin divergence (Mojo port)

"""Digital-twin divergence kernel in Mojo.

Communicates with Python over a one-line whitespace-separated stdin / stdout
protocol (see ``_twin_confidence_mojo.py``).

Stdin layout on one line (whitespace separated):

    n w n_bins
    model_phases[0..n]
    observed_phases[0..n]
    model_order[0..w]
    observed_order[0..w]

Stdout: two floats, ``phase_js_divergence`` then ``order_wasserstein``, one per
line. Matches the NumPy / Rust / Go / Julia references to 1e-9. The whole
kernel is inlined in ``main`` so no ``List`` values cross a function boundary.
"""

from std.math import floor, log
from std.collections import List


fn main() raises:
    var two_pi: Float64 = 6.283185307179586
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var n = Int(atol(tokens[idx])); idx += 1
    var w = Int(atol(tokens[idx])); idx += 1
    var n_bins = Int(atol(tokens[idx])); idx += 1

    var model_phases = List[Float64](capacity=n)
    for _ in range(n):
        model_phases.append(atof(tokens[idx])); idx += 1
    var observed_phases = List[Float64](capacity=n)
    for _ in range(n):
        observed_phases.append(atof(tokens[idx])); idx += 1
    var model_order = List[Float64](capacity=w)
    for _ in range(w):
        model_order.append(atof(tokens[idx])); idx += 1
    var observed_order = List[Float64](capacity=w)
    for _ in range(w):
        observed_order.append(atof(tokens[idx])); idx += 1

    var width = two_pi / Float64(n_bins)

    # Phase histograms p (model) and q (observed), normalised to PMFs.
    var p = List[Float64](capacity=n_bins)
    var q = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        p.append(0.0)
        q.append(0.0)
    for i in range(n):
        var phase = model_phases[i]
        var wrapped = phase - floor(phase / two_pi) * two_pi
        var bidx = Int(floor(wrapped / width))
        if bidx < 0:
            bidx = 0
        if bidx > n_bins - 1:
            bidx = n_bins - 1
        p[bidx] += 1.0
    for i in range(n):
        var phase = observed_phases[i]
        var wrapped = phase - floor(phase / two_pi) * two_pi
        var bidx = Int(floor(wrapped / width))
        if bidx < 0:
            bidx = 0
        if bidx > n_bins - 1:
            bidx = n_bins - 1
        q[bidx] += 1.0
    var total_p: Float64 = 0.0
    var total_q: Float64 = 0.0
    for i in range(n_bins):
        total_p += p[i]
        total_q += q[i]
    for i in range(n_bins):
        if total_p <= 0.0:
            p[i] = 1.0 / Float64(n_bins)
        else:
            p[i] = p[i] / total_p
        if total_q <= 0.0:
            q[i] = 1.0 / Float64(n_bins)
        else:
            q[i] = q[i] / total_q

    # Jensen–Shannon divergence (natural log).
    var js: Float64 = 0.0
    for i in range(n_bins):
        var mi = 0.5 * (p[i] + q[i])
        if p[i] > 0.0:
            js += 0.5 * (p[i] * log(p[i] / mi))
        if q[i] > 0.0:
            js += 0.5 * (q[i] * log(q[i] / mi))

    # Order-parameter Wasserstein-1 via ascending insertion sort.
    var sorted_model = List[Float64](capacity=w)
    var sorted_obs = List[Float64](capacity=w)
    for i in range(w):
        sorted_model.append(model_order[i])
        sorted_obs.append(observed_order[i])
    for i in range(1, w):
        var key_m = sorted_model[i]
        var j = i - 1
        while j >= 0 and sorted_model[j] > key_m:
            sorted_model[j + 1] = sorted_model[j]
            j -= 1
        sorted_model[j + 1] = key_m
    for i in range(1, w):
        var key_o = sorted_obs[i]
        var j = i - 1
        while j >= 0 and sorted_obs[j] > key_o:
            sorted_obs[j + 1] = sorted_obs[j]
            j -= 1
        sorted_obs[j + 1] = key_o
    var acc: Float64 = 0.0
    for i in range(w):
        var diff = sorted_model[i] - sorted_obs[i]
        if diff < 0.0:
            diff = -diff
        acc += diff
    var w1 = acc / Float64(w)

    print(js)
    print(w1)
