# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling (Mojo port)

"""Tort 2010 PAC modulation index and matrix builder as a Mojo
executable. Text stdin protocol.

Stdin line layout (whitespace separated):

* ``MI n n_bins theta[0..n] amp[0..n]`` → prints single MI value.
* ``MAT t n n_bins phases[0..t*n] amplitudes[0..t*n]`` → prints
  ``n*n`` flat MI matrix values, one per line.

Build with::

    mojo build mojo/pac.mojo -o mojo/pac_mojo -Xlinker -lm
"""

from std.math import log, sin, cos
from std.collections import List


fn modulation_index(
    theta: List[Float64],
    amp: List[Float64],
    n_bins: Int,
) -> Float64:
    if n_bins < 2:
        return 0.0
    var n = len(theta)
    if len(amp) < n:
        n = len(amp)
    if n == 0:
        return 0.0
    var two_pi = 2.0 * 3.141592653589793
    var bin_width = two_pi / Float64(n_bins)
    var mean_amp = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        mean_amp.append(0.0)
    var counts = List[Int](capacity=n_bins)
    for _ in range(n_bins):
        counts.append(0)
    for i in range(n):
        var wrapped = theta[i] % two_pi
        if wrapped < 0:
            wrapped += two_pi
        var k = Int(wrapped / bin_width)
        if k >= n_bins:
            k = n_bins - 1
        mean_amp[k] += amp[i]
        counts[k] += 1
    for k in range(n_bins):
        if counts[k] > 0:
            mean_amp[k] /= Float64(counts[k])
    var total: Float64 = 0.0
    for k in range(n_bins):
        total += mean_amp[k]
    if total <= 0.0:
        return 0.0
    var log_n = log(Float64(n_bins))
    if log_n < 1e-15:
        return 0.0
    var kl: Float64 = 0.0
    for k in range(n_bins):
        var pk = mean_amp[k] / total
        if pk > 0.0:
            kl += pk * log(pk * Float64(n_bins))
    var mi = kl / log_n
    if mi < 0.0:
        return 0.0
    if mi > 1.0:
        return 1.0
    return mi


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "MI":
        var n = Int(atol(tokens[idx])); idx += 1
        var n_bins = Int(atol(tokens[idx])); idx += 1
        var theta = List[Float64](capacity=n)
        for _ in range(n):
            theta.append(atof(tokens[idx])); idx += 1
        var amp = List[Float64](capacity=n)
        for _ in range(n):
            amp.append(atof(tokens[idx])); idx += 1
        print(modulation_index(theta, amp, n_bins))
    elif op == "MAT":
        var t = Int(atol(tokens[idx])); idx += 1
        var n = Int(atol(tokens[idx])); idx += 1
        var n_bins = Int(atol(tokens[idx])); idx += 1
        var tn = t * n
        var phases = List[Float64](capacity=tn)
        for _ in range(tn):
            phases.append(atof(tokens[idx])); idx += 1
        var amps = List[Float64](capacity=tn)
        for _ in range(tn):
            amps.append(atof(tokens[idx])); idx += 1

        var theta_col = List[Float64](capacity=t)
        for _ in range(t):
            theta_col.append(0.0)
        var amp_col = List[Float64](capacity=t)
        for _ in range(t):
            amp_col.append(0.0)

        for i in range(n):
            for s in range(t):
                theta_col[s] = phases[s * n + i]
            for j in range(n):
                for s in range(t):
                    amp_col[s] = amps[s * n + j]
                print(modulation_index(theta_col, amp_col, n_bins))
    else:
        print(-1)
