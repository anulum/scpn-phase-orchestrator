# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Envelope kernels (Mojo port)

"""Sliding-window RMS + modulation depth as a Mojo executable.

Stdin verbs:

* ``RMS T window amps[0..T]`` — prints ``T`` f64 envelope values.
* ``MOD T env[0..T]`` — prints one f64 modulation depth.

Build with::

    mojo build mojo/envelope.mojo -o mojo/envelope_mojo -Xlinker -lm
"""

from std.math import sqrt
from std.collections import List


fn extract_envelope(
    amps: List[Float64],
    window: Int,
    t: Int,
    mut out: List[Float64],
) -> None:
    if t == 0:
        return
    var cs = List[Float64](capacity=t + 1)
    for _ in range(t + 1):
        cs.append(0.0)
    for i in range(t):
        cs[i + 1] = cs[i] + amps[i] * amps[i]
    var n_valid = t - window + 1
    if n_valid <= 0:
        var v = sqrt(cs[t] / Float64(t))
        for i in range(t):
            out[i] = v
        return
    for i in range(n_valid):
        out[window - 1 + i] = sqrt(
            (cs[i + window] - cs[i]) / Float64(window)
        )
    var first = out[window - 1]
    for i in range(window - 1):
        out[i] = first


fn envelope_modulation_depth(
    env: List[Float64], t: Int,
) -> Float64:
    if t == 0:
        return 0.0
    var vmax = env[0]
    var vmin = env[0]
    for i in range(1, t):
        var v = env[i]
        if v > vmax:
            vmax = v
        if v < vmin:
            vmin = v
    var denom = vmax + vmin
    if denom <= 0.0:
        return 0.0
    return (vmax - vmin) / denom


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "RMS":
        var t = Int(atol(tokens[idx])); idx += 1
        var window = Int(atol(tokens[idx])); idx += 1
        var amps = List[Float64](capacity=t)
        for _ in range(t):
            amps.append(atof(tokens[idx])); idx += 1
        var out = List[Float64](capacity=t)
        for _ in range(t):
            out.append(0.0)
        extract_envelope(amps, window, t, out)
        for i in range(t):
            print(out[i])

    elif op == "MOD":
        var t = Int(atol(tokens[idx])); idx += 1
        var env = List[Float64](capacity=t)
        for _ in range(t):
            env.append(atof(tokens[idx])); idx += 1
        print(envelope_modulation_depth(env, t))

    else:
        print(-1)
