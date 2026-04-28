# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-Trial Phase Coherence (Mojo port)

"""Lachaux 1999 ITPC as a Mojo executable.

Stdin layout (single whitespace-separated line):

* ``ITPC n_trials n_timepoints phases[0..n_trials*n_timepoints]`` —
  prints ``n_timepoints`` f64 lines.
* ``PERS n_trials n_timepoints n_idx pause_idx[0..n_idx]
   phases[0..n_trials*n_timepoints]`` — prints a single f64 mean.

Build with::

    mojo build mojo/itpc.mojo -o mojo/itpc_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn compute_itpc(
    phases: List[Float64],
    n_trials: Int,
    n_tp: Int,
    mut out: List[Float64],
) -> None:
    if n_trials == 0:
        return
    var inv_n = 1.0 / Float64(n_trials)
    for t in range(n_tp):
        var sr: Float64 = 0.0
        var si: Float64 = 0.0
        for k in range(n_trials):
            var th = phases[k * n_tp + t]
            sr += cos(th)
            si += sin(th)
        sr = sr * inv_n
        si = si * inv_n
        out[t] = sqrt(sr * sr + si * si)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "ITPC":
        var n_trials = Int(atol(tokens[idx])); idx += 1
        var n_tp = Int(atol(tokens[idx])); idx += 1
        var phases = List[Float64](capacity=n_trials * n_tp)
        for _ in range(n_trials * n_tp):
            phases.append(atof(tokens[idx])); idx += 1
        var out = List[Float64](capacity=n_tp)
        for _ in range(n_tp):
            out.append(0.0)
        compute_itpc(phases, n_trials, n_tp, out)
        for t in range(n_tp):
            print(out[t])

    elif op == "PERS":
        var n_trials = Int(atol(tokens[idx])); idx += 1
        var n_tp = Int(atol(tokens[idx])); idx += 1
        var n_idx = Int(atol(tokens[idx])); idx += 1
        var pause = List[Int](capacity=n_idx)
        for _ in range(n_idx):
            pause.append(Int(atol(tokens[idx]))); idx += 1
        var phases = List[Float64](capacity=n_trials * n_tp)
        for _ in range(n_trials * n_tp):
            phases.append(atof(tokens[idx])); idx += 1
        if n_idx == 0 or n_trials == 0 or n_tp == 0:
            print(0.0)
            return
        var itpc = List[Float64](capacity=n_tp)
        for _ in range(n_tp):
            itpc.append(0.0)
        compute_itpc(phases, n_trials, n_tp, itpc)
        var acc: Float64 = 0.0
        var count = 0
        for i in range(n_idx):
            var p = pause[i]
            if p >= 0 and p < n_tp:
                acc += itpc[p]
                count += 1
        if count == 0:
            print(0.0)
        else:
            print(acc / Float64(count))

    else:
        print(-1)
