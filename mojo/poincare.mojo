# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincaré section kernels (Mojo port)

"""Generic + phase Poincaré sections as a Mojo executable.

Stdin verbs:

* ``SEC T d direction_id offset normal[0..d] traj[0..T*d]`` — generic
  hyperplane section. Prints ``n_cr`` then ``n_cr * d`` crossing
  coords then ``n_cr`` times (one line each).
* ``PHASE T N oscillator_idx section_phase phases[0..T*N]`` — phase
  Poincaré. Same print format.

Build with::

    mojo build mojo/poincare.mojo -o mojo/poincare_mojo -Xlinker -lm
"""

from std.math import sqrt, abs, floor
from std.collections import List


fn unwrap(mut target: List[Float64], t: Int) -> None:
    var two_pi = 6.283185307179586
    var pi_val = 3.141592653589793
    for i in range(1, t):
        var diff = target[i] - target[i - 1]
        if diff > pi_val:
            var delta = -two_pi * Float64(Int((diff + pi_val) / two_pi))
            for k in range(i, t):
                target[k] = target[k] + delta
        elif diff < -pi_val:
            var delta = two_pi * Float64(Int((-diff + pi_val) / two_pi))
            for k in range(i, t):
                target[k] = target[k] + delta


fn poincare_section(
    traj: List[Float64],
    t: Int,
    d: Int,
    normal: List[Float64],
    offset: Float64,
    direction_id: Int,
    mut crossings: List[Float64],
    mut times: List[Float64],
) -> Int:
    var norm_sq: Float64 = 0.0
    for k in range(d):
        norm_sq += normal[k] * normal[k]
    var norm_mag = sqrt(norm_sq)
    if norm_mag <= 0.0:
        return 0
    var n_vec = List[Float64](capacity=d)
    for k in range(d):
        n_vec.append(normal[k] / norm_mag)

    var signed = List[Float64](capacity=t)
    for _ in range(t):
        signed.append(0.0)
    for i in range(t):
        var s: Float64 = 0.0
        var base = i * d
        for k in range(d):
            s += traj[base + k] * n_vec[k]
        signed[i] = s - offset

    var n_cr = 0
    for i in range(t - 1):
        var d0 = signed[i]
        var d1 = signed[i + 1]
        var is_cross = False
        if direction_id == 0:
            is_cross = d0 < 0.0 and d1 >= 0.0
        elif direction_id == 1:
            is_cross = d0 > 0.0 and d1 <= 0.0
        else:
            is_cross = d0 * d1 < 0.0
        if not is_cross:
            continue
        var alpha: Float64 = 0.5
        if abs(d1 - d0) > 1e-15:
            alpha = -d0 / (d1 - d0)
        var base_i = i * d
        var base_next = (i + 1) * d
        for k in range(d):
            var xi = traj[base_i + k]
            var xj = traj[base_next + k]
            crossings[n_cr * d + k] = xi + alpha * (xj - xi)
        times[n_cr] = Float64(i) + alpha
        n_cr += 1
    return n_cr


fn phase_poincare(
    phases: List[Float64],
    t: Int,
    n: Int,
    osc_idx: Int,
    section_phase: Float64,
    mut crossings: List[Float64],
    mut times: List[Float64],
) -> Int:
    var two_pi = 6.283185307179586
    var pi_val = 3.141592653589793

    var target = List[Float64](capacity=t)
    for i in range(t):
        target.append(phases[i * n + osc_idx])
    unwrap(target, t)

    var shifted = List[Float64](capacity=t)
    for i in range(t):
        var x = target[i] - section_phase
        var r = x - Float64(Int(x / two_pi)) * two_pi
        if r < 0.0:
            r += two_pi
        shifted.append(r)

    var n_cr = 0
    for i in range(t - 1):
        if shifted[i] > pi_val and shifted[i + 1] < pi_val:
            var denom = shifted[i] - shifted[i + 1] + two_pi
            var alpha: Float64 = 0.5
            if denom != 0.0:
                alpha = shifted[i] / denom
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            var base_i = i * n
            var base_next = (i + 1) * n
            for k in range(n):
                var xi = phases[base_i + k]
                var xj = phases[base_next + k]
                crossings[n_cr * n + k] = xi + alpha * (xj - xi)
            times[n_cr] = Float64(i) + alpha
            n_cr += 1
    return n_cr


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "SEC":
        var t = Int(atol(tokens[idx])); idx += 1
        var d = Int(atol(tokens[idx])); idx += 1
        var direction_id = Int(atol(tokens[idx])); idx += 1
        var offset = atof(tokens[idx]); idx += 1
        var normal = List[Float64](capacity=d)
        for _ in range(d):
            normal.append(atof(tokens[idx])); idx += 1
        var traj = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj.append(atof(tokens[idx])); idx += 1
        var crossings = List[Float64](capacity=t * d)
        var times = List[Float64](capacity=t)
        for _ in range(t * d):
            crossings.append(0.0)
        for _ in range(t):
            times.append(0.0)
        var n_cr = poincare_section(
            traj, t, d, normal, offset, direction_id, crossings, times,
        )
        print(n_cr)
        for k in range(n_cr * d):
            print(crossings[k])
        for k in range(n_cr):
            print(times[k])

    elif op == "PHASE":
        var t = Int(atol(tokens[idx])); idx += 1
        var n = Int(atol(tokens[idx])); idx += 1
        var osc_idx = Int(atol(tokens[idx])); idx += 1
        var section_phase = atof(tokens[idx]); idx += 1
        var phases = List[Float64](capacity=t * n)
        for _ in range(t * n):
            phases.append(atof(tokens[idx])); idx += 1
        var crossings = List[Float64](capacity=t * n)
        var times = List[Float64](capacity=t)
        for _ in range(t * n):
            crossings.append(0.0)
        for _ in range(t):
            times.append(0.0)
        var n_cr = phase_poincare(
            phases, t, n, osc_idx, section_phase, crossings, times,
        )
        print(n_cr)
        for k in range(n_cr * n):
            print(crossings[k])
        for k in range(n_cr):
            print(times[k])

    else:
        print(-1)
