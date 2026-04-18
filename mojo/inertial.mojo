# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto (Mojo port)

"""Second-order (swing-equation) Kuramoto RK4 stepper as a Mojo
executable.

Stdin:

    INERT n dt theta[0..n] omega_dot[0..n] power[0..n]
          knm[0..n*n] inertia[0..n] damping[0..n]

Prints ``n`` new_theta floats followed by ``n`` new_omega_dot
floats (one per line).

Build with::

    mojo build mojo/inertial.mojo -o mojo/inertial_mojo -Xlinker -lm
"""

from std.math import sin, cos
from std.collections import List


fn _compute_derivative(
    theta: List[Float64],
    omega_dot: List[Float64],
    power: List[Float64],
    knm: List[Float64],
    inertia: List[Float64],
    damping: List[Float64],
    n: Int,
    mut out_t: List[Float64],
    mut out_o: List[Float64],
) -> None:
    var sin_th = List[Float64](capacity=n)
    var cos_th = List[Float64](capacity=n)
    for i in range(n):
        sin_th.append(sin(theta[i]))
        cos_th.append(cos(theta[i]))
    for i in range(n):
        out_t[i] = omega_dot[i]
        var ci = cos_th[i]
        var si = sin_th[i]
        var offset = i * n
        var coupling: Float64 = 0.0
        for j in range(n):
            coupling += knm[offset + j] * (sin_th[j] * ci - cos_th[j] * si)
        out_o[i] = (power[i] + coupling - damping[i] * omega_dot[i]) \
            / inertia[i]


fn inertial_step(
    theta: List[Float64],
    omega_dot: List[Float64],
    power: List[Float64],
    knm: List[Float64],
    inertia: List[Float64],
    damping: List[Float64],
    n: Int,
    dt: Float64,
    mut new_theta: List[Float64],
    mut new_omega_dot: List[Float64],
) -> None:
    var two_pi = 6.283185307179586
    var k1t = List[Float64](capacity=n)
    var k1o = List[Float64](capacity=n)
    var k2t = List[Float64](capacity=n)
    var k2o = List[Float64](capacity=n)
    var k3t = List[Float64](capacity=n)
    var k3o = List[Float64](capacity=n)
    var k4t = List[Float64](capacity=n)
    var k4o = List[Float64](capacity=n)
    var tmp_th = List[Float64](capacity=n)
    var tmp_od = List[Float64](capacity=n)
    for _ in range(n):
        k1t.append(0.0); k1o.append(0.0)
        k2t.append(0.0); k2o.append(0.0)
        k3t.append(0.0); k3o.append(0.0)
        k4t.append(0.0); k4o.append(0.0)
        tmp_th.append(0.0); tmp_od.append(0.0)

    _compute_derivative(theta, omega_dot, power, knm, inertia, damping,
                        n, k1t, k1o)
    for i in range(n):
        tmp_th[i] = theta[i] + 0.5 * dt * k1t[i]
        tmp_od[i] = omega_dot[i] + 0.5 * dt * k1o[i]
    _compute_derivative(tmp_th, tmp_od, power, knm, inertia, damping,
                        n, k2t, k2o)
    for i in range(n):
        tmp_th[i] = theta[i] + 0.5 * dt * k2t[i]
        tmp_od[i] = omega_dot[i] + 0.5 * dt * k2o[i]
    _compute_derivative(tmp_th, tmp_od, power, knm, inertia, damping,
                        n, k3t, k3o)
    for i in range(n):
        tmp_th[i] = theta[i] + dt * k3t[i]
        tmp_od[i] = omega_dot[i] + dt * k3o[i]
    _compute_derivative(tmp_th, tmp_od, power, knm, inertia, damping,
                        n, k4t, k4o)

    var dt6 = dt / 6.0
    for i in range(n):
        var raw = theta[i] + dt6 * (
            k1t[i] + 2.0 * k2t[i] + 2.0 * k3t[i] + k4t[i]
        )
        var v = raw - Float64(Int(raw / two_pi)) * two_pi
        if v < 0.0:
            v += two_pi
        new_theta[i] = v
        new_omega_dot[i] = omega_dot[i] + dt6 * (
            k1o[i] + 2.0 * k2o[i] + 2.0 * k3o[i] + k4o[i]
        )


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "INERT":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var dt = atof(tokens[idx]); idx += 1

    var theta = List[Float64](capacity=n)
    for _ in range(n):
        theta.append(atof(tokens[idx])); idx += 1
    var omega_dot = List[Float64](capacity=n)
    for _ in range(n):
        omega_dot.append(atof(tokens[idx])); idx += 1
    var power = List[Float64](capacity=n)
    for _ in range(n):
        power.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var inertia = List[Float64](capacity=n)
    for _ in range(n):
        inertia.append(atof(tokens[idx])); idx += 1
    var damping = List[Float64](capacity=n)
    for _ in range(n):
        damping.append(atof(tokens[idx])); idx += 1

    var new_th = List[Float64](capacity=n)
    var new_od = List[Float64](capacity=n)
    for _ in range(n):
        new_th.append(0.0); new_od.append(0.0)
    inertial_step(theta, omega_dot, power, knm, inertia, damping,
                  n, dt, new_th, new_od)
    for i in range(n):
        print(new_th[i])
    for i in range(n):
        print(new_od[i])
