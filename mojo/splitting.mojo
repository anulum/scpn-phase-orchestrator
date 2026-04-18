# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strang operator splitting (Mojo port)

"""Strang-split Kuramoto stepper (A-B-A) as a Mojo executable.

Stdin:

    SPLIT n zeta psi dt n_steps
          phases[0..n] omegas[0..n]
          knm[0..n*n] alpha[0..n*n]

Prints ``n`` final-phase floats on stdout (one per line).

Build with::

    mojo build mojo/splitting.mojo -o mojo/splitting_mojo -Xlinker -lm
"""

from std.math import sin, cos
from std.collections import List

alias TWO_PI: Float64 = 6.283185307179586


fn _mod_two_pi(x: Float64) -> Float64:
    var v = x - Float64(Int(x / TWO_PI)) * TWO_PI
    if v < 0.0:
        v += TWO_PI
    return v


fn _compute_coupling_deriv(
    theta: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    alpha_zero: Bool,
    mut out: List[Float64],
) -> None:
    var sin_th = List[Float64](capacity=n)
    var cos_th = List[Float64](capacity=n)
    for i in range(n):
        sin_th.append(sin(theta[i]))
        cos_th.append(cos(theta[i]))
    var zs_psi: Float64 = 0.0
    var zc_psi: Float64 = 0.0
    if zeta != 0.0:
        zs_psi = zeta * sin(psi)
        zc_psi = zeta * cos(psi)
    for i in range(n):
        var offset = i * n
        var ci = cos_th[i]
        var si = sin_th[i]
        var acc: Float64 = 0.0
        if alpha_zero:
            for j in range(n):
                acc += knm[offset + j] * (sin_th[j] * ci - cos_th[j] * si)
        else:
            for j in range(n):
                acc += knm[offset + j] * \
                    sin(theta[j] - theta[i] - alpha[offset + j])
        out[i] = acc
        if zeta != 0.0:
            out[i] += zs_psi * ci - zc_psi * si


fn _rk4_coupling(
    mut p: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    alpha_zero: Bool,
) -> None:
    var k1 = List[Float64](capacity=n)
    var k2 = List[Float64](capacity=n)
    var k3 = List[Float64](capacity=n)
    var k4 = List[Float64](capacity=n)
    var tmp = List[Float64](capacity=n)
    for _ in range(n):
        k1.append(0.0); k2.append(0.0); k3.append(0.0); k4.append(0.0)
        tmp.append(0.0)
    _compute_coupling_deriv(p, knm, alpha, n, zeta, psi, alpha_zero, k1)
    for i in range(n):
        tmp[i] = _mod_two_pi(p[i] + 0.5 * dt * k1[i])
    _compute_coupling_deriv(tmp, knm, alpha, n, zeta, psi, alpha_zero, k2)
    for i in range(n):
        tmp[i] = _mod_two_pi(p[i] + 0.5 * dt * k2[i])
    _compute_coupling_deriv(tmp, knm, alpha, n, zeta, psi, alpha_zero, k3)
    for i in range(n):
        tmp[i] = _mod_two_pi(p[i] + dt * k3[i])
    _compute_coupling_deriv(tmp, knm, alpha, n, zeta, psi, alpha_zero, k4)
    var dt6 = dt / 6.0
    for i in range(n):
        p[i] = _mod_two_pi(p[i] + dt6 * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
        ))


fn splitting_run(
    phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    n_steps: Int,
    mut out: List[Float64],
) -> None:
    for i in range(n):
        out[i] = phases[i]
    var alpha_zero = True
    for a in alpha:
        if a != 0.0:
            alpha_zero = False
            break
    var half_dt = 0.5 * dt
    for _ in range(n_steps):
        for i in range(n):
            out[i] = _mod_two_pi(out[i] + half_dt * omegas[i])
        _rk4_coupling(out, knm, alpha, n, zeta, psi, dt, alpha_zero)
        for i in range(n):
            out[i] = _mod_two_pi(out[i] + half_dt * omegas[i])


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "SPLIT":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var alpha = List[Float64](capacity=n * n)
    for _ in range(n * n):
        alpha.append(atof(tokens[idx])); idx += 1

    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(0.0)
    splitting_run(phases, omegas, knm, alpha, n,
                  zeta, psi, dt, n_steps, out)
    for i in range(n):
        print(out[i])
