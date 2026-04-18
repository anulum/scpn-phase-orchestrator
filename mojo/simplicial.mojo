# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial Kuramoto (Mojo port)

"""Pairwise + 3-body simplicial Kuramoto stepper as a Mojo
executable.

Stdin:

    SIMP n zeta psi sigma2 dt n_steps
         phases[0..n] omegas[0..n]
         knm[0..n*n] alpha[0..n*n]

Prints ``n`` final-phase floats on stdout (one per line).

Build with::

    mojo build mojo/simplicial.mojo -o mojo/simplicial_mojo -Xlinker -lm
"""

from std.math import sin, cos
from std.collections import List


fn _compute_derivative(
    theta: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    sigma2: Float64,
    alpha_zero: Bool,
    mut deriv: List[Float64],
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
    var gs: Float64 = 0.0
    var gc: Float64 = 0.0
    var use3 = (sigma2 != 0.0) and (n >= 3)
    if use3:
        for i in range(n):
            gs += sin_th[i]
            gc += cos_th[i]
    var inv_n2: Float64 = 0.0
    if n > 0:
        inv_n2 = sigma2 / (Float64(n) * Float64(n))

    for i in range(n):
        var offset = i * n
        var ci = cos_th[i]
        var si = sin_th[i]
        var pw: Float64 = 0.0
        if alpha_zero:
            for j in range(n):
                pw += knm[offset + j] * (sin_th[j] * ci - cos_th[j] * si)
        else:
            for j in range(n):
                pw += knm[offset + j] * \
                    sin(theta[j] - theta[i] - alpha[offset + j])
        deriv[i] = omegas[i] + pw
        if use3:
            deriv[i] += 2.0 * (gs * ci - gc * si) * \
                (gc * ci + gs * si) * inv_n2
        if zeta != 0.0:
            deriv[i] += zs_psi * ci - zc_psi * si


fn simplicial_run(
    phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    sigma2: Float64,
    dt: Float64,
    n_steps: Int,
    mut out: List[Float64],
) -> None:
    var two_pi = 6.283185307179586
    for i in range(n):
        out[i] = phases[i]
    var alpha_zero = True
    for a in alpha:
        if a != 0.0:
            alpha_zero = False
            break
    var deriv = List[Float64](capacity=n)
    for _ in range(n):
        deriv.append(0.0)
    for _ in range(n_steps):
        _compute_derivative(out, omegas, knm, alpha, n,
                            zeta, psi, sigma2, alpha_zero, deriv)
        for i in range(n):
            var raw = out[i] + dt * deriv[i]
            var v = raw - Float64(Int(raw / two_pi)) * two_pi
            if v < 0.0:
                v += two_pi
            out[i] = v


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "SIMP":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1
    var sigma2 = atof(tokens[idx]); idx += 1
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
    simplicial_run(phases, omegas, knm, alpha, n,
                   zeta, psi, sigma2, dt, n_steps, out)
    for i in range(n):
        print(out[i])
