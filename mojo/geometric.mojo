# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus symplectic Euler (Mojo port)

"""Torus-preserving geometric integrator as a Mojo executable.

Stdin:

    TORUS n zeta psi dt n_steps
          phases[0..n] omegas[0..n]
          knm[0..n*n] alpha[0..n*n]

Prints ``n`` final-phase floats (in [0, 2π)) on stdout.

Build with::

    mojo build mojo/geometric.mojo -o mojo/geometric_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt, atan2
from std.collections import List


fn torus_run(
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
    var two_pi = 6.283185307179586
    var z_re = List[Float64](capacity=n)
    var z_im = List[Float64](capacity=n)
    for i in range(n):
        z_re.append(cos(phases[i]))
        z_im.append(sin(phases[i]))

    var alpha_zero = True
    for a in alpha:
        if a != 0.0:
            alpha_zero = False
            break
    var zs_psi: Float64 = 0.0
    var zc_psi: Float64 = 0.0
    if zeta != 0.0:
        zs_psi = zeta * sin(psi)
        zc_psi = zeta * cos(psi)

    var next_re = List[Float64](capacity=n)
    var next_im = List[Float64](capacity=n)
    for _ in range(n):
        next_re.append(0.0)
        next_im.append(0.0)

    for _ in range(n_steps):
        for i in range(n):
            var coupling: Float64 = 0.0
            var offset = i * n
            if alpha_zero:
                for j in range(n):
                    coupling += knm[offset + j] * \
                        (z_im[j] * z_re[i] - z_re[j] * z_im[i])
            else:
                var ti = atan2(z_im[i], z_re[i])
                for j in range(n):
                    var tj = atan2(z_im[j], z_re[j])
                    coupling += knm[offset + j] * \
                        sin(tj - ti - alpha[offset + j])
            var omega_eff = omegas[i] + coupling
            if zeta != 0.0:
                omega_eff += zs_psi * z_re[i] - zc_psi * z_im[i]
            var angle = omega_eff * dt
            var sin_a = sin(angle)
            var cos_a = cos(angle)
            var nr = z_re[i] * cos_a - z_im[i] * sin_a
            var ni = z_re[i] * sin_a + z_im[i] * cos_a
            var norm_ = sqrt(nr * nr + ni * ni)
            if norm_ > 0.0:
                next_re[i] = nr / norm_
                next_im[i] = ni / norm_
            else:
                next_re[i] = nr
                next_im[i] = ni
        for i in range(n):
            z_re[i] = next_re[i]
            z_im[i] = next_im[i]

    for i in range(n):
        var raw = atan2(z_im[i], z_re[i])
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
    if op != "TORUS":
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
    torus_run(phases, omegas, knm, alpha, n,
              zeta, psi, dt, n_steps, out)
    for i in range(n):
        print(out[i])
