# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen reduction (Mojo port)

"""Ott-Antonsen mean-field RK4 integrator as a Mojo executable.

Stdin:

    OARUN z_re z_im omega_0 delta k_coupling dt n_steps

Prints four floats on stdout: ``re``, ``im``, ``R = |z|``,
``ψ = arg(z)`` (one per line).

Build with::

    mojo build mojo/reduction.mojo -o mojo/reduction_mojo -Xlinker -lm
"""

from std.math import sqrt, atan2
from std.collections import List


fn _oa_deriv(
    re: Float64, im: Float64,
    omega_0: Float64, delta: Float64, half_k: Float64,
) -> Tuple[Float64, Float64]:
    var abs_sq = re * re + im * im
    var lin_re = -delta * re + omega_0 * im
    var lin_im = -delta * im - omega_0 * re
    var cubic_factor = half_k * (1.0 - abs_sq)
    var cub_re = cubic_factor * re
    var cub_im = cubic_factor * im
    return (lin_re + cub_re, lin_im + cub_im)


fn oa_run(
    z_re: Float64,
    z_im: Float64,
    omega_0: Float64,
    delta: Float64,
    k_coupling: Float64,
    dt: Float64,
    n_steps: Int,
) -> Tuple[Float64, Float64, Float64, Float64]:
    var re = z_re
    var im = z_im
    var half_k = k_coupling / 2.0
    for _ in range(n_steps):
        var (k1r, k1i) = _oa_deriv(re, im, omega_0, delta, half_k)
        var (k2r, k2i) = _oa_deriv(
            re + 0.5 * dt * k1r, im + 0.5 * dt * k1i,
            omega_0, delta, half_k,
        )
        var (k3r, k3i) = _oa_deriv(
            re + 0.5 * dt * k2r, im + 0.5 * dt * k2i,
            omega_0, delta, half_k,
        )
        var (k4r, k4i) = _oa_deriv(
            re + dt * k3r, im + dt * k3i,
            omega_0, delta, half_k,
        )
        re += (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        im += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
    var r = sqrt(re * re + im * im)
    var psi = atan2(im, re)
    return (re, im, r, psi)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "OARUN":
        print(-1)
        return

    var z_re = atof(tokens[idx]); idx += 1
    var z_im = atof(tokens[idx]); idx += 1
    var omega_0 = atof(tokens[idx]); idx += 1
    var delta = atof(tokens[idx]); idx += 1
    var k_coupling = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1

    var (re, im, r, psi) = oa_run(
        z_re, z_im, omega_0, delta, k_coupling, dt, n_steps,
    )
    print(re)
    print(im)
    print(r)
    print(psi)
