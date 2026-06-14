# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Time-delayed Kuramoto integration (Mojo port)

"""Explicit-Euler time-delayed Kuramoto integration as a Mojo executable.

Stdin:

    DELAY n delay_steps n_steps zeta psi dt
        phases[0..n] omegas[0..n] knm[0..n*n] alpha[0..n*n]

Prints ``n`` final phases (one per line). A ring buffer of ``delay_steps + 1``
snapshots supplies the delayed source phase; the first ``delay_steps`` steps use
the current snapshot (zero-delay warmup), matching the NumPy reference.

Build with::

    mojo build mojo/delay.mojo -o mojo/delay_mojo -Xlinker -lm
"""

from std.math import sin, floor
from std.collections import List

alias TWO_PI = 6.283185307179586


fn delayed_kuramoto_run(
    phases_init: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    delay_steps: Int,
    n_steps: Int,
    mut out: List[Float64],
) -> None:
    var max_buf = delay_steps + 1
    var p = List[Float64](capacity=n)
    var newp = List[Float64](capacity=n)
    for j in range(n):
        p.append(phases_init[j])
        newp.append(0.0)
    var hist = List[Float64](capacity=max_buf * n)
    for _ in range(max_buf * n):
        hist.append(0.0)

    var alpha_zero = True
    for k in range(len(alpha)):
        if alpha[k] != 0.0:
            alpha_zero = False
            break

    for i in range(n_steps):
        var ring = i % max_buf
        for j in range(n):
            hist[ring * n + j] = p[j]
        var didx = ring
        if delay_steps > 0 and i >= delay_steps:
            didx = (i - delay_steps) % max_buf
        for ii in range(n):
            var theta_i = p[ii]
            var row = ii * n
            var coupling: Float64 = 0.0
            for jj in range(n):
                var dj = hist[didx * n + jj]
                if alpha_zero:
                    coupling += knm[row + jj] * sin(dj - theta_i)
                else:
                    coupling += knm[row + jj] * sin(dj - theta_i - alpha[row + jj])
            var dtheta = omegas[ii] + coupling
            if zeta != 0.0:
                dtheta += zeta * sin(psi - theta_i)
            var raw = theta_i + dt * dtheta
            var wrapped = raw - floor(raw / TWO_PI) * TWO_PI
            newp[ii] = wrapped
        for ii in range(n):
            p[ii] = newp[ii]

    for j in range(n):
        out[j] = p[j]


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "DELAY":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var delay_steps = Int(atol(tokens[idx])); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1

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
    delayed_kuramoto_run(
        phases, omegas, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps, out
    )
    for j in range(n):
        print(out[j])
