# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov spectrum (Mojo port)

"""Benettin 1980 / Shimada-Nagashima 1979 Lyapunov spectrum as a Mojo
executable. RK4 integration on the (phases, Q) tangent-space pair plus
periodic row-oriented Modified Gram-Schmidt to accumulate log-stretch
factors. Matches the Rust / NumPy / Julia / Go reference implementations
bit-for-bit up to float rounding.

Stdin line layout (single whitespace-separated payload):

    SPEC n dt n_steps qr_interval zeta psi
         phases_init[0..n] omegas[0..n]
         knm_flat[0..n*n] alpha_flat[0..n*n]

Prints ``n`` Lyapunov exponents (sorted descending), one per line.

Build with::

    mojo build mojo/lyapunov.mojo -o mojo/lyapunov_mojo -Xlinker -lm
"""

from std.math import sin, cos, log, sqrt, abs
from std.collections import List


fn kuramoto_rhs(
    phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    mut out: List[Float64],
) -> None:
    for i in range(n):
        var s: Float64 = 0.0
        for j in range(n):
            s += knm[i * n + j] * sin(
                phases[j] - phases[i] - alpha[i * n + j]
            )
        var driving: Float64 = 0.0
        if zeta != 0.0:
            driving = zeta * sin(psi - phases[i])
        out[i] = omegas[i] + s + driving


fn kuramoto_jacobian(
    phases: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    mut J: List[Float64],
) -> None:
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i * n + j] = 0.0
            else:
                J[i * n + j] = knm[i * n + j] * cos(
                    phases[j] - phases[i] - alpha[i * n + j]
                )
    for i in range(n):
        var s: Float64 = 0.0
        for j in range(n):
            if i != j:
                s += J[i * n + j]
        var driver_diag: Float64 = 0.0
        if zeta != 0.0:
            driver_diag = zeta * cos(psi - phases[i])
        J[i * n + i] = -(s + driver_diag)


fn mat_mul(
    A: List[Float64],
    B: List[Float64],
    mut out: List[Float64],
    n: Int,
) -> None:
    for i in range(n):
        for k in range(n):
            var s: Float64 = 0.0
            for j in range(n):
                s += A[i * n + j] * B[j * n + k]
            out[i * n + k] = s


fn row_mgs(
    mut Q: List[Float64], n: Int, mut diagR: List[Float64]
) -> None:
    """Row-oriented Modified Gram-Schmidt with two-pass reorthogonalisation.
    Matches the Rust kernel's ``modified_gram_schmidt`` convention.
    """
    for j in range(n):
        for _pass in range(2):
            for k in range(j):
                var dot: Float64 = 0.0
                for i in range(n):
                    dot += Q[k * n + i] * Q[j * n + i]
                for i in range(n):
                    Q[j * n + i] = Q[j * n + i] - dot * Q[k * n + i]
        var norm_sq: Float64 = 0.0
        for i in range(n):
            var v = Q[j * n + i]
            norm_sq += v * v
        var norm = sqrt(norm_sq)
        diagR[j] = norm
        if norm > 1e-300:
            var inv = 1.0 / norm
            for i in range(n):
                Q[j * n + i] = Q[j * n + i] * inv


fn sort_descending(mut xs: List[Float64], n: Int) -> None:
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] < key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


fn fmod_positive(x: Float64, m: Float64) -> Float64:
    var r = x - (Float64(Int(x / m))) * m
    if r < 0.0:
        r += m
    return r


fn lyapunov_spectrum(
    phases_init: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    dt: Float64,
    n_steps: Int,
    qr_interval: Int,
    zeta: Float64,
    psi: Float64,
) -> List[Float64]:
    var two_pi = 6.283185307179586
    var dt6 = dt / 6.0
    var nn = n * n

    var phases = List[Float64](capacity=n)
    for i in range(n):
        phases.append(phases_init[i])

    var Q = List[Float64](capacity=nn)
    for _ in range(nn):
        Q.append(0.0)
    for i in range(n):
        Q[i * n + i] = 1.0

    var k1p = List[Float64](capacity=n)
    var k2p = List[Float64](capacity=n)
    var k3p = List[Float64](capacity=n)
    var k4p = List[Float64](capacity=n)
    for _ in range(n):
        k1p.append(0.0)
        k2p.append(0.0)
        k3p.append(0.0)
        k4p.append(0.0)
    var tmp_p = List[Float64](capacity=n)
    for _ in range(n):
        tmp_p.append(0.0)
    var k1q = List[Float64](capacity=nn)
    var k2q = List[Float64](capacity=nn)
    var k3q = List[Float64](capacity=nn)
    var k4q = List[Float64](capacity=nn)
    for _ in range(nn):
        k1q.append(0.0)
        k2q.append(0.0)
        k3q.append(0.0)
        k4q.append(0.0)
    var tmp_q = List[Float64](capacity=nn)
    for _ in range(nn):
        tmp_q.append(0.0)
    var J = List[Float64](capacity=nn)
    for _ in range(nn):
        J.append(0.0)
    var diagR = List[Float64](capacity=n)
    for _ in range(n):
        diagR.append(0.0)
    var exponents = List[Float64](capacity=n)
    for _ in range(n):
        exponents.append(0.0)

    var total_time: Float64 = 0.0

    for step in range(n_steps):
        # Stage 1
        kuramoto_rhs(phases, omegas, knm, alpha, n, zeta, psi, k1p)
        kuramoto_jacobian(phases, knm, alpha, n, zeta, psi, J)
        mat_mul(J, Q, k1q, n)

        # Stage 2
        for i in range(n):
            tmp_p[i] = phases[i] + 0.5 * dt * k1p[i]
        for i in range(nn):
            tmp_q[i] = Q[i] + 0.5 * dt * k1q[i]
        kuramoto_rhs(tmp_p, omegas, knm, alpha, n, zeta, psi, k2p)
        kuramoto_jacobian(tmp_p, knm, alpha, n, zeta, psi, J)
        mat_mul(J, tmp_q, k2q, n)

        # Stage 3
        for i in range(n):
            tmp_p[i] = phases[i] + 0.5 * dt * k2p[i]
        for i in range(nn):
            tmp_q[i] = Q[i] + 0.5 * dt * k2q[i]
        kuramoto_rhs(tmp_p, omegas, knm, alpha, n, zeta, psi, k3p)
        kuramoto_jacobian(tmp_p, knm, alpha, n, zeta, psi, J)
        mat_mul(J, tmp_q, k3q, n)

        # Stage 4
        for i in range(n):
            tmp_p[i] = phases[i] + dt * k3p[i]
        for i in range(nn):
            tmp_q[i] = Q[i] + dt * k3q[i]
        kuramoto_rhs(tmp_p, omegas, knm, alpha, n, zeta, psi, k4p)
        kuramoto_jacobian(tmp_p, knm, alpha, n, zeta, psi, J)
        mat_mul(J, tmp_q, k4q, n)

        # Combine + wrap.
        for i in range(n):
            var nxt = phases[i] + dt6 * (
                k1p[i] + 2.0 * k2p[i] + 2.0 * k3p[i] + k4p[i]
            )
            phases[i] = fmod_positive(nxt, two_pi)
        for i in range(nn):
            Q[i] = Q[i] + dt6 * (
                k1q[i] + 2.0 * k2q[i] + 2.0 * k3q[i] + k4q[i]
            )
        total_time += dt

        # Periodic QR.
        if (step + 1) % qr_interval == 0:
            row_mgs(Q, n, diagR)
            for i in range(n):
                var d = abs(diagR[i])
                if d < 1e-300:
                    d = 1e-300
                exponents[i] = exponents[i] + log(d)

    if total_time > 0.0:
        for i in range(n):
            exponents[i] = exponents[i] / total_time

    sort_descending(exponents, n)
    return exponents^


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "SPEC":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1
    var qr_interval = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1

    var phases_init = List[Float64](capacity=n)
    for _ in range(n):
        phases_init.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var alpha = List[Float64](capacity=n * n)
    for _ in range(n * n):
        alpha.append(atof(tokens[idx])); idx += 1

    var out = lyapunov_spectrum(
        phases_init, omegas, knm, alpha,
        n, dt, n_steps, qr_interval, zeta, psi,
    )
    for i in range(n):
        print(out[i])
