# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE batched integrator (Mojo port)

"""Batched Kuramoto / Sakaguchi UPDE integrator as a Mojo executable.

Three methods — Euler, RK4, Dormand-Prince RK45 with adaptive step
control — matching the Rust reference (`spo-engine/src/upde.rs`)
bit-for-bit modulo text-stream rounding.

Stdin layout (single whitespace-separated line):

    RUN n zeta psi dt n_steps method_id n_substeps atol rtol
        phases[0..n] omegas[0..n]
        knm_flat[0..n*n] alpha_flat[0..n*n]

``method_id`` encodes 0 = Euler, 1 = RK4, 2 = RK45. Prints the ``n``
final phases (wrapped to ``[0, 2π)``) one per line.

Build with::

    mojo build mojo/upde_engine.mojo -o mojo/upde_engine_mojo -Xlinker -lm
"""

from std.math import sin, cos, abs
from std.collections import List


# Dormand-Prince tableau (shared with spo-engine/src/dp_tableau.rs).
alias A21: Float64 = 1.0 / 5.0
alias A31: Float64 = 3.0 / 40.0
alias A32: Float64 = 9.0 / 40.0
alias A41: Float64 = 44.0 / 45.0
alias A42: Float64 = -56.0 / 15.0
alias A43: Float64 = 32.0 / 9.0
alias A51: Float64 = 19372.0 / 6561.0
alias A52: Float64 = -25360.0 / 2187.0
alias A53: Float64 = 64448.0 / 6561.0
alias A54: Float64 = -212.0 / 729.0
alias A61: Float64 = 9017.0 / 3168.0
alias A62: Float64 = -355.0 / 33.0
alias A63: Float64 = 46732.0 / 5247.0
alias A64: Float64 = 49.0 / 176.0
alias A65: Float64 = -5103.0 / 18656.0
alias B5_0: Float64 = 35.0 / 384.0
alias B5_2: Float64 = 500.0 / 1113.0
alias B5_3: Float64 = 125.0 / 192.0
alias B5_4: Float64 = -2187.0 / 6784.0
alias B5_5: Float64 = 11.0 / 84.0
alias B4_0: Float64 = 5179.0 / 57600.0
alias B4_2: Float64 = 7571.0 / 16695.0
alias B4_3: Float64 = 393.0 / 640.0
alias B4_4: Float64 = -92097.0 / 339200.0
alias B4_5: Float64 = 187.0 / 2100.0
alias B4_6: Float64 = 1.0 / 40.0


fn compute_derivative(
    theta: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    zeta: Float64,
    psi: Float64,
    n: Int,
    mut out: List[Float64],
) -> None:
    for i in range(n):
        var s: Float64 = 0.0
        var offset = i * n
        for j in range(n):
            s += knm[offset + j] * sin(
                theta[j] - theta[i] - alpha[offset + j]
            )
        var driving: Float64 = 0.0
        if zeta != 0.0:
            driving = zeta * sin(psi - theta[i])
        out[i] = omegas[i] + s + driving


fn euler_substep(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    n: Int,
    mut buf: List[Float64],
) -> None:
    compute_derivative(phases, omegas, knm, alpha, zeta, psi, n, buf)
    for i in range(n):
        phases[i] = phases[i] + dt * buf[i]


fn rk4_substep(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    n: Int,
    mut k1: List[Float64],
    mut k2: List[Float64],
    mut k3: List[Float64],
    mut k4: List[Float64],
    mut tmp: List[Float64],
) -> None:
    compute_derivative(phases, omegas, knm, alpha, zeta, psi, n, k1)
    for i in range(n):
        tmp[i] = phases[i] + 0.5 * dt * k1[i]
    compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k2)
    for i in range(n):
        tmp[i] = phases[i] + 0.5 * dt * k2[i]
    compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k3)
    for i in range(n):
        tmp[i] = phases[i] + dt * k3[i]
    compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k4)
    var dt6 = dt / 6.0
    for i in range(n):
        phases[i] = phases[i] + dt6 * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
        )


fn pow_f64(x: Float64, p: Float64) -> Float64:
    # Mojo 0.26 stdlib `pow` is integer-only for Float64 in scalar form.
    # We only need x^(−0.2) and x^(−0.25) with positive x, so use
    # exp(p · log(x)). Local `ln`/`exp` via math intrinsics.
    from std.math import exp, log as log_fn
    if x <= 0.0:
        return 0.0
    return exp(p * log_fn(x))


fn rk45_step(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    zeta: Float64,
    psi: Float64,
    atol: Float64,
    rtol: Float64,
    dt_config: Float64,
    last_dt: Float64,
    n: Int,
    mut k1: List[Float64],
    mut k2: List[Float64],
    mut k3: List[Float64],
    mut k4: List[Float64],
    mut k5: List[Float64],
    mut k6: List[Float64],
    mut k7: List[Float64],
    mut y5: List[Float64],
    mut tmp: List[Float64],
) -> Float64:
    var dt = last_dt
    for _ in range(4):
        compute_derivative(phases, omegas, knm, alpha, zeta, psi, n, k1)
        for i in range(n):
            tmp[i] = phases[i] + dt * A21 * k1[i]
        compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k2)
        for i in range(n):
            tmp[i] = phases[i] + dt * (A31 * k1[i] + A32 * k2[i])
        compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k3)
        for i in range(n):
            tmp[i] = phases[i] + dt * (
                A41 * k1[i] + A42 * k2[i] + A43 * k3[i]
            )
        compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k4)
        for i in range(n):
            tmp[i] = phases[i] + dt * (
                A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]
            )
        compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k5)
        for i in range(n):
            tmp[i] = phases[i] + dt * (
                A61 * k1[i] + A62 * k2[i] + A63 * k3[i]
                + A64 * k4[i] + A65 * k5[i]
            )
        compute_derivative(tmp, omegas, knm, alpha, zeta, psi, n, k6)
        for i in range(n):
            y5[i] = phases[i] + dt * (
                B5_0 * k1[i] + B5_2 * k3[i] + B5_3 * k4[i]
                + B5_4 * k5[i] + B5_5 * k6[i]
            )
        compute_derivative(y5, omegas, knm, alpha, zeta, psi, n, k7)
        var err_norm: Float64 = 0.0
        for i in range(n):
            var y4_i = phases[i] + dt * (
                B4_0 * k1[i] + B4_2 * k3[i] + B4_3 * k4[i]
                + B4_4 * k5[i] + B4_5 * k6[i] + B4_6 * k7[i]
            )
            var err_i = abs(y5[i] - y4_i)
            var ap = abs(phases[i])
            var ay = abs(y5[i])
            var mx = ap
            if ay > mx:
                mx = ay
            var scale = atol + rtol * mx
            var ratio = err_i / scale
            if ratio > err_norm:
                err_norm = ratio
        if err_norm <= 1.0:
            var factor: Float64 = 5.0
            if err_norm > 0.0:
                var cand = 0.9 * pow_f64(err_norm, -0.2)
                if cand < 5.0:
                    factor = cand
            var new_last_dt = dt * factor
            var cap = dt_config * 10.0
            if new_last_dt > cap:
                new_last_dt = cap
            for i in range(n):
                phases[i] = y5[i]
            return new_last_dt
        var factor = 0.9 * pow_f64(err_norm, -0.25)
        if factor < 0.2:
            factor = 0.2
        dt = dt * factor
    for i in range(n):
        phases[i] = y5[i]
    return dt


fn fmod_positive(x: Float64, m: Float64) -> Float64:
    var r = x - Float64(Int(x / m)) * m
    if r < 0.0:
        r += m
    return r


fn upde_run(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm: List[Float64],
    alpha: List[Float64],
    n: Int,
    zeta: Float64,
    psi: Float64,
    dt: Float64,
    n_steps: Int,
    method: Int,
    n_substeps: Int,
    atol: Float64,
    rtol: Float64,
) -> None:
    var two_pi = 6.283185307179586
    var k1 = List[Float64](capacity=n)
    var k2 = List[Float64](capacity=n)
    var k3 = List[Float64](capacity=n)
    var k4 = List[Float64](capacity=n)
    var k5 = List[Float64](capacity=n)
    var k6 = List[Float64](capacity=n)
    var k7 = List[Float64](capacity=n)
    var y5 = List[Float64](capacity=n)
    var tmp = List[Float64](capacity=n)
    for _ in range(n):
        k1.append(0.0); k2.append(0.0); k3.append(0.0); k4.append(0.0)
        k5.append(0.0); k6.append(0.0); k7.append(0.0)
        y5.append(0.0); tmp.append(0.0)

    var last_dt = dt
    var sub_dt = dt / Float64(n_substeps)

    for _ in range(n_steps):
        if method == 2:
            last_dt = rk45_step(
                phases, omegas, knm, alpha, zeta, psi,
                atol, rtol, dt, last_dt, n,
                k1, k2, k3, k4, k5, k6, k7, y5, tmp,
            )
        elif method == 1:
            for _ in range(n_substeps):
                rk4_substep(
                    phases, omegas, knm, alpha, zeta, psi, sub_dt, n,
                    k1, k2, k3, k4, tmp,
                )
        else:
            for _ in range(n_substeps):
                euler_substep(
                    phases, omegas, knm, alpha, zeta, psi, sub_dt, n, k1,
                )
        for i in range(n):
            phases[i] = fmod_positive(phases[i], two_pi)


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "RUN":
        print(-1)
        return

    var n = Int(atol(tokens[idx])); idx += 1
    var zeta = atof(tokens[idx]); idx += 1
    var psi = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_steps = Int(atol(tokens[idx])); idx += 1
    var method = Int(atol(tokens[idx])); idx += 1
    var n_substeps = Int(atol(tokens[idx])); idx += 1
    var atol_ = atof(tokens[idx]); idx += 1
    var rtol_ = atof(tokens[idx]); idx += 1

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(atof(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(atof(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(atof(tokens[idx])); idx += 1
    var alpha_arr = List[Float64](capacity=n * n)
    for _ in range(n * n):
        alpha_arr.append(atof(tokens[idx])); idx += 1

    upde_run(
        phases, omegas, knm, alpha_arr,
        n, zeta, psi, dt, n_steps, method, n_substeps, atol_, rtol_,
    )
    for i in range(n):
        print(phases[i])
