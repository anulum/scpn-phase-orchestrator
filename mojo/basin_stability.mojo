# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Steady-state R (Mojo port)

"""One-trial finite-window Kuramoto mean R as a Mojo executable.

Stdin:

    STEADY n k_scale dt n_transient n_measure
           phases[0..n] omegas[0..n]
           knm[0..n*n] alpha[0..n*n]

Prints a single Float64 ``R`` on stdout.

Build with::

    mojo build mojo/basin_stability.mojo -o mojo/basin_stability_mojo -Xlinker -lm
"""

from std.math import sin, cos, sqrt
from std.collections import List


fn _kuramoto_step(
    mut phases: List[Float64],
    omegas: List[Float64],
    knm_flat: List[Float64],
    alpha_flat: List[Float64],
    n: Int,
    k_scale: Float64,
    dt: Float64,
) raises -> None:
    var old = List[Float64](capacity=n)
    for i in range(n):
        old.append(phases[i])
    for i in range(n):
        var coupling: Float64 = 0.0
        var base = i * n
        var theta_i = old[i]
        for j in range(n):
            var k_ij = knm_flat[base + j] * k_scale
            require_finite(k_ij)
            if k_ij == 0.0:
                continue
            var a_ij = alpha_flat[base + j]
            var angle = old[j] - theta_i - a_ij
            require_finite(angle)
            coupling += k_ij * sin(angle)
        var velocity = omegas[i] + coupling
        require_finite(velocity)
        phases[i] = theta_i + dt * velocity
        require_finite(phases[i])


fn _order_parameter(phases: List[Float64], n: Int) -> Float64:
    # The admitted native trial requires n to be positive.
    var nn = Float64(n)
    var sum_cos: Float64 = 0.0
    var sum_sin: Float64 = 0.0
    for i in range(n):
        sum_cos += cos(phases[i])
        sum_sin += sin(phases[i])
    var c = sum_cos / nn
    var s = sum_sin / nn
    return sqrt(c * c + s * s)


fn steady_state_r(
    phases_init: List[Float64],
    omegas: List[Float64],
    knm_flat: List[Float64],
    alpha_flat: List[Float64],
    n: Int,
    k_scale: Float64,
    dt: Float64,
    n_transient: Int,
    n_measure: Int,
) raises -> Float64:
    if n <= 0 or n_transient < 0 or n_measure < 0:
        raise Error("invalid trial dimensions or counts")
    if len(phases_init) != n or len(omegas) != n:
        raise Error("trial vectors must match n")
    if n > len(knm_flat) // n or len(knm_flat) != n * n or len(alpha_flat) != n * n:
        raise Error("trial matrices must match n squared")
    require_finite(k_scale)
    require_finite(dt)
    if dt <= 0.0:
        raise Error("dt must be positive")
    for value in phases_init:
        require_finite(value)
    for value in omegas:
        require_finite(value)
    for value in knm_flat:
        require_finite(value)
    for value in alpha_flat:
        require_finite(value)
    if n_measure == 0:
        return 0.0
    var phases = List[Float64](capacity=n)
    for i in range(n):
        phases.append(phases_init[i])
    for _ in range(n_transient):
        _kuramoto_step(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
    var r_sum: Float64 = 0.0
    for _ in range(n_measure):
        _kuramoto_step(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
        r_sum += _order_parameter(phases, n)
    var r = r_sum / Float64(n_measure)
    require_finite(r)
    if r < 0.0 or r > 1.0 + 1e-12:
        raise Error("mean R must lie in [0, 1]")
    return min(r, Float64(1.0))


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    if len(tokens) < 6:
        raise Error("incomplete STEADY header")
    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "STEADY":
        raise Error("unknown operation; expected STEADY")

    var n = Int(atol(tokens[idx])); idx += 1
    var k_scale = atof(tokens[idx]); idx += 1
    var dt = atof(tokens[idx]); idx += 1
    var n_transient = Int(atol(tokens[idx])); idx += 1
    var n_measure = Int(atol(tokens[idx])); idx += 1

    if n <= 0 or n_transient < 0 or n_measure < 0:
        raise Error("positive n and nonnegative step counts required")
    require_finite(k_scale)
    require_finite(dt)
    if dt <= 0.0:
        raise Error("dt must be positive")
    # Bound every product using the actual request before allocating buffers.
    if n > len(tokens) or n > len(tokens) // n:
        raise Error("dimensions exceed request length")
    var remaining = len(tokens) - idx
    if n > remaining // 2:
        raise Error("vector dimensions exceed request length")
    if n * n != (remaining - 2 * n) // 2 or remaining != 2 * n + 2 * n * n:
        raise Error("STEADY buffer cardinality mismatch")

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        phases.append(value)
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        omegas.append(value)
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        knm.append(value)
    var alpha = List[Float64](capacity=n * n)
    for _ in range(n * n):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        alpha.append(value)

    var r = steady_state_r(
        phases, omegas, knm, alpha,
        n, k_scale, dt, n_transient, n_measure,
    )
    print(r)


fn require_finite(value: Float64) raises:
    """Reject nonfinite values before numerical computation or publication."""
    if not (value <= Float64.MAX_FINITE and value >= Float64.MIN_FINITE):
        raise Error("STEADY numerical values must be finite")
