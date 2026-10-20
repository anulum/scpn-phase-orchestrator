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

from std.math import sin
from std.collections import List
from std.ffi import external_call
from std.sys import exit

comptime TWO_PI = 6.283185307179586


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
) raises -> None:
    """Apply simultaneous Euler steps to validated finite delay buffers."""
    if n <= 0 or delay_steps < 0 or n_steps < 0:
        raise Error("invalid delay dimensions or counts")
    if len(phases_init) != n or len(omegas) != n or len(out) != n:
        raise Error("delay vectors must match n")
    if n > len(knm) // n or len(knm) != n * n or len(alpha) != n * n:
        raise Error("delay matrices must match n squared")
    require_finite(dt)
    require_finite(zeta)
    require_finite(psi)
    if dt <= 0.0:
        raise Error("dt must be positive")
    for value in phases_init:
        require_finite(value)
    for value in omegas:
        require_finite(value)
    for value in knm:
        require_finite(value)
    for value in alpha:
        require_finite(value)
    if n_steps == 0:
        for j in range(n):
            out[j] = phases_init[j]
        return
    var effective_delay = min(delay_steps, n_steps)
    if effective_delay == Int.MAX or effective_delay + 1 > Int.MAX // n:
        raise Error("delay history size overflow")
    var max_buf = effective_delay + 1
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
            require_finite(raw)
            var wrapped = external_call["fmod", Float64](raw, TWO_PI)
            if wrapped < 0.0:
                wrapped += TWO_PI
            if wrapped >= TWO_PI or wrapped == 0.0:
                wrapped = 0.0
            newp[ii] = wrapped
        for ii in range(n):
            p[ii] = newp[ii]

    for j in range(n):
        out[j] = p[j]


fn main():
    """Keep protocol errors on stderr with a nonzero process status."""
    try:
        run_request()
    except error:
        print(String(error), file=FileDescriptor(2))
        exit(1)


fn run_request() raises:
    """Read one exact DELAY request and emit only a successful numerical batch."""
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    if len(tokens) < 7:
        raise Error("incomplete DELAY header")

    var idx = 0
    var op = tokens[idx]; idx += 1
    if op != "DELAY":
        raise Error("expected DELAY operation")

    var n = parse_count(tokens[idx]); idx += 1
    var delay_steps = parse_count(tokens[idx]); idx += 1
    var n_steps = parse_count(tokens[idx]); idx += 1
    var zeta = parse_real(tokens[idx]); idx += 1
    var psi = parse_real(tokens[idx]); idx += 1
    var dt = parse_real(tokens[idx]); idx += 1

    if n <= 0 or delay_steps < 0 or n_steps < 0:
        raise Error("invalid DELAY dimensions or counts")
    var remaining = len(tokens) - 7
    if n > remaining // 2 or n > remaining // n // 2:
        raise Error("DELAY dimensions exceed payload")
    if remaining - 2 * n != 2 * n * n:
        raise Error("DELAY payload must contain exact vectors and matrices")

    var phases = List[Float64](capacity=n)
    for _ in range(n):
        phases.append(parse_real(tokens[idx])); idx += 1
    var omegas = List[Float64](capacity=n)
    for _ in range(n):
        omegas.append(parse_real(tokens[idx])); idx += 1
    var knm = List[Float64](capacity=n * n)
    for _ in range(n * n):
        knm.append(parse_real(tokens[idx])); idx += 1
    var alpha = List[Float64](capacity=n * n)
    for _ in range(n * n):
        alpha.append(parse_real(tokens[idx])); idx += 1

    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(0.0)
    delayed_kuramoto_run(
        phases, omegas, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps, out
    )
    for j in range(n):
        print(out[j])


fn require_finite(value: Float64) raises:
    """Refuse nonfinite controls, inputs or computed phases."""
    if not (value <= Float64.MAX_FINITE and value >= Float64.MIN_FINITE):
        raise Error("DELAY numerical values must be finite")


fn parse_count(token: String) raises -> Int:
    """Parse a whole signed decimal count, refusing overflow and partial tokens."""
    var data = token.as_bytes()
    if len(data) == 0:
        raise Error("empty DELAY count")
    var first = 0
    var negative = False
    if data[0] == 43 or data[0] == 45:
        negative = data[0] == 45
        first = 1
    if first == len(data):
        raise Error("DELAY count needs decimal digits")
    var result = 0
    for i in range(first, len(data)):
        var digit = Int(data[i]) - 48
        if digit < 0 or digit > 9:
            raise Error("DELAY counts must be whole decimal integers")
        if result > (Int.MAX - digit) // 10:
            raise Error("DELAY count exceeds the native integer range")
        result = result * 10 + digit
    return -result if negative else result


fn parse_real(token: String) raises -> Float64:
    """Parse a complete finite decimal/exponent value before C conversion."""
    var data = token.as_bytes()
    var mantissa_digit = False
    var exponent_digit = False
    var exponent = False
    var decimal_point = False
    for i in range(len(data)):
        var byte = Int(data[i])
        if byte >= 48 and byte <= 57:
            mantissa_digit = True
            if exponent:
                exponent_digit = True
        elif byte == 43 or byte == 45:
            if i != 0 and data[i - 1] != 69 and data[i - 1] != 101:
                raise Error("misplaced DELAY numerical sign")
        elif byte == 46:
            if decimal_point or exponent:
                raise Error("misplaced DELAY decimal point")
            decimal_point = True
        elif byte == 69 or byte == 101:
            if exponent or not mantissa_digit:
                raise Error("misplaced DELAY exponent")
            exponent = True
        else:
            raise Error("DELAY values must be decimal numbers")
    if not mantissa_digit or (exponent and not exponent_digit):
        raise Error("incomplete DELAY decimal number")
    var result = atof(token)
    require_finite(result)
    return result
