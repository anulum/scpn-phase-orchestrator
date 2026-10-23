# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension kernels (Mojo port)

"""Grassberger-Procaccia correlation integral + Kaplan-Yorke
dimension as a Mojo executable.

Stdin verbs:

* ``CI T d nP nK idx_i[0..nP] idx_j[0..nP] epsilons[0..nK]
     traj_flat[0..T*d]`` — prints ``nK`` correlation fractions.
* ``KY n lambda[0..n]`` — prints the Kaplan-Yorke scalar.

Build with::

    mojo build mojo/dimension.mojo -o mojo/dimension_mojo -Xlinker -lm
"""

from std.math import abs
from std.collections import List
from std.ffi import external_call
from std.sys import exit


fn correlation_integral(
    traj: List[Float64],
    t: Int,
    d: Int,
    idx_i: List[Int],
    idx_j: List[Int],
    epsilons: List[Float64],
    mut out: List[Float64],
) raises -> None:
    """Count validated nonself pairs with stable norms and strict finite radii."""
    if t < 0 or d <= 0 or len(idx_i) != len(idx_j):
        raise Error("invalid dimension metadata")
    if t > len(traj) // d or t * d != len(traj) or len(out) != len(epsilons):
        raise Error("dimension cardinality mismatch")
    for value in traj:
        require_finite(value)
    for k in range(len(epsilons)):
        require_finite(epsilons[k])
        if epsilons[k] < 0.0 or (k > 0 and epsilons[k] < epsilons[k - 1]):
            raise Error("radii must be nonnegative and sorted")
    for p in range(len(idx_i)):
        if idx_i[p] < 0 or idx_i[p] >= t or idx_j[p] < 0 or idx_j[p] >= t or idx_i[p] == idx_j[p]:
            raise Error("invalid nonself pair indices")
    var n_p = len(idx_i)
    var n_k = len(epsilons)
    if n_p == 0:
        for k in range(n_k):
            out[k] = 0.0
        return
    var dists = List[Float64](capacity=n_p)
    for _ in range(n_p):
        dists.append(0.0)
    for p in range(n_p):
        var i = idx_i[p]
        var j = idx_j[p]
        var s: Float64 = 0.0
        var base_i = i * d
        var base_j = j * d
        for k in range(d):
            var delta = traj[base_i + k] - traj[base_j + k]
            s = external_call["hypot", Float64](s, delta)
        dists[p] = s
    for k in range(n_k):
        var cnt = 0
        var eps = epsilons[k]
        for p in range(n_p):
            if dists[p] < eps:
                cnt += 1
        out[k] = Float64(cnt) / Float64(n_p)


fn sort_descending(mut xs: List[Float64], n: Int) -> None:
    """Sort the private spectrum copy into descending order."""
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] < key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


fn kaplan_yorke_dimension(lyap: List[Float64]) raises -> Float64:
    """Evaluate finite spectra with scaled compensated cumulative prefixes."""
    var n = len(lyap)
    var scale: Float64 = 0.0
    for value in lyap:
        require_finite(value)
        scale = max(scale, abs(value))
    if scale == 0.0:
        return Float64(n)
    var le = List[Float64](capacity=n)
    for value in lyap:
        le.append(value)
    sort_descending(le, n)
    var total: Float64 = 0.0
    var correction: Float64 = 0.0
    var accepted: Float64 = 0.0
    for i in range(n):
        var value = le[i] / scale
        var updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
        var prefix = total + correction
        if prefix < 0.0:
            return Float64(i) + accepted / abs(value)
        accepted = prefix
    return Float64(n)


fn main():
    """Emit protocol failures on stderr with nonzero status and no result batch."""
    try:
        run_request()
    except error:
        print(String(error), file=FileDescriptor(2))
        exit(1)


fn run_request() raises:
    """Read exactly one finite CI or KY request and emit only its valid result."""
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    if len(tokens) == 0:
        raise Error("empty dimension request")
    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "CI":
        if len(tokens) < 5:
            raise Error("incomplete CI header")
        var t = parse_count(tokens[idx]); idx += 1
        var d = parse_count(tokens[idx]); idx += 1
        var n_p = parse_count(tokens[idx]); idx += 1
        var n_k = parse_count(tokens[idx]); idx += 1
        if t < 0 or d <= 0 or n_p < 0 or n_k < 0:
            raise Error("invalid CI dimensions or counts")
        var remaining = len(tokens) - 5
        if n_p > remaining // 2:
            raise Error("pair counts exceed CI payload")
        remaining -= 2 * n_p
        if n_k > remaining:
            raise Error("radius count exceeds CI payload")
        remaining -= n_k
        if t > remaining // d or t * d != remaining:
            raise Error("CI payload must match trajectory cardinality")
        var idx_i = List[Int](capacity=n_p)
        for _ in range(n_p):
            idx_i.append(parse_count(tokens[idx])); idx += 1
        var idx_j = List[Int](capacity=n_p)
        for _ in range(n_p):
            idx_j.append(parse_count(tokens[idx])); idx += 1
        var epsilons = List[Float64](capacity=n_k)
        for _ in range(n_k):
            epsilons.append(parse_real(tokens[idx])); idx += 1
        var traj = List[Float64](capacity=t * d)
        for _ in range(t * d):
            traj.append(parse_real(tokens[idx])); idx += 1
        var out = List[Float64](capacity=n_k)
        for _ in range(n_k):
            out.append(0.0)
        correlation_integral(traj, t, d, idx_i, idx_j, epsilons, out)
        for k in range(n_k):
            print(out[k])

    elif op == "KY":
        if len(tokens) < 2:
            raise Error("incomplete KY header")
        var n = parse_count(tokens[idx]); idx += 1
        if n < 0 or n != len(tokens) - 2:
            raise Error("KY spectrum cardinality mismatch")
        var lyap = List[Float64](capacity=n)
        for _ in range(n):
            lyap.append(parse_real(tokens[idx])); idx += 1
        var result = kaplan_yorke_dimension(lyap)
        print(result)

    else:
        raise Error("expected CI or KY operation")


fn require_finite(value: Float64) raises:
    """Refuse nonfinite controls or measurements."""
    if not (value <= Float64.MAX_FINITE and value >= Float64.MIN_FINITE):
        raise Error("dimension numerical values must be finite")


fn parse_count(token: String) raises -> Int:
    """Parse a whole signed decimal count, refusing overflow and partial tokens."""
    var data = token.as_bytes()
    if len(data) == 0:
        raise Error("empty dimension count")
    var first = 0
    var negative = False
    if data[0] == 43 or data[0] == 45:
        negative = data[0] == 45
        first = 1
    if first == len(data):
        raise Error("dimension count needs decimal digits")
    var result = 0
    for i in range(first, len(data)):
        var digit = Int(data[i]) - 48
        if digit < 0 or digit > 9:
            raise Error("dimension counts must be whole decimal integers")
        if result > (Int.MAX - digit) // 10:
            raise Error("dimension count exceeds the native integer range")
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
                raise Error("misplaced dimension numerical sign")
        elif byte == 46:
            if decimal_point or exponent:
                raise Error("misplaced dimension decimal point")
            decimal_point = True
        elif byte == 69 or byte == 101:
            if exponent or not mantissa_digit:
                raise Error("misplaced dimension exponent")
            exponent = True
        else:
            raise Error("dimension values must be decimal numbers")
    if not mantissa_digit or (exponent and not exponent_digit):
        raise Error("incomplete dimension decimal number")
    var result = atof(token)
    require_finite(result)
    return result
