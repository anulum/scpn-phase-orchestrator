# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD-with-control solve (Mojo port)

# Reads ``EDMD k n_lift m n_state reg <x_lift> <inputs> <y_lift> <states>`` from
# stdin (row-major flat matrices) and prints the row-major flat A, B then C, one
# scalar per line. The least-squares algebra matches the NumPy reference.
#
# Build with::
#
#     mojo build mojo/koopman_edmd.mojo -o mojo/koopman_edmd_mojo -Xlinker -lm

from std.collections import List


fn fabsf(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


fn solve_multi(
    dim: Int, mat: List[Float64], rhs: List[Float64], n_rhs: Int
) -> List[Float64]:
    var width = dim + n_rhs
    var aug = List[Float64](capacity=dim * width)
    for _ in range(dim * width):
        aug.append(0.0)
    for r in range(dim):
        for c in range(dim):
            aug[r * width + c] = mat[r * dim + c]
        for c in range(n_rhs):
            aug[r * width + dim + c] = rhs[r * n_rhs + c]

    for col in range(dim):
        var pivot_row = col
        var pivot_mag = fabsf(aug[col * width + col])
        for r in range(col + 1, dim):
            var mag = fabsf(aug[r * width + col])
            if mag > pivot_mag:
                pivot_mag = mag
                pivot_row = r
        if pivot_row != col:
            for c in range(width):
                var tmp = aug[col * width + c]
                aug[col * width + c] = aug[pivot_row * width + c]
                aug[pivot_row * width + c] = tmp
        var pivot = aug[col * width + col]
        for r in range(dim):
            if r == col:
                continue
            var factor = aug[r * width + col] / pivot
            for c in range(col, width):
                aug[r * width + c] = aug[r * width + c] - factor * aug[col * width + c]

    var sol = List[Float64](capacity=dim * n_rhs)
    for _ in range(dim * n_rhs):
        sol.append(0.0)
    for r in range(dim):
        var pivot = aug[r * width + r]
        for c in range(n_rhs):
            sol[r * n_rhs + c] = aug[r * width + dim + c] / pivot
    return sol^


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]
    idx += 1
    if op != "EDMD":
        print(-1)
        return

    var k = Int(atol(tokens[idx]))
    idx += 1
    var n_lift = Int(atol(tokens[idx]))
    idx += 1
    var m = Int(atol(tokens[idx]))
    idx += 1
    var n_state = Int(atol(tokens[idx]))
    idx += 1
    var reg = atof(tokens[idx])
    idx += 1

    var x_lift = List[Float64](capacity=k * n_lift)
    for _ in range(k * n_lift):
        x_lift.append(atof(tokens[idx]))
        idx += 1
    var inputs = List[Float64](capacity=k * m)
    for _ in range(k * m):
        inputs.append(atof(tokens[idx]))
        idx += 1
    var y_lift = List[Float64](capacity=k * n_lift)
    for _ in range(k * n_lift):
        y_lift.append(atof(tokens[idx]))
        idx += 1
    var states = List[Float64](capacity=k * n_state)
    for _ in range(k * n_state):
        states.append(atof(tokens[idx]))
        idx += 1

    var p = n_lift + m
    var gram = List[Float64](capacity=p * p)
    for _ in range(p * p):
        gram.append(0.0)
    var cross = List[Float64](capacity=p * n_lift)
    for _ in range(p * n_lift):
        cross.append(0.0)
    for i in range(k):
        for a in range(p):
            var phi_a: Float64
            if a < n_lift:
                phi_a = x_lift[i * n_lift + a]
            else:
                phi_a = inputs[i * m + (a - n_lift)]
            for b in range(p):
                var phi_b: Float64
                if b < n_lift:
                    phi_b = x_lift[i * n_lift + b]
                else:
                    phi_b = inputs[i * m + (b - n_lift)]
                gram[a * p + b] = gram[a * p + b] + phi_a * phi_b
            for j in range(n_lift):
                cross[a * n_lift + j] = cross[a * n_lift + j] + phi_a * y_lift[
                    i * n_lift + j
                ]
    for a in range(p):
        gram[a * p + a] = gram[a * p + a] + reg
    var m_sol = solve_multi(p, gram, cross, n_lift)

    for i in range(n_lift):
        for c in range(n_lift):
            print(m_sol[c * n_lift + i])
    for i in range(n_lift):
        for c in range(m):
            print(m_sol[(n_lift + c) * n_lift + i])

    var lift_gram = List[Float64](capacity=n_lift * n_lift)
    for _ in range(n_lift * n_lift):
        lift_gram.append(0.0)
    var cc = List[Float64](capacity=n_lift * n_state)
    for _ in range(n_lift * n_state):
        cc.append(0.0)
    for i in range(k):
        for a in range(n_lift):
            var xa = x_lift[i * n_lift + a]
            for b in range(n_lift):
                lift_gram[a * n_lift + b] = lift_gram[a * n_lift + b] + xa * x_lift[
                    i * n_lift + b
                ]
            for j in range(n_state):
                cc[a * n_state + j] = cc[a * n_state + j] + xa * states[
                    i * n_state + j
                ]
    for a in range(n_lift):
        lift_gram[a * n_lift + a] = lift_gram[a * n_lift + a] + reg
    var ct = solve_multi(n_lift, lift_gram, cc, n_state)
    for i in range(n_state):
        for c in range(n_lift):
            print(ct[c * n_state + i])
