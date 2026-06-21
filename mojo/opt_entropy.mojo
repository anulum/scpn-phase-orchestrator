# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ordinal-Pattern Transition Entropy (Mojo port)

"""Ordinal-pattern transition entropy as a Mojo executable.

Stdin line layout (whitespace separated):

* ``OPS n dimension delay series[0..n]`` — prints ``window_count`` integer
  ordinal-pattern codes, one per line.
* ``OTE n dimension delay series[0..n]`` — prints the single normalised
  transition entropy value.

Build with::

    mojo build mojo/opt_entropy.mojo -o mojo/opt_entropy_mojo -Xlinker -lm
"""

from std.math import log
from std.collections import List


fn factorial(value: Int) -> Int:
    var result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


fn window_count(length: Int, dimension: Int, delay: Int) -> Int:
    var span = (dimension - 1) * delay
    if length > span:
        return length - span
    return 0


fn stable_argsort(window: List[Float64], dimension: Int) -> List[Int]:
    var used = List[Int](capacity=dimension)
    for _ in range(dimension):
        used.append(0)
    var perm = List[Int](capacity=dimension)
    for _ in range(dimension):
        perm.append(0)
    for rank in range(dimension):
        var best = -1
        for idx in range(dimension):
            if used[idx] == 1:
                continue
            if (
                best == -1
                or window[idx] < window[best]
                or (window[idx] == window[best] and idx < best)
            ):
                best = idx
        perm[rank] = best
        used[best] = 1
    return perm^


fn lehmer_code(perm: List[Int], dimension: Int, fact: List[Int]) -> Int:
    var code = 0
    for i in range(dimension):
        var smaller = 0
        for j in range(i + 1, dimension):
            if perm[j] < perm[i]:
                smaller += 1
        code += smaller * fact[dimension - 1 - i]
    return code


fn ordinal_codes(
    series: List[Float64], n: Int, dimension: Int, delay: Int
) -> List[Int]:
    var count = window_count(n, dimension, delay)
    var fact = List[Int](capacity=dimension)
    for k in range(dimension):
        fact.append(factorial(k))
    var codes = List[Int]()
    for m in range(count):
        var window = List[Float64](capacity=dimension)
        for k in range(dimension):
            window.append(series[m + k * delay])
        codes.append(lehmer_code(stable_argsort(window, dimension), dimension, fact))
    return codes^


fn transition_entropy(
    series: List[Float64], n: Int, dimension: Int, delay: Int
) -> Float64:
    var codes = ordinal_codes(series, n, dimension, delay)
    var n_codes = len(codes)
    if n_codes < 2:
        return 0.0
    var fact_d = factorial(dimension)
    var total = n_codes - 1
    var keys = List[Int](capacity=total)
    for m in range(total):
        keys.append(codes[m] * fact_d + codes[m + 1])

    # Insertion sort on integer keys (ascending) to match the other backends.
    for a in range(1, total):
        var key = keys[a]
        var b = a - 1
        while b >= 0 and keys[b] > key:
            keys[b + 1] = keys[b]
            b -= 1
        keys[b + 1] = key

    var counts = List[Int]()
    var run = 1
    for idx in range(1, total):
        if keys[idx] == keys[idx - 1]:
            run += 1
        else:
            counts.append(run)
            run = 1
    counts.append(run)

    var distinct = len(counts)
    if distinct < 2:
        return 0.0
    var total_f = Float64(total)
    var entropy: Float64 = 0.0
    for count in counts:
        var probability = Float64(count) / total_f
        entropy -= probability * log(probability)
    var max_entropy = log(Float64(distinct))
    if max_entropy < 1e-15:
        return 0.0
    var value = entropy / max_entropy
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "OPS":
        var n = Int(atol(tokens[idx])); idx += 1
        var dimension = Int(atol(tokens[idx])); idx += 1
        var delay = Int(atol(tokens[idx])); idx += 1
        var series = List[Float64](capacity=n)
        for _ in range(n):
            series.append(atof(tokens[idx])); idx += 1
        var codes = ordinal_codes(series, n, dimension, delay)
        for c in codes:
            print(c)
    elif op == "OTE":
        var n = Int(atol(tokens[idx])); idx += 1
        var dimension = Int(atol(tokens[idx])); idx += 1
        var delay = Int(atol(tokens[idx])); idx += 1
        var series = List[Float64](capacity=n)
        for _ in range(n):
            series.append(atof(tokens[idx])); idx += 1
        print(transition_entropy(series, n, dimension, delay))
    else:
        print(-1)
