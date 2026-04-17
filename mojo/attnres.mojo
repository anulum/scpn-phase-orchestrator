# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Mojo port)

"""Mojo implementation of AttnRes state-dependent K modulation.

Mojo 0.26 is still stabilising the C-ABI story for non-scalar argument
types; ``UnsafePointer`` now demands an explicit ``origin`` parameter
that the C boundary cannot carry, and ``MutableAnyOrigin`` is not yet
public. Rather than lock the whole chain behind a single API rev, this
backend uses a text-based stdin/stdout protocol and the Python bridge
runs it through ``mojo run``.

Input (one float per line, in order):

* line 1: ``n``      — matrix dimension, integer
* line 2: ``bs``     — block_size, integer
* line 3: ``temp``   — temperature
* line 4: ``lam``    — lambda_
* lines 5…4+n²:      — ``knm`` row-major, one f64 per line
* lines 5+n²…4+n²+n: — ``theta``, one f64 per line

Output: ``n²`` lines, each one f64 of the flattened modulated matrix.

Build and run::

    mojo build mojo/attnres.mojo -o mojo/attnres_mojo
    ./mojo/attnres_mojo < input.txt > output.txt
"""

from std.math import cos, exp
from std.collections import List


fn main() raises:
    # Mojo 0.26's ``input()`` reads a single line; the Python bridge
    # therefore sends the full header + knm + theta payload on one
    # whitespace-separated line, which we split into tokens here.
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))
    var idx = 0

    var n = Int(atol(tokens[idx]))
    idx += 1
    var bs = Int(atol(tokens[idx]))
    idx += 1
    var temperature = atof(tokens[idx])
    idx += 1
    var lambda_val = atof(tokens[idx])
    idx += 1

    var nn = n * n
    var knm = List[Float64](capacity=nn)
    for _ in range(nn):
        knm.append(atof(tokens[idx]))
        idx += 1
    var theta = List[Float64](capacity=n)
    for _ in range(n):
        theta.append(atof(tokens[idx]))
        idx += 1

    var inv_t = 1.0 / temperature
    var rowwise = List[Float64](capacity=nn)
    for _ in range(nn):
        rowwise.append(0.0)
    var logits = List[Float64](capacity=n)
    for _ in range(n):
        logits.append(0.0)

    for i in range(n):
        for j in range(n):
            logits[j] = Float64.MIN_FINITE

        var lo = i - bs
        if lo < 0:
            lo = 0
        var hi = i + bs + 1
        if hi > n:
            hi = n

        var any_unmasked = False
        var row_off = i * n
        for j in range(lo, hi):
            if j == i:
                continue
            var kij = knm[row_off + j]
            if kij == 0.0:
                continue
            logits[j] = cos(theta[j] - theta[i]) * inv_t
            any_unmasked = True

        if any_unmasked:
            var row_max = Float64.MIN_FINITE
            for j in range(n):
                if logits[j] > row_max:
                    row_max = logits[j]
            var denom = 0.0
            for j in range(n):
                if logits[j] > Float64.MIN_FINITE:
                    var e = exp(logits[j] - row_max)
                    logits[j] = e
                    denom += e
                else:
                    logits[j] = 0.0
            if denom > 0.0:
                var inv_denom = 1.0 / denom
                for j in range(n):
                    logits[j] *= inv_denom
        else:
            for j in range(n):
                logits[j] = 0.0

        for j in range(n):
            rowwise[row_off + j] = knm[row_off + j] * (
                1.0 + lambda_val * logits[j]
            )

    for i in range(n):
        var row_i = i * n
        for j in range(n):
            var v: Float64 = 0.0
            if i != j:
                v = 0.5 * (rowwise[row_i + j] + rowwise[j * n + i])
            print(v)
