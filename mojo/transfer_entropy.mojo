# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy (Mojo port)

"""Phase transfer entropy as a Mojo executable.

Stdin line layout (whitespace separated):

* ``PTE n n_bins source[0..n] target[0..n]`` — single TE value.
* ``MAT n_osc n_time n_bins series[0..n_osc*n_time]`` — flat
  ``n_osc × n_osc`` matrix row-major.

Build with::

    mojo build mojo/transfer_entropy.mojo -o mojo/transfer_entropy_mojo -Xlinker -lm
"""

from std.math import log
from std.collections import List


fn conditional_entropy(
    target: List[Int], condition: List[Int], n_cond_bins: Int
) -> Float64:
    var n = len(target)
    if n == 0:
        return 0.0
    # bucket-by-condition. Use parallel arrays: condition_groups[c] is
    # the list of target values whose condition equals c.
    var groups = List[List[Int]](capacity=n_cond_bins)
    for _ in range(n_cond_bins):
        groups.append(List[Int]())
    for i in range(n):
        var c = condition[i]
        if c >= 0 and c < n_cond_bins:
            groups[c].append(target[i])

    var h: Float64 = 0.0
    for c in range(n_cond_bins):
        var count = len(groups[c])
        if count < 2:
            continue
        # Count unique values in this bucket using a simple nested loop
        # (N is small here; typical count ≤ n_bins²).
        var unique_values = List[Int]()
        var unique_counts = List[Int]()
        for v in groups[c]:
            var found = False
            for k in range(len(unique_values)):
                if unique_values[k] == v:
                    unique_counts[k] = unique_counts[k] + 1
                    found = True
                    break
            if not found:
                unique_values.append(v)
                unique_counts.append(1)
        var sub: Float64 = 0.0
        for k in range(len(unique_values)):
            var p = Float64(unique_counts[k]) / Float64(count)
            sub += p * log(p + 1e-30)
        h -= (Float64(count) / Float64(n)) * sub
    return h


fn phase_te(
    source: List[Float64],
    target: List[Float64],
    n_bins: Int,
) -> Float64:
    if len(source) < 3 or len(target) < 3:
        return 0.0
    var two_pi = 2.0 * 3.141592653589793
    var n = len(source)
    if len(target) < n:
        n = len(target)
    n -= 1
    var bin_width = two_pi / Float64(n_bins)

    var src_bin = List[Int](capacity=n)
    var tgt_bin = List[Int](capacity=n)
    var tgt_next = List[Int](capacity=n)
    for _ in range(n):
        src_bin.append(0)
        tgt_bin.append(0)
        tgt_next.append(0)
    for i in range(n):
        var s = source[i] % two_pi
        if s < 0:
            s += two_pi
        var t = target[i] % two_pi
        if t < 0:
            t += two_pi
        var tn = target[i + 1] % two_pi
        if tn < 0:
            tn += two_pi
        var sb = Int(s / bin_width)
        if sb >= n_bins:
            sb = n_bins - 1
        var tb = Int(t / bin_width)
        if tb >= n_bins:
            tb = n_bins - 1
        var nb = Int(tn / bin_width)
        if nb >= n_bins:
            nb = n_bins - 1
        src_bin[i] = sb
        tgt_bin[i] = tb
        tgt_next[i] = nb

    var h_y_yt = conditional_entropy(tgt_next, tgt_bin, n_bins)
    var joint = List[Int](capacity=n)
    for i in range(n):
        joint.append(tgt_bin[i] * n_bins + src_bin[i])
    var h_y_yt_x = conditional_entropy(tgt_next, joint, n_bins * n_bins)
    var te = h_y_yt - h_y_yt_x
    if te < 0.0:
        return 0.0
    return te


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var op = tokens[idx]; idx += 1

    if op == "PTE":
        var n = Int(atol(tokens[idx])); idx += 1
        var n_bins = Int(atol(tokens[idx])); idx += 1
        var src = List[Float64](capacity=n)
        for _ in range(n):
            src.append(atof(tokens[idx])); idx += 1
        var tgt = List[Float64](capacity=n)
        for _ in range(n):
            tgt.append(atof(tokens[idx])); idx += 1
        print(phase_te(src, tgt, n_bins))
    elif op == "MAT":
        var n_osc = Int(atol(tokens[idx])); idx += 1
        var n_time = Int(atol(tokens[idx])); idx += 1
        var n_bins = Int(atol(tokens[idx])); idx += 1
        var total = n_osc * n_time
        var series = List[Float64](capacity=total)
        for _ in range(total):
            series.append(atof(tokens[idx])); idx += 1
        var src = List[Float64](capacity=n_time)
        var tgt = List[Float64](capacity=n_time)
        for _ in range(n_time):
            src.append(0.0)
            tgt.append(0.0)
        for i in range(n_osc):
            for s in range(n_time):
                src[s] = series[i * n_time + s]
            for j in range(n_osc):
                if i == j:
                    print(0.0)
                    continue
                for s in range(n_time):
                    tgt[s] = series[j * n_time + s]
                print(phase_te(src, tgt, n_bins))
    else:
        print(-1)
