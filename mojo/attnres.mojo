# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Mojo multi-head)

"""Full multi-head AttnRes in Mojo.

Mojo 0.26 ``UnsafePointer`` C-ABI is still in transition, so this
file ships as a ``mojo build`` executable and communicates with
Python over a one-line whitespace-separated stdin / stdout protocol.
When the C-ABI surface stabilises in Mojo 0.27+ we swap to ctypes.

Stdin layout on one line (whitespace separated):

    n n_heads block_size temperature lambda
    knm[0..n*n]
    theta[0..n]
    w_q[0..n_heads*d_model*d_head]
    w_k[...]
    w_v[...]
    w_o[0..n_heads*d_head*d_model]

``block_size`` negative means full-N attention.

Stdout: ``n*n`` floats, one per line.
"""

from std.math import cos, exp, sqrt
from std.collections import List


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    var idx = 0
    var n = Int(atol(tokens[idx])); idx += 1
    var n_heads = Int(atol(tokens[idx])); idx += 1
    var block_size = Int(atol(tokens[idx])); idx += 1
    var temperature = atof(tokens[idx]); idx += 1
    var lambda_val = atof(tokens[idx]); idx += 1

    var nn = n * n
    var knm = List[Float64](capacity=nn)
    for _ in range(nn):
        knm.append(atof(tokens[idx])); idx += 1
    var theta = List[Float64](capacity=n)
    for _ in range(n):
        theta.append(atof(tokens[idx])); idx += 1

    # W matrices: d_model = 8 (PHASE_EMBED_DIM), d_head = d_model / n_heads.
    var d_model = 8
    var d_head = d_model // n_heads
    var qkv_len = n_heads * d_model * d_head
    var wo_len = n_heads * d_head * d_model

    var w_q = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        w_q.append(atof(tokens[idx])); idx += 1
    var w_k = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        w_k.append(atof(tokens[idx])); idx += 1
    var w_v = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        w_v.append(atof(tokens[idx])); idx += 1
    var w_o = List[Float64](capacity=wo_len)
    for _ in range(wo_len):
        w_o.append(atof(tokens[idx])); idx += 1

    # Allocate workspaces.
    var x = List[Float64](capacity=n * d_model)
    for _ in range(n * d_model):
        x.append(0.0)
    var q = List[Float64](capacity=n_heads * n * d_head)
    for _ in range(n_heads * n * d_head):
        q.append(0.0)
    var k = List[Float64](capacity=n_heads * n * d_head)
    for _ in range(n_heads * n * d_head):
        k.append(0.0)
    var v = List[Float64](capacity=n_heads * n * d_head)
    for _ in range(n_heads * n * d_head):
        v.append(0.0)
    var attn = List[Float64](capacity=n_heads * n * n)
    for _ in range(n_heads * n * n):
        attn.append(0.0)
    var concat_width = n_heads * d_head
    var concat = List[Float64](capacity=n * concat_width)
    for _ in range(n * concat_width):
        concat.append(0.0)
    var o = List[Float64](capacity=n * d_model)
    for _ in range(n * d_model):
        o.append(0.0)
    var o_norm = List[Float64](capacity=n)
    for _ in range(n):
        o_norm.append(0.0)
    var a_agg = List[Float64](capacity=nn)
    for _ in range(nn):
        a_agg.append(0.0)
    var rowwise = List[Float64](capacity=nn)
    for _ in range(nn):
        rowwise.append(0.0)

    # 1. Fourier-feature embedding.
    for i in range(n):
        for h_idx in range(d_model // 2):
            var freq = Float64(h_idx + 1)
            x[i * d_model + 2 * h_idx] = cos(freq * theta[i])
            x[i * d_model + 2 * h_idx + 1] = sin_fn(freq * theta[i])

    # 2. Per-head Q, K, V.
    for h in range(n_heads):
        for i in range(n):
            for e in range(d_head):
                var qs: Float64 = 0.0
                var ks: Float64 = 0.0
                var vs: Float64 = 0.0
                for d in range(d_model):
                    var xd = x[i * d_model + d]
                    var widx = h * d_model * d_head + d * d_head + e
                    qs += xd * w_q[widx]
                    ks += xd * w_k[widx]
                    vs += xd * w_v[widx]
                q[h * n * d_head + i * d_head + e] = qs
                k[h * n * d_head + i * d_head + e] = ks
                v[h * n * d_head + i * d_head + e] = vs

    # 3. Softmax attention.
    var inv_scale = 1.0 / (sqrt(Float64(d_head)) * temperature)
    for h in range(n_heads):
        for i in range(n):
            var row_logits = List[Float64](capacity=n)
            for _ in range(n):
                row_logits.append(Float64.MIN_FINITE)
            var any_unmasked = False
            for j in range(n):
                if j == i:
                    continue
                if knm[i * n + j] == 0.0:
                    continue
                if block_size >= 0:
                    var diff = i - j
                    if diff < 0:
                        diff = -diff
                    if diff > block_size:
                        continue
                var dot: Float64 = 0.0
                for e in range(d_head):
                    dot += q[h * n * d_head + i * d_head + e] * (
                        k[h * n * d_head + j * d_head + e]
                    )
                row_logits[j] = dot * inv_scale
                any_unmasked = True
            if not any_unmasked:
                continue
            var row_max = Float64.MIN_FINITE
            for j in range(n):
                if row_logits[j] > row_max:
                    row_max = row_logits[j]
            var denom: Float64 = 0.0
            for j in range(n):
                if row_logits[j] > Float64.MIN_FINITE:
                    var e_val = exp(row_logits[j] - row_max)
                    row_logits[j] = e_val
                    denom += e_val
                else:
                    row_logits[j] = 0.0
            if denom > 0.0:
                var inv_denom = 1.0 / denom
                for j in range(n):
                    attn[h * n * n + i * n + j] = row_logits[j] * inv_denom

    # 4. heads · V, concat.
    for h in range(n_heads):
        for i in range(n):
            for e in range(d_head):
                var s: Float64 = 0.0
                for j in range(n):
                    s += attn[h * n * n + i * n + j] * v[
                        h * n * d_head + j * d_head + e
                    ]
                concat[i * concat_width + h * d_head + e] = s

    # 5. Output projection.
    for i in range(n):
        for d_out in range(d_model):
            var s: Float64 = 0.0
            for c in range(concat_width):
                s += concat[i * concat_width + c] * w_o[c * d_model + d_out]
            o[i * d_model + d_out] = s

    # 6. Cosine similarity.
    for i in range(n):
        var s: Float64 = 0.0
        for d in range(d_model):
            var val = o[i * d_model + d]
            s += val * val
        o_norm[i] = sqrt(s) + 1e-12
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if knm[i * n + j] == 0.0:
                continue
            if block_size >= 0:
                var diff = i - j
                if diff < 0:
                    diff = -diff
                if diff > block_size:
                    continue
            var dot: Float64 = 0.0
            for d in range(d_model):
                dot += o[i * d_model + d] * o[j * d_model + d]
            var cos_sim = dot / (o_norm[i] * o_norm[j])
            a_agg[i * n + j] = 0.5 * (1.0 + cos_sim)

    # 7. Modulation + symmetrisation.
    if lambda_val == 0.0:
        for i in range(n):
            for j in range(n):
                print(knm[i * n + j])
        return

    for i in range(n):
        for j in range(n):
            rowwise[i * n + j] = knm[i * n + j] * (
                1.0 + lambda_val * a_agg[i * n + j]
            )
    for i in range(n):
        for j in range(n):
            var v_out: Float64 = 0.0
            if i != j:
                v_out = 0.5 * (rowwise[i * n + j] + rowwise[j * n + i])
            print(v_out)


fn sin_fn(x: Float64) -> Float64:
    from std.math import sin
    return sin(x)
