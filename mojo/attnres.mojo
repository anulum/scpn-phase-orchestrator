# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Mojo multi-head)

"""SPO phase attention coupling through a versioned text protocol.

This spatial Q/K/V adaptation is not the paper's depth-residual operator.
The original five-field D=8 request remains supported. V2 sends:

    V2 n n_heads block_size temperature lambda d_model buffers...

Buffers are row-major K[N*N], theta[N], then four projections[D*D].
Exactly N*N finite values are returned. PROTOCOL returns 2 for negotiation.
"""

from std.math import cos, exp, sqrt, abs
from std.collections import List


fn main() raises:
    var line = input()
    var tokens = List[String]()
    for tok in line.split():
        tokens.append(String(tok))

    if len(tokens) == 1 and tokens[0] == "PROTOCOL":
        print(2)
        return
    var versioned = len(tokens) > 0 and tokens[0] == "V2"
    var idx = Int(versioned)
    var header_size = 5
    if versioned:
        header_size = 7
    if len(tokens) < header_size:
        raise Error("incomplete AttnRes header")
    var n = Int(atol(tokens[idx])); idx += 1
    var n_heads = Int(atol(tokens[idx])); idx += 1
    var block_size = Int(atol(tokens[idx])); idx += 1
    var temperature = atof(tokens[idx]); idx += 1
    var lambda_val = atof(tokens[idx]); idx += 1
    var d_model = 8
    if versioned:
        d_model = Int(atol(tokens[idx])); idx += 1
    if n < 0 or n_heads < 1 or d_model < 2 or d_model % 2 != 0:
        raise Error("invalid oscillator/head/model dimensions")
    if d_model % n_heads != 0:
        raise Error("d_model must be divisible by n_heads")
    if block_size != -1 and block_size < 1:
        raise Error("block_size must be -1 or positive")
    require_finite(temperature)
    require_finite(lambda_val)
    if temperature <= 0.0 or lambda_val < 0.0:
        raise Error("invalid temperature or lambda")
    # Bound products by the actual input length before allocating buffers.
    if n > len(tokens) or d_model > len(tokens):
        raise Error("dimensions exceed request length")
    if n > 0 and n > len(tokens) // n:
        raise Error("coupling dimensions exceed request length")
    if d_model > len(tokens) // d_model:
        raise Error("projection dimensions exceed request length")
    if len(tokens) - idx != n * n + n + 4 * d_model * d_model:
        raise Error("AttnRes buffer cardinality mismatch")
    var d_head = d_model // n_heads

    var nn = n * n
    var knm = List[Float64](capacity=nn)
    for _ in range(nn):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        knm.append(value)
    var theta = List[Float64](capacity=n)
    for _ in range(n):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        theta.append(value)

    var qkv_len = n_heads * d_model * d_head
    var wo_len = n_heads * d_head * d_model

    var w_q = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        w_q.append(value)
    var w_k = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        w_k.append(value)
    var w_v = List[Float64](capacity=qkv_len)
    for _ in range(qkv_len):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        w_v.append(value)
    var w_o = List[Float64](capacity=wo_len)
    for _ in range(wo_len):
        var value = atof(tokens[idx]); idx += 1
        require_finite(value)
        w_o.append(value)

    for i in range(n):
        if abs(knm[i * n + i]) > 1e-12:
            raise Error("knm diagonal must be zero")
        for j in range(n):
            var reverse = knm[j * n + i]
            if abs(knm[i * n + j] - reverse) > 1e-12 + 1e-12 * abs(reverse):
                raise Error("knm must be symmetric")
    if n == 0 or lambda_val == 0.0:
        for value in knm:
            print(value)
        return

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
    for value in q:
        require_finite(value)
    for value in k:
        require_finite(value)
    for value in v:
        require_finite(value)
    var inv_scale = (1.0 / sqrt(Float64(d_head))) / temperature
    require_finite(inv_scale)
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
                require_finite(row_logits[j])
                any_unmasked = True
            if not any_unmasked:
                continue
            var row_max = Float64.MIN_FINITE
            for j in range(n):
                if row_logits[j] > row_max:
                    row_max = row_logits[j]
            var denom: Float64 = 0.0
            for j in range(n):
                if i != j and knm[i * n + j] != 0.0 and (
                    block_size == -1 or abs(i - j) <= block_size
                ):
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
        require_finite(o_norm[i])
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
                dot += (o[i * d_model + d] / o_norm[i]) * (o[j * d_model + d] / o_norm[j])
            var cos_sim = max(Float64(-1.0), min(Float64(1.0), dot))
            a_agg[i * n + j] = 0.5 * (1.0 + cos_sim)

    # 7. Modulation + symmetrisation.
    for i in range(n):
        for j in range(n):
            rowwise[i * n + j] = knm[i * n + j] * (
                1.0 + lambda_val * a_agg[i * n + j]
            )
    for i in range(n):
        for j in range(n):
            var v_out: Float64 = 0.0
            if i != j:
                v_out = 0.5 * rowwise[i * n + j] + 0.5 * rowwise[j * n + i]
                require_finite(v_out)
            print(v_out)


fn sin_fn(x: Float64) -> Float64:
    from std.math import sin
    return sin(x)


fn require_finite(value: Float64) raises:
    """Refuse NaN and infinities before arithmetic or publication."""
    if not (value <= Float64.MAX_FINITE and value >= Float64.MIN_FINITE):
        raise Error("AttnRes numerical values must be finite")
