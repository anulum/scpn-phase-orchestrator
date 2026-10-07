# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Julia multi-head)

"""
attnres.jl

SPO multi-head phase attention coupling. Mirrors
``spo-kernel/crates/spo-engine/src/attnres.rs`` and the NumPy
reference at ``src/scpn_phase_orchestrator/coupling/attention_residuals.py``.

This spatial adaptation is not the depth-residual operator in arXiv:2603.15031.
Its components are Fourier-feature embedding,
per-head Q/K/V projections, scaled dot-product attention with
softmax, optional local mask, output projection, symmetric
pairwise aggregation.

Callable from Python via `juliacall` through
``src/scpn_phase_orchestrator/coupling/_attnres_julia.py``.
"""

module AttnRes

export attnres_modulate

"""
    attnres_modulate(knm, theta, w_q, w_k, w_v, w_o,
                     n, n_heads, block_size, temperature, lambda_) -> Vector{Float64}

Full multi-head AttnRes modulation. All inputs are flat ``Float64``
vectors in row-major order. ``block_size < 0`` means unbounded
(full-N) attention.
"""
function attnres_modulate(
    knm::AbstractVector{Float64},
    theta::AbstractVector{Float64},
    w_q::AbstractVector{Float64},
    w_k::AbstractVector{Float64},
    w_v::AbstractVector{Float64},
    w_o::AbstractVector{Float64},
    n::Integer,
    n_heads::Integer,
    block_size::Integer,
    temperature::Float64,
    lambda_::Float64,
)
    (n isa Bool || n_heads isa Bool || block_size isa Bool) &&
        error("count and band controls must not be boolean")
    # ── Validation ────────────────────────────────────────────────────
    n >= 0 && n <= isqrt(typemax(Int)) || error("invalid n*n size")
    length(knm) == n * n       || error("knm length mismatch")
    length(theta) == n         || error("theta length mismatch")
    n_heads >= 1               || error("n_heads must be ≥ 1")
    temperature > 0.0 && isfinite(temperature) ||
        error("temperature must be finite and > 0")
    lambda_ >= 0.0 && isfinite(lambda_) || error("lambda_ must be finite and ≥ 0")
    (block_size == -1 || block_size >= 1) || error("block_size must be -1 or positive")
    all(values -> all(isfinite, values), (knm, theta, w_q, w_k, w_v, w_o)) ||
        error("inputs must contain only finite values")
    for i in 0:(n - 1)
        abs(knm[i * n + i + 1]) <= 1e-12 || error("knm diagonal must be zero")
        for j in 0:(n - 1)
            reverse = knm[j * n + i + 1]
            abs(knm[i * n + j + 1] - reverse) <= 1e-12 + 1e-12 * abs(reverse) ||
                error("knm must be symmetric")
        end
    end
    length(w_q) == length(w_k) || error("w_k shape mismatch")
    length(w_q) == length(w_v) || error("w_v shape mismatch")
    length(w_q) % n_heads == 0 || error("w_q not divisible by n_heads")

    d_model = isqrt(length(w_o))
    d_model >= 2 && iseven(d_model) && d_model^2 == length(w_o) ||
        error("w_o must encode an even square d_model")
    d_model % n_heads == 0 || error("d_model must be divisible by n_heads")
    length(w_q) == length(w_o) || error("w_q/w_o shape mismatch")
    d_head = d_model ÷ n_heads
    if n == 0 || lambda_ == 0.0
        return copy(knm)
    end

    # ── 1. Fourier-feature embedding ─────────────────────────────────
    x = zeros(Float64, n * d_model)
    @inbounds for i in 0:(n - 1)
        for h_idx in 0:(d_model ÷ 2 - 1)
            freq = Float64(h_idx + 1)
            x[i * d_model + 2 * h_idx + 1] = cos(freq * theta[i + 1])
            x[i * d_model + 2 * h_idx + 2] = sin(freq * theta[i + 1])
        end
    end

    # ── 2. Per-head Q, K, V via X·W. ──────────────────────────────────
    q = zeros(Float64, n_heads * n * d_head)
    k = zeros(Float64, n_heads * n * d_head)
    v = zeros(Float64, n_heads * n * d_head)
    @inbounds for h in 0:(n_heads - 1)
        for i in 0:(n - 1)
            for e in 0:(d_head - 1)
                qs = 0.0; ks = 0.0; vs = 0.0
                for d in 0:(d_model - 1)
                    xd = x[i * d_model + d + 1]
                    idx = h * d_model * d_head + d * d_head + e + 1
                    qs += xd * w_q[idx]
                    ks += xd * w_k[idx]
                    vs += xd * w_v[idx]
                end
                q[h * n * d_head + i * d_head + e + 1] = qs
                k[h * n * d_head + i * d_head + e + 1] = ks
                v[h * n * d_head + i * d_head + e + 1] = vs
            end
        end
    end

    # ── 3. Attention logits + softmax per head ───────────────────────
    all(values -> all(isfinite, values), (q, k, v)) || error("attention projections must be finite")
    inv_scale = (1.0 / sqrt(Float64(d_head))) / temperature
    isfinite(inv_scale) || error("attention scale must be finite")
    attn = zeros(Float64, n_heads * n * n)
    @inbounds for h in 0:(n_heads - 1)
        for i in 0:(n - 1)
            row_logits = fill(-Inf, n)
            any_unmasked = false
            for j in 0:(n - 1)
                i == j && continue
                knm[i * n + j + 1] == 0.0 && continue
                if block_size >= 0 && abs(i - j) > block_size
                    continue
                end
                dot = 0.0
                for e in 0:(d_head - 1)
                    dot += q[h * n * d_head + i * d_head + e + 1] *
                           k[h * n * d_head + j * d_head + e + 1]
                end
                row_logits[j + 1] = dot * inv_scale
                isfinite(row_logits[j + 1]) || error("attention logits must be finite")
                any_unmasked = true
            end
            any_unmasked || continue
            row_max = maximum(row_logits)
            denom = 0.0
            for j in 1:n
                if isfinite(row_logits[j])
                    e = exp(row_logits[j] - row_max)
                    row_logits[j] = e
                    denom += e
                else
                    row_logits[j] = 0.0
                end
            end
            if denom > 0.0
                inv_denom = 1.0 / denom
                for j in 0:(n - 1)
                    attn[h * n * n + i * n + j + 1] =
                        row_logits[j + 1] * inv_denom
                end
            end
        end
    end

    # ── 4. heads = attn · V, concat along d-axis ─────────────────────
    concat_width = n_heads * d_head
    concat = zeros(Float64, n * concat_width)
    @inbounds for h in 0:(n_heads - 1)
        for i in 0:(n - 1)
            for e in 0:(d_head - 1)
                s = 0.0
                for j in 0:(n - 1)
                    s += attn[h * n * n + i * n + j + 1] *
                         v[h * n * d_head + j * d_head + e + 1]
                end
                concat[i * concat_width + h * d_head + e + 1] = s
            end
        end
    end

    # ── 5. Output projection: o = concat · W_O ───────────────────────
    o = zeros(Float64, n * d_model)
    @inbounds for i in 0:(n - 1)
        for d_out in 0:(d_model - 1)
            s = 0.0
            for c in 0:(concat_width - 1)
                s += concat[i * concat_width + c + 1] *
                     w_o[c * d_model + d_out + 1]
            end
            o[i * d_model + d_out + 1] = s
        end
    end

    # ── 6. Pairwise cosine similarity a_agg ───────────────────────────
    o_norm = zeros(Float64, n)
    @inbounds for i in 0:(n - 1)
        s = 0.0
        for d in 0:(d_model - 1)
            val = o[i * d_model + d + 1]
            s += val * val
        end
        o_norm[i + 1] = sqrt(s) + 1e-12
        isfinite(o_norm[i + 1]) || error("attention output norms must be finite")
    end
    a_agg = zeros(Float64, n * n)
    @inbounds for i in 0:(n - 1)
        for j in 0:(n - 1)
            i == j && continue
            knm[i * n + j + 1] == 0.0 && continue
            if block_size >= 0 && abs(i - j) > block_size
                continue
            end
            dot = 0.0
            for d in 0:(d_model - 1)
                dot += (o[i * d_model + d + 1] / o_norm[i + 1]) *
                       (o[j * d_model + d + 1] / o_norm[j + 1])
            end
            cos_sim = clamp(dot, -1.0, 1.0)
            a_agg[i * n + j + 1] = 0.5 * (1.0 + cos_sim)
        end
    end

    # ── 7. Row-wise modulation + symmetrisation ──────────────────────
    rowwise = zeros(Float64, n * n)
    @inbounds for i in 0:(n - 1)
        for j in 0:(n - 1)
            rowwise[i * n + j + 1] = knm[i * n + j + 1] *
                (1.0 + lambda_ * a_agg[i * n + j + 1])
        end
    end
    out = zeros(Float64, n * n)
    @inbounds for i in 0:(n - 1)
        for j in 0:(n - 1)
            if i != j
                out[i * n + j + 1] = 0.5 * rowwise[i * n + j + 1] +
                    0.5 * rowwise[j * n + i + 1]
                isfinite(out[i * n + j + 1]) || error("modulated coupling must be finite")
            end
        end
    end
    return out
end

end  # module
