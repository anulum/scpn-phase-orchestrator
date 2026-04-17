# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes coupling modulation (Julia port)

"""
attnres.jl

Julia implementation of the AttnRes state-dependent coupling
modulation. Mirrors the Rust port at
``spo-kernel/crates/spo-engine/src/attnres.rs`` and the NumPy
reference at ``src/scpn_phase_orchestrator/coupling/attention_residuals.py``.

Called from Python via `juliacall` in the fallback chain when the
Rust and Mojo backends are unavailable.
"""

module AttnRes

export attnres_modulate

"""
    attnres_modulate(knm, theta, n, block_size, temperature, lambda_) -> Vector{Float64}

Compute the state-dependent modulation of a row-major ``N × N``
coupling matrix ``knm`` given the current phase vector ``theta``.

Contract matches the Rust / Python versions:

* Output is symmetric with zero diagonal.
* ``lambda_ == 0`` returns ``knm`` unchanged.
* Attention is restricted to ``|i − j| ≤ block_size``.
* Zero ``knm`` entries stay zero.
"""
function attnres_modulate(
    knm::AbstractVector{Float64},
    theta::AbstractVector{Float64},
    n::Integer,
    block_size::Integer,
    temperature::Float64,
    lambda_::Float64,
)
    if length(knm) != n * n
        error("knm length $(length(knm)) does not match n*n = $(n * n)")
    end
    if length(theta) != n
        error("theta length $(length(theta)) does not match n = $n")
    end
    if block_size < 1
        error("block_size must be ≥ 1")
    end
    if !(temperature > 0.0)
        error("temperature must be > 0")
    end
    if lambda_ < 0.0
        error("lambda_ must be ≥ 0")
    end

    if lambda_ == 0.0
        return copy(knm)
    end

    inv_t = 1.0 / temperature
    rowwise = zeros(Float64, n * n)

    @inbounds for i in 0:(n - 1)
        lo = max(0, i - block_size)
        hi = min(n - 1, i + block_size)

        # Logits + mask in a single pass.
        logits = fill(-Inf, n)
        any_unmasked = false
        for j in lo:hi
            if j == i
                continue
            end
            kij = knm[i * n + j + 1]  # Julia 1-indexed
            if kij == 0.0
                continue
            end
            logits[j + 1] = cos(theta[j + 1] - theta[i + 1]) * inv_t
            any_unmasked = true
        end

        # Numerically-stable softmax.
        attn = zeros(Float64, n)
        if any_unmasked
            row_max = maximum(logits)
            denom = 0.0
            for j in 1:n
                if isfinite(logits[j])
                    e = exp(logits[j] - row_max)
                    attn[j] = e
                    denom += e
                end
            end
            if denom > 0.0
                attn ./= denom
            end
        end

        for j in 0:(n - 1)
            rowwise[i * n + j + 1] =
                knm[i * n + j + 1] * (1.0 + lambda_ * attn[j + 1])
        end
    end

    # Symmetrise: (R + Rᵀ) / 2.
    out = zeros(Float64, n * n)
    @inbounds for i in 0:(n - 1)
        for j in 0:(n - 1)
            if i == j
                out[i * n + j + 1] = 0.0
            else
                out[i * n + j + 1] =
                    0.5 * (rowwise[i * n + j + 1] + rowwise[j * n + i + 1])
            end
        end
    end
    return out
end

end  # module
