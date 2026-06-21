# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ordinal-Pattern Transition Entropy (Julia port)

"""
opt_entropy.jl

Ordinal-pattern transition entropy of a scalar series. Bandt–Pompe ordinal
patterns (Lehmer-encoded stable argsort) feed a consecutive-pattern
transition distribution whose normalised Shannon entropy is returned in
``[0, 1]``. Matches the NumPy, Rust, Go, and Mojo references bit-for-bit.
"""

module OptEntropy

export ordinal_pattern_sequence, transition_entropy

function _factorial(value::Int)
    result = Int64(1)
    for factor in 2:value
        result *= Int64(factor)
    end
    return result
end


function _window_count(length::Int, dimension::Int, delay::Int)
    span = (dimension - 1) * delay
    return length > span ? length - span : 0
end


# Stable ascending argsort with index tie-breaking; positions are 1-based but
# the inversion count is identical to the 0-based reference.
function _stable_argsort(window::Vector{Float64}, dimension::Int)
    used = falses(dimension)
    perm = zeros(Int, dimension)
    for rank in 1:dimension
        best = -1
        for idx in 1:dimension
            used[idx] && continue
            if best == -1 || window[idx] < window[best] ||
               (window[idx] == window[best] && idx < best)
                best = idx
            end
        end
        perm[rank] = best
        used[best] = true
    end
    return perm
end


function _lehmer_code(perm::Vector{Int}, dimension::Int, fact::Vector{Int64})
    code = Int64(0)
    for i in 1:dimension
        smaller = Int64(0)
        for j in (i + 1):dimension
            if perm[j] < perm[i]
                smaller += 1
            end
        end
        code += smaller * fact[dimension - i + 1]
    end
    return code
end


"""
    ordinal_pattern_sequence(series, dimension, delay) -> Vector{Int64}

Lehmer-encoded ordinal-pattern code sequence.
"""
function ordinal_pattern_sequence(
    series::AbstractVector{Float64}, dimension::Int, delay::Int
)
    n = length(series)
    count = _window_count(n, dimension, delay)
    fact = Int64[_factorial(k) for k in 0:(dimension - 1)]
    codes = zeros(Int64, count)
    window = zeros(Float64, dimension)
    @inbounds for m in 0:(count - 1)
        for k in 0:(dimension - 1)
            window[k + 1] = series[m + k * delay + 1]
        end
        codes[m + 1] = _lehmer_code(_stable_argsort(window, dimension), dimension, fact)
    end
    return codes
end


"""
    transition_entropy(series, dimension, delay) -> Float64

Normalised ordinal-pattern transition entropy in ``[0, 1]``.
"""
function transition_entropy(
    series::AbstractVector{Float64}, dimension::Int, delay::Int
)
    codes = ordinal_pattern_sequence(series, dimension, delay)
    n_codes = length(codes)
    n_codes >= 2 || return 0.0
    fact_d = _factorial(dimension)
    total = n_codes - 1
    keys = Vector{Int64}(undef, total)
    @inbounds for m in 1:total
        keys[m] = codes[m] * fact_d + codes[m + 1]
    end
    sort!(keys)

    counts = Int64[]
    run = Int64(1)
    @inbounds for idx in 2:length(keys)
        if keys[idx] == keys[idx - 1]
            run += 1
        else
            push!(counts, run)
            run = 1
        end
    end
    push!(counts, run)

    distinct = length(counts)
    distinct >= 2 || return 0.0
    total_f = Float64(total)
    entropy = 0.0
    for count in counts
        probability = Float64(count) / total_f
        entropy -= probability * log(probability)
    end
    max_entropy = log(Float64(distinct))
    max_entropy < 1e-15 && return 0.0
    return clamp(entropy / max_entropy, 0.0, 1.0)
end

end  # module
