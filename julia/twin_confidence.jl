# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin divergence (Julia port)

"""
twin_confidence.jl — digital-twin divergence kernel.

    twin_divergence(model_phases, observed_phases, model_order, observed_order,
                    n, w, n_bins) -> Vector{Float64} of length 2

Returns ``[phase_js_divergence, order_wasserstein]``: the phase-histogram
Jensen–Shannon divergence (natural log, range ``[0, ln 2]``) and the
order-parameter one-dimensional Wasserstein-1 distance (range ``[0, 1]``).
Matches the NumPy / Rust / Go / Mojo references to 1e-9.
"""

module TwinConfidence

export twin_divergence

const TWO_PI = 2.0 * pi

function phase_histogram(phases::AbstractVector{Float64}, n_bins::Integer)
    width = TWO_PI / Float64(n_bins)
    counts = zeros(Float64, n_bins)
    for phase in phases
        wrapped = phase - floor(phase / TWO_PI) * TWO_PI
        idx = Int(floor(wrapped / width))
        if idx < 0
            idx = 0
        elseif idx > n_bins - 1
            idx = n_bins - 1
        end
        counts[idx + 1] += 1.0
    end
    total = sum(counts)
    if total <= 0.0
        return fill(1.0 / Float64(n_bins), n_bins)
    end
    return counts ./ total
end

function kl_divergence(p::AbstractVector{Float64}, m::AbstractVector{Float64})
    acc = 0.0
    @inbounds for i in eachindex(p)
        if p[i] > 0.0
            acc += p[i] * log(p[i] / m[i])
        end
    end
    return acc
end

function jensen_shannon(p::AbstractVector{Float64}, q::AbstractVector{Float64})
    m = 0.5 .* (p .+ q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
end

function wasserstein1(
    model_order::AbstractVector{Float64},
    observed_order::AbstractVector{Float64},
)
    sorted_model = sort(collect(Float64, model_order))
    sorted_obs = sort(collect(Float64, observed_order))
    acc = 0.0
    @inbounds for i in eachindex(sorted_model)
        acc += abs(sorted_model[i] - sorted_obs[i])
    end
    return acc / Float64(length(sorted_model))
end

function twin_divergence(
    model_phases::AbstractVector{Float64},
    observed_phases::AbstractVector{Float64},
    model_order::AbstractVector{Float64},
    observed_order::AbstractVector{Float64},
    n::Integer,
    w::Integer,
    n_bins::Integer,
)
    (length(model_phases) == n && length(observed_phases) == n) ||
        error("phase vector lengths must equal n")
    (length(model_order) == w && length(observed_order) == w) ||
        error("order vector lengths must equal w")
    n_bins >= 1 || error("n_bins must be a positive integer")
    p = phase_histogram(model_phases, n_bins)
    q = phase_histogram(observed_phases, n_bins)
    js = jensen_shannon(p, q)
    w1 = wasserstein1(model_order, observed_order)
    return Float64[js, w1]
end

end # module TwinConfidence
