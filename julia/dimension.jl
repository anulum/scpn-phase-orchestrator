# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension kernels (Julia port)

"""
dimension.jl

* ``correlation_integral(traj_flat, T, d, idx_i, idx_j, epsilons)``
  — Grassberger-Procaccia pair-counting at caller-prepared pair
  indices. Python owns RNG / subsampling.
* ``kaplan_yorke_dimension(lyapunov_exponents)`` — Kaplan-Yorke /
  information dimension from a sorted Lyapunov spectrum.
"""

module DimensionJL

export correlation_integral, kaplan_yorke_dimension


"""
    correlation_integral(traj_flat, t, d, idx_i, idx_j, epsilons)

Count caller-prepared zero-based nonself pairs at sorted finite nonnegative radii.
Return fresh fractions using strict Euclidean distance comparison. Reject malformed
shapes, nonfinite coordinates, boolean metadata and invalid indices before indexing.
"""
function correlation_integral(
    traj_flat::AbstractVector{Float64},
    t::Integer,
    d::Integer,
    idx_i::AbstractVector{<:Integer},
    idx_j::AbstractVector{<:Integer},
    epsilons::AbstractVector{Float64},
)
    (t isa Bool || d isa Bool || t < 0 || d <= 0) &&
        throw(ArgumentError("invalid trajectory dimensions"))
    shape = Base.checked_mul(Int(t), Int(d))
    length(traj_flat) == shape || throw(ArgumentError("traj_flat shape mismatch"))
    length(idx_i) == length(idx_j) || throw(ArgumentError("pair length mismatch"))
    all(isfinite, traj_flat) || throw(ArgumentError("coordinates must be finite"))
    all(x -> isfinite(x) && x >= 0.0, epsilons) && issorted(epsilons) ||
        throw(ArgumentError("radii must be sorted finite and nonnegative"))
    for (i, j) in zip(idx_i, idx_j)
        (i isa Bool || j isa Bool || !(0 <= i < t) || !(0 <= j < t) || i == j) &&
            throw(ArgumentError("invalid zero-based nonself pair"))
    end
    np = length(idx_i)
    nk = length(epsilons)
    if np == 0
        return zeros(Float64, nk)
    end
    dists = Vector{Float64}(undef, np)
    @inbounds for p in 1:np
        i = idx_i[p] + 1  # Python 0-index → Julia 1-index
        j = idx_j[p] + 1
        s = 0.0
        base_i = (i - 1) * d
        base_j = (j - 1) * d
        for k in 1:d
            δ = traj_flat[base_i + k] - traj_flat[base_j + k]
            s = hypot(s, δ)
        end
        dists[p] = s
    end
    out = zeros(Float64, nk)
    @inbounds for k in 1:nk
        cnt = 0
        ε = epsilons[k]
        for p in 1:np
            if dists[p] < ε
                cnt += 1
            end
        end
        out[k] = Float64(cnt) / Float64(np)
    end
    return out
end


"""
    kaplan_yorke_dimension(lyapunov_exponents)

Evaluate the sorted spectrum using scaled compensated cumulative sums. Return a
finite dimension in `[0, length]`; reject nonfinite exponents. Inputs are preserved.
"""
function kaplan_yorke_dimension(lyapunov_exponents::AbstractVector{Float64})
    all(isfinite, lyapunov_exponents) || throw(ArgumentError("exponents must be finite"))
    scale = maximum(abs, lyapunov_exponents; init = 0.0)
    scale == 0.0 && return Float64(length(lyapunov_exponents))
    le = sort(collect(lyapunov_exponents); rev = true)
    total = 0.0
    correction = 0.0
    accepted = 0.0
    for (index, exponent) in enumerate(le)
        value = exponent / scale
        updated = total + value
        correction += abs(total) >= abs(value) ?
            (total - updated) + value : (value - updated) + total
        total = updated
        prefix = total + correction
        prefix < 0.0 && return Float64(index - 1) + accepted / abs(value)
        accepted = prefix
    end
    return Float64(length(le))
end

end  # module
