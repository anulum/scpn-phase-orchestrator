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


function correlation_integral(
    traj_flat::AbstractVector{Float64},
    t::Integer,
    d::Integer,
    idx_i::AbstractVector{<:Integer},
    idx_j::AbstractVector{<:Integer},
    epsilons::AbstractVector{Float64},
)
    length(traj_flat) == t * d || error("traj_flat shape mismatch")
    length(idx_i) == length(idx_j) || error("idx_i / idx_j length mismatch")
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
            s += δ * δ
        end
        dists[p] = sqrt(s)
    end
    out = zeros(Float64, nk)
    inv_np = 1.0 / Float64(np)
    @inbounds for k in 1:nk
        cnt = 0
        ε = epsilons[k]
        for p in 1:np
            if dists[p] < ε
                cnt += 1
            end
        end
        out[k] = Float64(cnt) * inv_np
    end
    return out
end


function kaplan_yorke_dimension(lyapunov_exponents::AbstractVector{Float64})
    n = length(lyapunov_exponents)
    if n == 0
        return 0.0
    end
    le = sort(collect(lyapunov_exponents); rev = true)
    cumsum_val = 0.0
    j = -1
    @inbounds for i in 1:n
        cumsum_val += le[i]
        if cumsum_val >= 0.0
            j = i
        else
            break
        end
    end
    if j == -1
        return 0.0                     # λ_1 < 0 → D_KY = 0
    end
    if j >= n
        return Float64(n)              # all λ ≥ 0 → volume expanding
    end
    denom = abs(le[j + 1])
    if denom == 0.0
        return Float64(j)
    end
    # cumsum_val walked one step past rejection, so recompute Σ_{i=1}^{j}.
    s_j = 0.0
    for i in 1:j
        s_j += le[i]
    end
    return Float64(j) + s_j / denom
end

end  # module
