# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Normalised Persistent Entropy (Julia port)

"""
npe.jl

Normalised Persistent Entropy (NPE) from the H₀ persistence diagram
of the pairwise circular-distance matrix. Uses single-linkage union-
find as a lightweight substitute for full Vietoris–Rips persistence
(ripser). Matches the NumPy and Rust references bit-for-bit.
"""

module NPE

export phase_distance_matrix, compute_npe

"""
    phase_distance_matrix(phases) -> Vector{Float64}

Flat row-major ``N × N`` matrix of pairwise circular distances in
``[0, π]``.
"""
function phase_distance_matrix(phases::AbstractVector{Float64})
    n = length(phases)
    out = zeros(Float64, n * n)
    @inbounds for i in 0:(n - 1)
        for j in 0:(n - 1)
            d = phases[i + 1] - phases[j + 1]
            out[i * n + j + 1] = abs(atan(sin(d), cos(d)))
        end
    end
    return out
end


# Iterative union-find with path compression and union-by-rank.
function _find(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end


"""
    compute_npe(phases, max_radius) -> Float64

Compute the NPE. ``max_radius < 0`` means default π.
"""
function compute_npe(
    phases::AbstractVector{Float64}, max_radius::Float64
)
    n = length(phases)
    n >= 2 || return 0.0
    radius = max_radius < 0.0 ? Float64(pi) : max_radius

    dist = phase_distance_matrix(phases)

    # Upper-triangle edges (i < j) sorted ascending.
    n_edges = n * (n - 1) ÷ 2
    edges = Vector{Tuple{Float64, Int, Int}}(undef, n_edges)
    k = 1
    @inbounds for i in 0:(n - 2)
        for j in (i + 1):(n - 1)
            edges[k] = (dist[i * n + j + 1], i, j)
            k += 1
        end
    end
    sort!(edges, by = e -> e[1])

    parent = collect(1:n)  # 1-based union-find
    rank = zeros(Int, n)
    lifetimes = Float64[]

    @inbounds for (d, i, j) in edges
        d > radius && break
        ri = _find(parent, i + 1)
        rj = _find(parent, j + 1)
        if ri != rj
            push!(lifetimes, d)
            if rank[ri] < rank[rj]
                parent[ri] = rj
            elseif rank[ri] > rank[rj]
                parent[rj] = ri
            else
                parent[rj] = ri
                rank[ri] += 1
            end
        end
    end

    isempty(lifetimes) && return 0.0
    total = sum(lifetimes)
    total < 1e-15 && return 0.0

    probs = filter(p -> p > 0.0, lifetimes ./ total)
    entropy = -sum(p * log(p) for p in probs)
    max_entropy = length(probs) > 1 ? log(Float64(length(probs))) : 1.0
    max_entropy < 1e-15 && return 0.0
    return entropy / max_entropy
end

end  # module
