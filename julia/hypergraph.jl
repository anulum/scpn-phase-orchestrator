# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hypergraph k-body Kuramoto (Julia port)

"""
hypergraph.jl — generalised k-body Kuramoto stepper with an optional
pairwise dense-``K`` component.

``hypergraph_run(phases, omegas, n, edge_nodes, edge_offsets,
edge_strengths, knm_flat, alpha_flat, zeta, psi, dt, n_steps)
-> Vector{Float64}``

* ``edge_nodes``: flat row-major indices for every hyperedge.
* ``edge_offsets[i]`` = start index of edge ``i`` in ``edge_nodes``;
  edge ``i`` spans ``edge_nodes[offsets[i]+1 : (i < n_edges-1 ?
  offsets[i+1] : end)]`` (1-based Julia arithmetic).
* ``knm_flat`` has length ``n*n`` when pairwise coupling is active,
  ``0`` otherwise.
* ``alpha_flat`` mirrors the Rust convention: length-``n*n`` or ``0``.

The pairwise derivative uses the Rust kernel's
``sin(θ_j - θ_i) = s_j·c_i - c_j·s_i`` expansion in the
``alpha == 0`` branch to preserve bit-for-bit parity.
"""

module HypergraphJL

export hypergraph_run

const TWO_PI = 2.0 * pi


function _compute_derivative!(
    theta::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    n::Integer,
    edge_nodes::AbstractVector{Int},
    edge_offsets::AbstractVector{Int},
    edge_strengths::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    zeta::Float64,
    psi::Float64,
    deriv::AbstractVector{Float64},
)
    sin_th = Vector{Float64}(undef, n)
    cos_th = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        sin_th[i] = sin(theta[i])
        cos_th[i] = cos(theta[i])
    end

    has_pairwise = length(knm_flat) == n * n
    alpha_zero = all(a -> a == 0.0, alpha_flat)
    (zs_psi, zc_psi) = if zeta != 0.0
        (zeta * sin(psi), zeta * cos(psi))
    else
        (0.0, 0.0)
    end

    @inbounds for i in 1:n
        pw = 0.0
        if has_pairwise
            offset = (i - 1) * n
            ci = cos_th[i]
            si = sin_th[i]
            if alpha_zero
                for j in 1:n
                    pw += knm_flat[offset + j] *
                        (sin_th[j] * ci - cos_th[j] * si)
                end
            else
                for j in 1:n
                    pw += knm_flat[offset + j] *
                        sin(theta[j] - theta[i] - alpha_flat[offset + j])
                end
            end
        end
        deriv[i] = omegas[i] + pw
        if zeta != 0.0
            deriv[i] += zs_psi * cos_th[i] - zc_psi * sin_th[i]
        end
    end

    # Hyperedges — sequential accumulation onto the derivative buffer.
    n_edges = length(edge_offsets)
    @inbounds for e in 1:n_edges
        start = edge_offsets[e] + 1  # 1-based
        stop = e < n_edges ? edge_offsets[e + 1] : length(edge_nodes)
        k = stop - start + 1
        phase_sum = 0.0
        for p in start:stop
            phase_sum += theta[edge_nodes[p] + 1]  # 1-based
        end
        sigma = edge_strengths[e]
        for p in start:stop
            m = edge_nodes[p] + 1
            deriv[m] += sigma * sin(phase_sum - Float64(k) * theta[m])
        end
    end
    return nothing
end


function hypergraph_run(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    n::Integer,
    edge_nodes::AbstractVector{<:Integer},
    edge_offsets::AbstractVector{<:Integer},
    edge_strengths::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    n_steps::Integer,
)
    length(phases) == n || error("phases shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    p = Vector{Float64}(phases)
    deriv = Vector{Float64}(undef, n)
    en = Vector{Int}(edge_nodes)
    eo = Vector{Int}(edge_offsets)
    for _ in 1:n_steps
        _compute_derivative!(p, omegas, n, en, eo, edge_strengths,
                             knm_flat, alpha_flat, zeta, psi, deriv)
        @inbounds for i in 1:n
            p[i] = mod(p[i] + dt * deriv[i], TWO_PI)
        end
    end
    return p
end

end  # module
