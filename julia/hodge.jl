# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Combinatorial Hodge decomposition (Julia port)

"""
hodge.jl — combinatorial (Helmholtz–Hodge) decomposition of the
Kuramoto coupling current into gradient, curl, and harmonic edge
flows.

``hodge_decomposition(knm_flat, phases, n, edges_flat, n_edges,
tris_flat, n_tris)`` returns three flattened row-major ``N×N``
antisymmetric flow matrices ``(grad, curl, harmonic)``.

The alternating edge flow is ``f_ij = ½(K_ij + K_ji)·sin(θ_j − θ_i)``.
With node–edge incidence ``B1`` and edge–triangle incidence ``B2``,
``f_grad = B1ᵀ L0⁺ (B1 f)``, ``f_curl = B2 L2⁺ (B2ᵀ f)``, and
``f_harm = f − f_grad − f_curl``. The pseudoinverse uses a relative
cutoff ``rtol = 1e-9`` shared with the Python reference. Node indices
in ``edges_flat`` / ``tris_flat`` are 0-based.
"""

module HodgeJL

using LinearAlgebra

export hodge_decomposition

const PINV_RTOL = 1e-9


function hodge_decomposition(
    knm_flat::AbstractVector{Float64},
    phases::AbstractVector{Float64},
    n::Integer,
    edges_flat::AbstractVector{<:Integer},
    n_edges::Integer,
    tris_flat::AbstractVector{<:Integer},
    n_tris::Integer,
)
    big_n = Int(n)
    ne = Int(n_edges)
    nt = Int(n_tris)
    length(knm_flat) == big_n * big_n || error("knm shape mismatch")
    length(phases) == big_n || error("phases shape mismatch")

    edge_i = Vector{Int}(undef, ne)
    edge_j = Vector{Int}(undef, ne)
    flow = zeros(Float64, ne)
    @inbounds for e in 1:ne
        i = Int(edges_flat[2 * (e - 1) + 1])
        j = Int(edges_flat[2 * (e - 1) + 2])
        edge_i[e] = i
        edge_j[e] = j
        kij = knm_flat[i * big_n + j + 1]
        kji = knm_flat[j * big_n + i + 1]
        ksym = 0.5 * (kij + kji)
        flow[e] = ksym * sin(phases[j + 1] - phases[i + 1])
    end

    # L0 = B1 B1ᵀ and div = B1 f.
    l0 = zeros(Float64, big_n, big_n)
    divf = zeros(Float64, big_n)
    @inbounds for e in 1:ne
        i = edge_i[e] + 1
        j = edge_j[e] + 1
        l0[i, i] += 1.0
        l0[j, j] += 1.0
        l0[i, j] -= 1.0
        l0[j, i] -= 1.0
        divf[i] -= flow[e]
        divf[j] += flow[e]
    end
    potential = ne == 0 ? zeros(Float64, big_n) : pinv(l0; rtol = PINV_RTOL) * divf
    f_grad = zeros(Float64, ne)
    @inbounds for e in 1:ne
        f_grad[e] = potential[edge_j[e] + 1] - potential[edge_i[e] + 1]
    end

    f_curl = zeros(Float64, ne)
    if nt > 0
        emap = Dict{Tuple{Int,Int},Int}()
        @inbounds for e in 1:ne
            emap[(edge_i[e], edge_j[e])] = e
        end
        tri_edges = Vector{NTuple{3,Tuple{Int,Float64}}}(undef, nt)
        @inbounds for t in 1:nt
            i = Int(tris_flat[3 * (t - 1) + 1])
            j = Int(tris_flat[3 * (t - 1) + 2])
            k = Int(tris_flat[3 * (t - 1) + 3])
            tri_edges[t] = (
                (emap[(i, j)], 1.0),
                (emap[(j, k)], 1.0),
                (emap[(i, k)], -1.0),
            )
        end
        l2 = zeros(Float64, nt, nt)
        c2 = zeros(Float64, nt)
        @inbounds for t in 1:nt
            for (et, st) in tri_edges[t]
                c2[t] += st * flow[et]
            end
            for u in 1:nt
                acc = 0.0
                for (et, st) in tri_edges[t], (eu, su) in tri_edges[u]
                    if et == eu
                        acc += st * su
                    end
                end
                l2[t, u] = acc
            end
        end
        tri_pot = pinv(l2; rtol = PINV_RTOL) * c2
        @inbounds for t in 1:nt
            for (e, sg) in tri_edges[t]
                f_curl[e] += sg * tri_pot[t]
            end
        end
    end

    grad_m = zeros(Float64, big_n * big_n)
    curl_m = zeros(Float64, big_n * big_n)
    harm_m = zeros(Float64, big_n * big_n)
    @inbounds for e in 1:ne
        i = edge_i[e]
        j = edge_j[e]
        g = f_grad[e]
        c = f_curl[e]
        h = flow[e] - g - c
        grad_m[i * big_n + j + 1] = g
        grad_m[j * big_n + i + 1] = -g
        curl_m[i * big_n + j + 1] = c
        curl_m[j * big_n + i + 1] = -c
        harm_m[i * big_n + j + 1] = h
        harm_m[j * big_n + i + 1] = -h
    end
    return grad_m, curl_m, harm_m
end

end  # module
