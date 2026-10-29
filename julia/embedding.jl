# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Embedding primitives (Julia port)

"""
embedding.jl — three primitives for delay-embedding analysis.

* ``delay_embed(signal, delay, dim)`` — ``v(t) = [x(t), x(t+τ), …]``.
* ``mutual_information(signal, lag, n_bins)`` — Fraser-Swinney 1986
  average mutual information.
* ``nearest_neighbor_distances(embedded, t, m)`` — brute-force
  kNN (k = 1) in the embedded space.

Wrappers (optimal_delay, optimal_dimension, auto_embed) stay in
Python for uniform control-flow behaviour across backends.
"""

module EmbeddingJL

using LinearAlgebra

export delay_embed, mutual_information, nearest_neighbor_distances


function delay_embed(
    signal::AbstractVector{Float64},
    delay::Integer,
    dimension::Integer,
)
    t_total = length(signal)
    t_eff = t_total - (dimension - 1) * delay
    t_eff > 0 || error("signal too short for delay=$delay, dim=$dimension")
    out = zeros(Float64, t_eff * dimension)
    @inbounds for i in 1:t_eff
        for d in 1:dimension
            out[(i - 1) * dimension + d] = signal[i + (d - 1) * delay]
        end
    end
    return out
end


"""
    histogram_edges(lo, hi, n_bins)

NumPy histogram edges over ``[lo, hi]``: edge ``k`` is ``k*step + lo`` for
``k < n_bins`` (product rounded before the addition) and the last edge is ``hi``.
"""
function histogram_edges(lo::Float64, hi::Float64, n_bins::Integer)
    step = (hi - lo) / Float64(n_bins)
    edges = Vector{Float64}(undef, n_bins + 1)
    for k in 0:(n_bins - 1)
        product = Float64(k) * step
        edges[k + 1] = product + lo
    end
    edges[n_bins + 1] = hi
    return edges
end

"""
    histogram_bin(edges, value)

Zero-based bin of ``value``: the last edge not greater than it, with a value
equal to the last edge counted in the last bin (NumPy ``histogramdd``).
"""
function histogram_bin(edges::Vector{Float64}, value::Float64)
    bin = searchsortedlast(edges, value) - 1
    return clamp(bin, 0, length(edges) - 2)
end

function mutual_information(
    signal::AbstractVector{Float64},
    lag::Integer,
    n_bins::Integer,
)
    t_total = length(signal) - lag
    if t_total <= 0
        return 0.0
    end

    # Bin range via min/max — matches numpy histogram2d default edges.
    x_min = minimum(signal[1:t_total])
    x_max = maximum(signal[1:t_total])
    y_min = minimum(signal[(lag + 1):(lag + t_total)])
    y_max = maximum(signal[(lag + 1):(lag + t_total)])

    if x_max <= x_min || y_max <= y_min
        return 0.0
    end

    x_edges = histogram_edges(x_min, x_max, n_bins)
    y_edges = histogram_edges(y_min, y_max, n_bins)

    hist = zeros(Float64, n_bins * n_bins)
    @inbounds for i in 1:t_total
        bx = histogram_bin(x_edges, signal[i])
        by = histogram_bin(y_edges, signal[i + lag])
        hist[bx * n_bins + by + 1] += 1.0
    end

    total = Float64(t_total)
    hx = zeros(Float64, n_bins)
    hy = zeros(Float64, n_bins)
    @inbounds for i in 0:(n_bins - 1)
        for j in 0:(n_bins - 1)
            h = hist[i * n_bins + j + 1]
            hx[i + 1] += h
            hy[j + 1] += h
        end
    end

    mi = 0.0
    @inbounds for i in 0:(n_bins - 1)
        for j in 0:(n_bins - 1)
            h = hist[i * n_bins + j + 1]
            if h > 0 && hx[i + 1] > 0 && hy[j + 1] > 0
                p_xy = h / total
                p_x = hx[i + 1] / total
                p_y = hy[j + 1] / total
                mi += p_xy * log(p_xy / (p_x * p_y))
            end
        end
    end
    return mi
end


function nearest_neighbor_distances(
    embedded_flat::AbstractVector{Float64},
    t::Integer,
    m::Integer,
)
    length(embedded_flat) == t * m || error("embedded shape mismatch")
    nn_dist = fill(Inf, t)
    nn_idx = zeros(Int64, t)
    @inbounds for i in 1:t
        best = Inf
        best_j = 0
        base_i = (i - 1) * m
        for j in 1:t
            if j == i
                continue
            end
            base_j = (j - 1) * m
            d = 0.0
            for k in 1:m
                δ = embedded_flat[base_i + k] - embedded_flat[base_j + k]
                d += δ * δ
            end
            if d < best
                best = d
                best_j = j - 1  # 0-indexed output for Python
            end
        end
        nn_dist[i] = sqrt(best)
        nn_idx[i] = best_j
    end
    return nn_dist, nn_idx
end

end  # module
