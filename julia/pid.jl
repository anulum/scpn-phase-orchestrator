# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Partial Information Decomposition (Julia port)

"""
pid.jl — time-series Williams & Beer partial information decomposition of two
oscillator groups about the global synchronisation state.

``pid_decomposition(history_flat, t, n, group_a, group_b, n_bins)
-> (redundancy, synergy)``

``history_flat`` is the row-major ``(t, n)`` phase history; ``group_a`` /
``group_b`` hold 0-based oscillator indices. Each timestep is reduced to the
global and per-group order-parameter phases, binned into ``n_bins`` phase bins,
and decomposed via ``I_min`` redundancy and the synergy identity. Matches the
Rust/Python reference.
"""

module PidJL

export pid_decomposition

const TAU = 2.0 * pi


@inline function bin_angle(angle::Float64, n_bins::Int)
    w = mod(angle, TAU)
    b = Int(floor(w / (TAU / n_bins)))
    return min(b, n_bins - 1)  # 0-based bin
end


function group_phase(history, n::Int, row::Int, members)
    sin_sum = 0.0
    cos_sum = 0.0
    @inbounds for j in members
        θ = history[row * n + j + 1]
        sin_sum += sin(θ)
        cos_sum += cos(θ)
    end
    count = Float64(length(members))
    return atan(sin_sum / count, cos_sum / count)
end


function global_phase(history, n::Int, row::Int)
    sin_sum = 0.0
    cos_sum = 0.0
    @inbounds for j in 0:(n - 1)
        θ = history[row * n + j + 1]
        sin_sum += sin(θ)
        cos_sum += cos(θ)
    end
    count = Float64(n)
    return atan(sin_sum / count, cos_sum / count)
end


function mutual_information(joint, marg_x, marg_y, n_x::Int, n_bins::Int, total::Float64)
    total <= 0.0 && return 0.0
    mi = 0.0
    @inbounds for x in 1:n_x
        marg_x[x] <= 0.0 && continue
        for y in 1:n_bins
            cxy = joint[x, y]
            (cxy <= 0.0 || marg_y[y] <= 0.0) && continue
            p_xy = cxy / total
            mi += p_xy * log(p_xy / ((marg_x[x] / total) * (marg_y[y] / total)))
        end
    end
    return max(0.0, mi)
end


function i_min_redundancy(cay, cby, ca, cb, cy, n_bins::Int, total::Float64)
    total <= 0.0 && return 0.0
    i_red = 0.0
    @inbounds for y in 1:n_bins
        cy[y] <= 0.0 && continue
        p_y = cy[y] / total
        ispec_a = 0.0
        for x in 1:n_bins
            (cay[x, y] <= 0.0 || ca[x] <= 0.0) && continue
            p_a_given_y = cay[x, y] / cy[y]
            p_y_given_a = cay[x, y] / ca[x]
            ispec_a += p_a_given_y * log(p_y_given_a / p_y)
        end
        ispec_b = 0.0
        for x in 1:n_bins
            (cby[x, y] <= 0.0 || cb[x] <= 0.0) && continue
            p_b_given_y = cby[x, y] / cy[y]
            p_y_given_b = cby[x, y] / cb[x]
            ispec_b += p_b_given_y * log(p_y_given_b / p_y)
        end
        i_red += p_y * min(ispec_a, ispec_b)
    end
    return max(0.0, i_red)
end


function pid_decomposition(
    history_flat::AbstractVector{Float64},
    t::Integer,
    n::Integer,
    group_a::AbstractVector{<:Integer},
    group_b::AbstractVector{<:Integer},
    n_bins::Integer,
)
    big_t = Int(t)
    big_n = Int(n)
    nb = Int(n_bins)
    if big_t == 0 || big_n == 0 || isempty(group_a) || isempty(group_b) || nb == 0
        return 0.0, 0.0
    end
    ga = [Int(g) for g in group_a]
    gb = [Int(g) for g in group_b]

    cy = zeros(Float64, nb)
    ca = zeros(Float64, nb)
    cb = zeros(Float64, nb)
    cay = zeros(Float64, nb, nb)
    cby = zeros(Float64, nb, nb)
    cab = zeros(Float64, nb * nb)
    caby = zeros(Float64, nb * nb, nb)

    @inbounds for row in 0:(big_t - 1)
        y = bin_angle(global_phase(history_flat, big_n, row), nb)
        a = bin_angle(group_phase(history_flat, big_n, row, ga), nb)
        b = bin_angle(group_phase(history_flat, big_n, row, gb), nb)
        cy[y + 1] += 1.0
        ca[a + 1] += 1.0
        cb[b + 1] += 1.0
        cay[a + 1, y + 1] += 1.0
        cby[b + 1, y + 1] += 1.0
        ab = a * nb + b
        cab[ab + 1] += 1.0
        caby[ab + 1, y + 1] += 1.0
    end

    total = Float64(big_t)
    mi_a = mutual_information(cay, ca, cy, nb, nb, total)
    mi_b = mutual_information(cby, cb, cy, nb, nb, total)
    mi_ab = mutual_information(caby, cab, cy, nb * nb, nb, total)
    i_red = i_min_redundancy(cay, cby, ca, cb, cy, nb, total)
    synergy = max(0.0, mi_ab - mi_a - mi_b + i_red)
    return i_red, synergy
end

end  # module
