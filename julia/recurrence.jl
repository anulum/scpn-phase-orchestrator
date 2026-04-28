# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence matrix kernels (Julia port)

"""
recurrence.jl

``R_ij = Θ(ε − ‖x_i − x_j‖)`` for a single trajectory and the
cross-recurrence matrix ``CR_ij = Θ(ε − ‖x_i − y_j‖)`` for two
trajectories. ``metric_angular = true`` switches to the chord
distance ``√Σ 4·sin(Δ/2)²``.

Outputs are flat ``(T·T)`` UInt8 row-major; Python reshapes + casts
to bool.
"""

module RecurrenceJL

export recurrence_matrix, cross_recurrence_matrix


function _squared_distance(
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    ia::Int,
    ib::Int,
    d::Int,
    angular::Bool,
)
    s = 0.0
    if angular
        @inbounds for k in 1:d
            δ = a[(ia - 1) * d + k] - b[(ib - 1) * d + k]
            c = 2.0 * sin(δ / 2.0)
            s += c * c
        end
    else
        @inbounds for k in 1:d
            δ = a[(ia - 1) * d + k] - b[(ib - 1) * d + k]
            s += δ * δ
        end
    end
    return s
end


function recurrence_matrix(
    traj::AbstractVector{Float64},
    t::Integer,
    d::Integer,
    epsilon::Float64,
    angular::Bool,
)
    length(traj) == t * d || error("traj shape mismatch")
    eps_sq = epsilon * epsilon
    out = zeros(UInt8, t * t)
    @inbounds for i in 1:t
        for j in 1:t
            dist_sq = _squared_distance(traj, traj, i, j, d, angular)
            if dist_sq <= eps_sq
                out[(i - 1) * t + j] = UInt8(1)
            end
        end
    end
    return out
end


function cross_recurrence_matrix(
    traj_a::AbstractVector{Float64},
    traj_b::AbstractVector{Float64},
    t::Integer,
    d::Integer,
    epsilon::Float64,
    angular::Bool,
)
    length(traj_a) == t * d || error("traj_a shape mismatch")
    length(traj_b) == t * d || error("traj_b shape mismatch")
    eps_sq = epsilon * epsilon
    out = zeros(UInt8, t * t)
    @inbounds for i in 1:t
        for j in 1:t
            dist_sq = _squared_distance(traj_a, traj_b, i, j, d, angular)
            if dist_sq <= eps_sq
                out[(i - 1) * t + j] = UInt8(1)
            end
        end
    end
    return out
end

end  # module
