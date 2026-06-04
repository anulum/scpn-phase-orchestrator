# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spatial coupling modulation (Julia port)

module SpatialModulatorJL

export spatial_modulate

function _weight(distance::Float64, k_base::Float64, form::Integer, exponent::Float64, length::Float64, epsilon::Float64)::Float64
    if form == 0
        return k_base / (1.0 + distance)
    elseif form == 1
        return k_base * exp(-distance / length)
    elseif form == 2
        return k_base * (1.0 + distance / length)^(-exponent)
    elseif form == 3
        return k_base / sqrt(distance * distance + epsilon)
    else
        error("unknown decay form")
    end
end

function spatial_modulate(
    knm_flat::AbstractVector{Float64},
    positions_flat::AbstractVector{Float64},
    n::Integer,
    dim::Integer,
    k_base::Float64,
    form::Integer,
    exponent::Float64,
    scale::Float64,
    epsilon::Float64,
)
    length(knm_flat) == n * n || error("knm shape mismatch")
    length(positions_flat) == n * dim || error("positions shape mismatch")
    out = zeros(Float64, n * n)
    @inbounds for i in 1:n
        for j in 1:n
            idx = (i - 1) * n + j
            if i == j
                out[idx] = 0.0
                continue
            end
            d2 = 0.0
            for d in 1:dim
                delta = positions_flat[(i - 1) * dim + d] - positions_flat[(j - 1) * dim + d]
                d2 += delta * delta
            end
            distance = sqrt(d2)
            out[idx] = knm_flat[idx] * _weight(distance, k_base, form, exponent, scale, epsilon)
        end
    end
    return out
end

end  # module
