# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition (Julia port)

"""
hodge.jl — Hodge decomposition of a coupling × phase-difference
matrix into symmetric (gradient), antisymmetric (curl), and
harmonic (residual) components per oscillator.
"""

module HodgeJL

export hodge_decomposition


function hodge_decomposition(
    knm_flat::AbstractVector{Float64},
    phases::AbstractVector{Float64},
    n::Integer,
)
    length(knm_flat) == n * n || error("knm shape mismatch")
    length(phases) == n || error("phases shape mismatch")
    gradient = zeros(Float64, n)
    curl = zeros(Float64, n)
    harmonic = zeros(Float64, n)
    @inbounds for i in 1:n
        g = 0.0
        c = 0.0
        t = 0.0
        θi = phases[i]
        base_i = (i - 1) * n
        for j in 1:n
            kij = knm_flat[base_i + j]
            kji = knm_flat[(j - 1) * n + i]
            cd = cos(phases[j] - θi)
            sym = 0.5 * (kij + kji)
            anti = 0.5 * (kij - kji)
            g += sym * cd
            c += anti * cd
            t += kij * cd
        end
        gradient[i] = g
        curl[i] = c
        harmonic[i] = t - g - c
    end
    return gradient, curl, harmonic
end

end  # module
