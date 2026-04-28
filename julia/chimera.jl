# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera local order-parameter kernel (Julia port)

"""
chimera.jl — local order parameter

    R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|

where ``N(i) = { j : K_{ij} > 0 }``. Returns the ``R_local`` vector;
the coherent / incoherent partition + chimera index stay Python-side.
"""

module ChimeraJL

export local_order_parameter


function local_order_parameter(
    phases::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    n::Integer,
)
    length(phases) == n || error("phases shape mismatch")
    length(knm_flat) == n * n || error("knm shape mismatch")
    out = zeros(Float64, n)
    @inbounds for i in 1:n
        sr = 0.0
        si = 0.0
        cnt = 0
        theta_i = phases[i]
        base = (i - 1) * n
        for j in 1:n
            if knm_flat[base + j] > 0.0
                δ = phases[j] - theta_i
                sr += cos(δ)
                si += sin(δ)
                cnt += 1
            end
        end
        if cnt == 0
            out[i] = 0.0
        else
            inv = 1.0 / Float64(cnt)
            sr *= inv
            si *= inv
            out[i] = sqrt(sr * sr + si * si)
        end
    end
    return out
end

end  # module
