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

"""
    local_order_parameter(phases, knm_flat, n)

Return unweighted local coherence on positive directed row-major edges,
excluding self even within the admitted diagonal tolerance of 1e-15.
Finite Float64 buffers must have exact n and n*n lengths; n is a plain
nonnegative integer. Empty input returns an empty vector. ArgumentError
reports invalid domains before allocation/indexing. Factoring the centre
phasor avoids overflow in finite unwrapped phase differences.
"""

function local_order_parameter(
    phases::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    n::Integer,
)
    n isa Bool && throw(ArgumentError("n must be a plain nonnegative integer"))
    0 <= n <= typemax(Int) || throw(ArgumentError("n must fit a nonnegative Int"))
    nn = Int(n)
    Base.require_one_based_indexing(phases, knm_flat)
    (nn == 0 || nn <= typemax(Int) ÷ nn) || throw(ArgumentError("n*n overflows Int"))
    length(phases) == nn || throw(ArgumentError("phases shape mismatch"))
    length(knm_flat) == nn * nn || throw(ArgumentError("knm shape mismatch"))
    all(isfinite, phases) && all(isfinite, knm_flat) ||
        throw(ArgumentError("phases and knm must be finite"))
    all(i -> abs(knm_flat[(i - 1) * nn + i]) <= 1e-15, 1:nn) ||
        throw(ArgumentError("knm self-coupling diagonal must be zero"))
    out = zeros(Float64, nn)
    @inbounds for i in 1:nn
        sr = 0.0
        si = 0.0
        cnt = 0
        base = (i - 1) * nn
        for j in 1:nn
            if j != i && knm_flat[base + j] > 0.0
                sr += cos(phases[j])
                si += sin(phases[j])
                cnt += 1
            end
        end
        if cnt == 0
            out[i] = 0.0
        else
            inv = 1.0 / Float64(cnt)
            sr *= inv
            si *= inv
            out[i] = min(hypot(sr, si), 1.0)
        end
    end
    return out
end

end  # module
