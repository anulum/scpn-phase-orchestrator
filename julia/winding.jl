# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding numbers (Julia port)

"""
winding.jl — cumulative winding number per oscillator.

    w_i = floor(Σ_t wrap(Δθ_{i,t}) / 2π)

where ``wrap(x) ∈ (-π, π]``. Matches the NumPy / Rust / Go / Mojo
references.
"""

module WindingJL

using LinearAlgebra

export winding_numbers

const TWO_PI = 2.0 * pi


function winding_numbers(
    phases_flat::AbstractVector{Float64},
    t::Integer,
    n::Integer,
)
    # ``phases_flat`` is row-major (T, N) as numpy would pass.
    length(phases_flat) == t * n ||
        error("phases_flat shape mismatch")
    if t < 2
        return zeros(Int64, n)
    end
    cumulative = zeros(Float64, n)
    @inbounds for step in 2:t
        base_now = (step - 1) * n
        base_prev = (step - 2) * n
        for i in 1:n
            δ = phases_flat[base_now + i] - phases_flat[base_prev + i]
            wrapped = mod(δ + pi, TWO_PI) - pi
            cumulative[i] += wrapped
        end
    end
    out = Vector{Int64}(undef, n)
    @inbounds for i in 1:n
        out[i] = floor(Int64, cumulative[i] / TWO_PI)
    end
    return out
end

end  # module
