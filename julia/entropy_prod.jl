# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Thermodynamic entropy production rate (Julia port)

"""
entropy_prod.jl

Overdamped-Kuramoto dissipation rate Σ (dθ/dt)² · dt with
    dθ_i/dt = ω_i + (α / N) Σ_j K_ij sin(θ_j − θ_i).

Matches the Rust and NumPy references.
"""

module EntropyProd

export entropy_production_rate


function entropy_production_rate(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha::Float64,
    dt::Float64,
)
    n = length(phases)
    if n == 0 || dt <= 0.0
        return 0.0
    end
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm shape mismatch")
    inv_n = alpha / Float64(n)
    acc = 0.0
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            s += knm_flat[(i - 1) * n + j] * sin(phases[j] - phases[i])
        end
        d = omegas[i] + inv_n * s
        acc += d * d
    end
    return acc * dt
end

end  # module
