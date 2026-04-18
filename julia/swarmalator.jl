# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator stepper (Julia port)

"""
swarmalator.jl — single O(N² · d) swarmalator step.

``swarmalator_step(pos_flat, phases, omegas, n, dim, a, b, j, k, dt)
-> (new_pos_flat, new_phases)``

* ``pos_flat`` / ``new_pos_flat`` are row-major ``(N, d)``.
* Matches the Rust ``PySwarmalatorStepper.step`` semantics exactly.

Regularisation constants (``1e-6`` inside sqrt and distance cube)
guard against singularities at coincident agents.
"""

module SwarmalatorJL

export swarmalator_step

const TWO_PI = 2.0 * pi


function swarmalator_step(
    pos_flat::AbstractVector{Float64},
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    n::Integer,
    dim::Integer,
    a::Float64,
    b::Float64,
    j::Float64,
    k::Float64,
    dt::Float64,
)
    length(pos_flat) == n * dim || error("pos shape mismatch")
    length(phases) == n || error("phases shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    new_pos = copy(pos_flat)
    new_phases = zeros(Float64, n)
    @inbounds for i in 1:n
        # Velocity + phase-derivative accumulators.
        vel = zeros(Float64, dim)
        phase_acc = 0.0
        base_i = (i - 1) * dim
        θi = phases[i]
        for m in 1:n
            base_m = (m - 1) * dim
            # Squared distance with 1e-6 regularisation.
            s = 0.0
            for d in 1:dim
                δ = pos_flat[base_m + d] - pos_flat[base_i + d]
                s += δ * δ
            end
            dist = sqrt(s + 1e-6)
            cos_d = cos(phases[m] - θi)
            sin_d = sin(phases[m] - θi)
            attract = (a + j * cos_d) / dist
            # Rust canonical: b / (dist * d²ₛᵤₘ + eps).
            repulse = b / (dist * s + 1e-6)
            factor = attract - repulse
            for d in 1:dim
                δ = pos_flat[base_m + d] - pos_flat[base_i + d]
                vel[d] += δ * factor
            end
            phase_acc += sin_d / dist
        end
        inv_n = 1.0 / Float64(n)
        for d in 1:dim
            new_pos[base_i + d] = pos_flat[base_i + d] + dt * vel[d] * inv_n
        end
        dth = omegas[i] + k * phase_acc * inv_n
        new_phases[i] = mod(θi + dt * dth, TWO_PI)
    end
    return new_pos, new_phases
end

end  # module
