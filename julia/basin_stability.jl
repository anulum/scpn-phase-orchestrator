# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Steady-state R (Julia port)

"""
basin_stability.jl — one-trial Kuramoto steady-state R.

``steady_state_r(phases_init, omegas, knm_flat, alpha_flat, n,
k_scale, dt, n_transient, n_measure) -> Float64``

Integrates the Kuramoto ODE with explicit Euler, discards the
first ``n_transient`` steps, then averages the order parameter
``R = |<e^{iθ}>|`` over the following ``n_measure`` steps.

* ``knm_flat`` / ``alpha_flat`` are row-major ``(N, N)``.
* Matches the Rust ``bifurcation::steady_state_r`` semantics
  exactly (full-snapshot Euler step, same coupling accumulator
  order, same order-parameter formula).
"""

module BasinStabilityJL

export steady_state_r

function _kuramoto_step!(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    k_scale::Float64,
    dt::Float64,
)
    old = copy(phases)
    @inbounds for i in 1:n
        coupling = 0.0
        base = (i - 1) * n
        θi = old[i]
        for j in 1:n
            k_ij = knm_flat[base + j] * k_scale
            if abs(k_ij) < 1e-30
                continue
            end
            a_ij = alpha_flat[base + j]
            coupling += k_ij * sin(old[j] - θi - a_ij)
        end
        phases[i] = θi + dt * (omegas[i] + coupling)
    end
    return nothing
end

function _order_parameter(phases::AbstractVector{Float64})
    isempty(phases) && return 0.0
    n = Float64(length(phases))
    sum_cos = 0.0
    sum_sin = 0.0
    @inbounds for θ in phases
        sum_cos += cos(θ)
        sum_sin += sin(θ)
    end
    return sqrt((sum_cos / n)^2 + (sum_sin / n)^2)
end

function steady_state_r(
    phases_init::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    k_scale::Float64,
    dt::Float64,
    n_transient::Integer,
    n_measure::Integer,
)
    length(phases_init) == n || error("phases_init shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm_flat shape mismatch")
    length(alpha_flat) == n * n || error("alpha_flat shape mismatch")
    phases = copy(phases_init)
    for _ in 1:n_transient
        _kuramoto_step!(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
    end
    r_sum = 0.0
    for _ in 1:n_measure
        _kuramoto_step!(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
        r_sum += _order_parameter(phases)
    end
    return n_measure == 0 ? 0.0 : r_sum / Float64(n_measure)
end

end  # module
