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
  within floating-point tolerance (full-snapshot Euler, target/source
  coupling, post-step measurement). No convergence certificate is made.
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
            isfinite(k_ij) || throw(ArgumentError("scaled coupling overflow"))
            if k_ij == 0.0
                continue
            end
            a_ij = alpha_flat[base + j]
            angle = old[j] - θi - a_ij
            isfinite(angle) || throw(ArgumentError("phase difference overflow"))
            coupling += k_ij * sin(angle)
        end
        velocity = omegas[i] + coupling
        phases[i] = θi + dt * velocity
        (isfinite(velocity) && isfinite(phases[i])) || throw(ArgumentError("Euler step overflow"))
    end
    return nothing
end

function _order_parameter(phases::AbstractVector{Float64})
    # Exported input validation requires a nonempty population.
    n = Float64(length(phases))
    sum_cos = 0.0
    sum_sin = 0.0
    @inbounds for θ in phases
        sum_cos += cos(θ)
        sum_sin += sin(θ)
    end
    return sqrt((sum_cos / n)^2 + (sum_sin / n)^2)
end

"""
    steady_state_r(phases, omegas, knm_flat, alpha_flat, n, scale, dt, transient, measure)

Return the mean post-step R after a discarded transient, using explicit Euler.
Validate shapes and finite values before indexing. An empty measurement window
returns zero without integration. Invalid inputs/arithmetic throw ArgumentError.
"""
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
    (n isa Bool || n <= 0 || n > typemax(Int) || n > div(typemax(Int), n)) &&
        throw(ArgumentError("n must be positive and n squared fit Int"))
    (n_transient isa Bool || n_measure isa Bool || n_transient < 0 || n_measure < 0 ||
     n_transient > typemax(Int) || n_measure > typemax(Int)) &&
        throw(ArgumentError("step counts must be nonnegative integers fitting Int"))
    (isfinite(k_scale) && isfinite(dt) && dt > 0) ||
        throw(ArgumentError("scale must be finite and dt finite and positive"))
    all(values -> all(isfinite, values), (phases_init, omegas, knm_flat, alpha_flat)) ||
        throw(ArgumentError("trial arrays must contain only finite values"))
    length(phases_init) == n || throw(ArgumentError("phases_init shape mismatch"))
    length(omegas) == n || throw(ArgumentError("omegas shape mismatch"))
    length(knm_flat) == n * n || throw(ArgumentError("knm_flat shape mismatch"))
    length(alpha_flat) == n * n || throw(ArgumentError("alpha_flat shape mismatch"))
    n_measure == 0 && return 0.0
    phases = copy(phases_init)
    for _ in 1:n_transient
        _kuramoto_step!(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
    end
    r_sum = 0.0
    for _ in 1:n_measure
        _kuramoto_step!(phases, omegas, knm_flat, alpha_flat, n, k_scale, dt)
        r_sum += _order_parameter(phases)
    end
    r = r_sum / Float64(n_measure)
    (isfinite(r) && 0 <= r <= 1 + 1e-12) || throw(ArgumentError("invalid mean R"))
    return min(r, 1.0)
end

end  # module
