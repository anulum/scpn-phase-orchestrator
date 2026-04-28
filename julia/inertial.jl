# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto (Julia port)

"""
inertial.jl — second-order (swing-equation) Kuramoto RK4 stepper.

``inertial_step(theta, omega_dot, power, knm_flat, inertia, damping,
n, dt) -> (new_theta, new_omega_dot)``

The derivative term uses the ``sin(θ_j - θ_i) = s_j·c_i - c_j·s_i``
expansion so that floating-point rounding matches the Rust kernel
(``crate::inertial::compute_derivative``) bit-for-bit.
"""

module InertialJL

export inertial_step

const TWO_PI = 2.0 * pi


function _compute_derivative!(
    theta::AbstractVector{Float64},
    omega_dot::AbstractVector{Float64},
    power::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    inertia::AbstractVector{Float64},
    damping::AbstractVector{Float64},
    n::Integer,
    out_t::AbstractVector{Float64},
    out_o::AbstractVector{Float64},
)
    sin_theta = Vector{Float64}(undef, n)
    cos_theta = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        sin_theta[i] = sin(theta[i])
        cos_theta[i] = cos(theta[i])
    end
    @inbounds for i in 1:n
        out_t[i] = omega_dot[i]
        ci = cos_theta[i]
        si = sin_theta[i]
        offset = (i - 1) * n
        coupling = 0.0
        for j in 1:n
            coupling += knm_flat[offset + j] *
                (sin_theta[j] * ci - cos_theta[j] * si)
        end
        out_o[i] = (power[i] + coupling - damping[i] * omega_dot[i]) /
            inertia[i]
    end
    return nothing
end


function inertial_step(
    theta::AbstractVector{Float64},
    omega_dot::AbstractVector{Float64},
    power::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    inertia::AbstractVector{Float64},
    damping::AbstractVector{Float64},
    n::Integer,
    dt::Float64,
)
    length(theta) == n || error("theta shape mismatch")
    length(omega_dot) == n || error("omega_dot shape mismatch")
    length(knm_flat) == n * n || error("knm_flat shape mismatch")
    k1t = Vector{Float64}(undef, n)
    k1o = Vector{Float64}(undef, n)
    k2t = Vector{Float64}(undef, n)
    k2o = Vector{Float64}(undef, n)
    k3t = Vector{Float64}(undef, n)
    k3o = Vector{Float64}(undef, n)
    k4t = Vector{Float64}(undef, n)
    k4o = Vector{Float64}(undef, n)
    tmp_th = Vector{Float64}(undef, n)
    tmp_od = Vector{Float64}(undef, n)

    _compute_derivative!(theta, omega_dot, power, knm_flat, inertia,
                         damping, n, k1t, k1o)
    @inbounds for i in 1:n
        tmp_th[i] = theta[i] + 0.5 * dt * k1t[i]
        tmp_od[i] = omega_dot[i] + 0.5 * dt * k1o[i]
    end
    _compute_derivative!(tmp_th, tmp_od, power, knm_flat, inertia,
                         damping, n, k2t, k2o)
    @inbounds for i in 1:n
        tmp_th[i] = theta[i] + 0.5 * dt * k2t[i]
        tmp_od[i] = omega_dot[i] + 0.5 * dt * k2o[i]
    end
    _compute_derivative!(tmp_th, tmp_od, power, knm_flat, inertia,
                         damping, n, k3t, k3o)
    @inbounds for i in 1:n
        tmp_th[i] = theta[i] + dt * k3t[i]
        tmp_od[i] = omega_dot[i] + dt * k3o[i]
    end
    _compute_derivative!(tmp_th, tmp_od, power, knm_flat, inertia,
                         damping, n, k4t, k4o)

    new_theta = Vector{Float64}(undef, n)
    new_omega = copy(omega_dot)
    dt6 = dt / 6.0
    @inbounds for i in 1:n
        raw = theta[i] + dt6 * (k1t[i] + 2.0 * k2t[i] + 2.0 * k3t[i] + k4t[i])
        new_theta[i] = mod(raw, TWO_PI)
        new_omega[i] += dt6 * (k1o[i] + 2.0 * k2o[i] + 2.0 * k3o[i] + k4o[i])
    end
    return new_theta, new_omega
end

end  # module
