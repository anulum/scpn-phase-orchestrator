# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strang operator splitting (Julia port)

"""
splitting.jl — Strang second-order operator splitting for the
Kuramoto ODE:

    A(dt/2) — exact rotation by ω
    B(dt)   — RK4 on the coupling-only derivative
    A(dt/2) — exact rotation by ω

``splitting_run(phases, omegas, knm_flat, alpha_flat, n, zeta,
psi, dt, n_steps) -> Vector{Float64}``

Pairwise derivative uses the Rust kernel's sincos expansion on
the alpha-zero branch; nonzero alpha falls back to the direct
``sin(diff)`` form. Matches ``spo-engine/src/splitting.rs``
bit-for-bit.
"""

module SplittingJL

export splitting_run

const TWO_PI = 2.0 * pi


function _compute_coupling_deriv!(
    theta::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    alpha_zero::Bool,
    out::AbstractVector{Float64},
)
    sin_th = Vector{Float64}(undef, n)
    cos_th = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        sin_th[i] = sin(theta[i])
        cos_th[i] = cos(theta[i])
    end
    (zs_psi, zc_psi) = zeta != 0.0 ?
        (zeta * sin(psi), zeta * cos(psi)) : (0.0, 0.0)
    @inbounds for i in 1:n
        offset = (i - 1) * n
        ci = cos_th[i]
        si = sin_th[i]
        acc = 0.0
        if alpha_zero
            for j in 1:n
                acc += knm_flat[offset + j] *
                    (sin_th[j] * ci - cos_th[j] * si)
            end
        else
            for j in 1:n
                acc += knm_flat[offset + j] *
                    sin(theta[j] - theta[i] - alpha_flat[offset + j])
            end
        end
        out[i] = acc
        if zeta != 0.0
            out[i] += zs_psi * ci - zc_psi * si
        end
    end
    return nothing
end


function _rk4_coupling!(
    p::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    alpha_zero::Bool,
)
    k1 = Vector{Float64}(undef, n)
    k2 = Vector{Float64}(undef, n)
    k3 = Vector{Float64}(undef, n)
    k4 = Vector{Float64}(undef, n)
    tmp = Vector{Float64}(undef, n)
    _compute_coupling_deriv!(p, knm_flat, alpha_flat, n, zeta, psi,
                             alpha_zero, k1)
    @inbounds for i in 1:n
        tmp[i] = mod(p[i] + 0.5 * dt * k1[i], TWO_PI)
    end
    _compute_coupling_deriv!(tmp, knm_flat, alpha_flat, n, zeta, psi,
                             alpha_zero, k2)
    @inbounds for i in 1:n
        tmp[i] = mod(p[i] + 0.5 * dt * k2[i], TWO_PI)
    end
    _compute_coupling_deriv!(tmp, knm_flat, alpha_flat, n, zeta, psi,
                             alpha_zero, k3)
    @inbounds for i in 1:n
        tmp[i] = mod(p[i] + dt * k3[i], TWO_PI)
    end
    _compute_coupling_deriv!(tmp, knm_flat, alpha_flat, n, zeta, psi,
                             alpha_zero, k4)
    dt6 = dt / 6.0
    @inbounds for i in 1:n
        p[i] = mod(p[i] + dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]),
                   TWO_PI)
    end
    return nothing
end


function splitting_run(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    n_steps::Integer,
)
    length(phases) == n || error("phases shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm_flat shape mismatch")
    length(alpha_flat) == n * n || error("alpha_flat shape mismatch")
    p = Vector{Float64}(phases)
    alpha_zero = all(a -> a == 0.0, alpha_flat)
    half_dt = 0.5 * dt
    for _ in 1:n_steps
        @inbounds for i in 1:n
            p[i] = mod(p[i] + half_dt * omegas[i], TWO_PI)
        end
        _rk4_coupling!(p, knm_flat, alpha_flat, n, zeta, psi, dt,
                       alpha_zero)
        @inbounds for i in 1:n
            p[i] = mod(p[i] + half_dt * omegas[i], TWO_PI)
        end
    end
    return p
end

end  # module
