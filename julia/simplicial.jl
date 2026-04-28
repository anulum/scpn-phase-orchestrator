# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial (higher-order) Kuramoto (Julia port)

"""
simplicial.jl — pairwise + 3-body (all-to-all simplicial) Kuramoto
stepper.

``simplicial_run(phases, omegas, knm_flat, alpha_flat, n, zeta,
psi, sigma2, dt, n_steps) -> Vector{Float64}``

Uses the O(N²) closed-form

    Σ_{j,k} sin(θ_j + θ_k − 2θ_i) = 2 · S_i · C_i

with ``S_i = Σ_j sin(θ_j − θ_i)``, ``C_i = Σ_j cos(θ_j − θ_i)``.
Pairwise uses the sincos expansion on the alpha-zero branch for
bit-exact parity with Rust (``spo-engine/src/simplicial.rs``).
"""

module SimplicialJL

export simplicial_run

const TWO_PI = 2.0 * pi


function _compute_derivative!(
    theta::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    sigma2::Float64,
    alpha_zero::Bool,
    deriv::AbstractVector{Float64},
)
    sin_th = Vector{Float64}(undef, n)
    cos_th = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        sin_th[i] = sin(theta[i])
        cos_th[i] = cos(theta[i])
    end
    (zs_psi, zc_psi) = zeta != 0.0 ?
        (zeta * sin(psi), zeta * cos(psi)) : (0.0, 0.0)
    (gs, gc) = if sigma2 != 0.0 && n >= 3
        (sum(sin_th), sum(cos_th))
    else
        (0.0, 0.0)
    end
    inv_n2 = n > 0 ? sigma2 / (Float64(n) * Float64(n)) : 0.0

    @inbounds for i in 1:n
        offset = (i - 1) * n
        ci = cos_th[i]
        si = sin_th[i]
        pw = 0.0
        if alpha_zero
            for j in 1:n
                pw += knm_flat[offset + j] *
                    (sin_th[j] * ci - cos_th[j] * si)
            end
        else
            for j in 1:n
                pw += knm_flat[offset + j] *
                    sin(theta[j] - theta[i] - alpha_flat[offset + j])
            end
        end
        deriv[i] = omegas[i] + pw
        if sigma2 != 0.0 && n >= 3
            deriv[i] += 2.0 * (gs * ci - gc * si) *
                (gc * ci + gs * si) * inv_n2
        end
        if zeta != 0.0
            deriv[i] += zs_psi * ci - zc_psi * si
        end
    end
    return nothing
end


function simplicial_run(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    sigma2::Float64,
    dt::Float64,
    n_steps::Integer,
)
    length(phases) == n || error("phases shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm_flat shape mismatch")
    length(alpha_flat) == n * n || error("alpha_flat shape mismatch")
    alpha_zero = all(a -> a == 0.0, alpha_flat)
    p = Vector{Float64}(phases)
    deriv = Vector{Float64}(undef, n)
    for _ in 1:n_steps
        _compute_derivative!(p, omegas, knm_flat, alpha_flat, n,
                             zeta, psi, sigma2, alpha_zero, deriv)
        @inbounds for i in 1:n
            p[i] = mod(p[i] + dt * deriv[i], TWO_PI)
        end
    end
    return p
end

end  # module
