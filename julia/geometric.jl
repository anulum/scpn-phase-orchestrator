# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus symplectic Euler (Julia port)

"""
geometric.jl — torus-preserving geometric integrator.

``torus_run(phases, omegas, knm_flat, alpha_flat, n, zeta, psi,
dt, n_steps) -> Vector{Float64}``

Lifts phases to ``z_i = (cos θ_i, sin θ_i)``, computes the
Kuramoto derivative ``ω_eff_i`` in the tangent space, rotates
``z_i`` by ``ω_eff_i · dt`` via the exponential map, renormalises
to the unit circle, and projects back to ``[0, 2π)``. Matches
``spo-engine/src/geometric.rs`` bit-for-bit: sincos expansion
on the alpha-zero branch, atan2 reconstruction otherwise.
"""

module GeometricJL

export torus_run

const TWO_PI = 2.0 * pi


function torus_run(
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

    z_re = Vector{Float64}(undef, n)
    z_im = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        z_re[i] = cos(phases[i])
        z_im[i] = sin(phases[i])
    end

    alpha_zero = all(a -> a == 0.0, alpha_flat)
    (zs_psi, zc_psi) = zeta != 0.0 ?
        (zeta * sin(psi), zeta * cos(psi)) : (0.0, 0.0)

    next_re = Vector{Float64}(undef, n)
    next_im = Vector{Float64}(undef, n)

    for _ in 1:n_steps
        @inbounds for i in 1:n
            coupling = 0.0
            offset = (i - 1) * n
            if alpha_zero
                for j in 1:n
                    coupling += knm_flat[offset + j] *
                        (z_im[j] * z_re[i] - z_re[j] * z_im[i])
                end
            else
                ti = atan(z_im[i], z_re[i])
                for j in 1:n
                    tj = atan(z_im[j], z_re[j])
                    coupling += knm_flat[offset + j] *
                        sin(tj - ti - alpha_flat[offset + j])
                end
            end
            omega_eff = omegas[i] + coupling
            if zeta != 0.0
                omega_eff += zs_psi * z_re[i] - zc_psi * z_im[i]
            end
            angle = omega_eff * dt
            sin_a = sin(angle)
            cos_a = cos(angle)
            nr = z_re[i] * cos_a - z_im[i] * sin_a
            ni = z_re[i] * sin_a + z_im[i] * cos_a
            norm_ = sqrt(nr * nr + ni * ni)
            if norm_ > 0.0
                next_re[i] = nr / norm_
                next_im[i] = ni / norm_
            else
                next_re[i] = nr
                next_im[i] = ni
            end
        end
        @inbounds for i in 1:n
            z_re[i] = next_re[i]
            z_im[i] = next_im[i]
        end
    end

    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = mod(atan(z_im[i], z_re[i]), TWO_PI)
    end
    return out
end

end  # module
