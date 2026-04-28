# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov spectrum (Julia port)

"""
lyapunov.jl

Benettin 1980 / Shimada & Nagashima 1979 Lyapunov spectrum on the
Kuramoto tangent space via RK4 integration + periodic row-oriented
Modified Gram-Schmidt. Matches the Rust / NumPy / Go / Mojo reference
implementations bit-for-bit up to float rounding.
"""

module LyapunovSpectrum

using LinearAlgebra

export lyapunov_spectrum

const TWO_PI = 2.0 * pi


function kuramoto_rhs(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
)
    n = length(phases)
    out = similar(phases)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            s += knm[i, j] * sin(phases[j] - phases[i] - alpha[i, j])
        end
        driving = zeta != 0.0 ? zeta * sin(psi - phases[i]) : 0.0
        out[i] = omegas[i] + s + driving
    end
    return out
end


function kuramoto_jacobian(
    phases::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
)
    n = length(phases)
    J = zeros(Float64, n, n)
    @inbounds for i in 1:n
        for j in 1:n
            if i != j
                J[i, j] = knm[i, j] * cos(phases[j] - phases[i] - alpha[i, j])
            end
        end
    end
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            if i != j
                s += J[i, j]
            end
        end
        driver_diag = zeta != 0.0 ? zeta * cos(psi - phases[i]) : 0.0
        J[i, i] = -(s + driver_diag)
    end
    return J
end


"""
    lyapunov_spectrum(phases_init, omegas, knm_flat, alpha_flat,
                      n, dt, n_steps, qr_interval, zeta, psi) -> Vector{Float64}

Compute the full Lyapunov spectrum. `knm_flat` and `alpha_flat` are
flat row-major representations of the ``(N, N)`` matrices. Return
sorted descending.
"""
function lyapunov_spectrum(
    phases_init::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    dt::Float64,
    n_steps::Integer,
    qr_interval::Integer,
    zeta::Float64,
    psi::Float64,
)
    length(phases_init) == n || error("phases_init shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm shape mismatch")
    length(alpha_flat) == n * n || error("alpha shape mismatch")

    # Row-major flat arrays arrive as Python would store them. Reshape +
    # transpose to recover an (N, N) matrix with Julia's column-major
    # indexing where M[i, j] is the (i, j) entry in the original layout.
    knm = permutedims(reshape(collect(knm_flat), n, n))
    alpha = permutedims(reshape(collect(alpha_flat), n, n))

    phases = collect(phases_init)
    Q = Matrix{Float64}(I, n, n)
    exponents = zeros(Float64, n)
    total_time = 0.0
    dt6 = dt / 6.0

    for step in 1:n_steps
        # RK4 on (phases, Q).
        k1p = kuramoto_rhs(phases, omegas, knm, alpha, zeta, psi)
        k1q = kuramoto_jacobian(phases, knm, alpha, zeta, psi) * Q

        p2 = phases .+ 0.5 .* dt .* k1p
        Q2 = Q .+ 0.5 .* dt .* k1q
        k2p = kuramoto_rhs(p2, omegas, knm, alpha, zeta, psi)
        k2q = kuramoto_jacobian(p2, knm, alpha, zeta, psi) * Q2

        p3 = phases .+ 0.5 .* dt .* k2p
        Q3 = Q .+ 0.5 .* dt .* k2q
        k3p = kuramoto_rhs(p3, omegas, knm, alpha, zeta, psi)
        k3q = kuramoto_jacobian(p3, knm, alpha, zeta, psi) * Q3

        p4 = phases .+ dt .* k3p
        Q4 = Q .+ dt .* k3q
        k4p = kuramoto_rhs(p4, omegas, knm, alpha, zeta, psi)
        k4q = kuramoto_jacobian(p4, knm, alpha, zeta, psi) * Q4

        phases = mod.(
            phases .+ dt6 .* (k1p .+ 2.0 .* k2p .+ 2.0 .* k3p .+ k4p),
            TWO_PI,
        )
        Q = Q .+ dt6 .* (k1q .+ 2.0 .* k2q .+ 2.0 .* k3q .+ k4q)
        total_time += dt

        # Periodic QR reorthogonalisation on rows (Rust convention).
        if step % qr_interval == 0
            F = qr(Q')
            Q = Matrix(F.Q)'
            R = F.R
            @inbounds for i in 1:n
                d = abs(R[i, i])
                if d < 1e-300
                    d = 1e-300
                end
                exponents[i] += log(d)
            end
        end
    end

    if total_time > 0.0
        exponents ./= total_time
    end

    return sort(exponents; rev = true)
end

end  # module
