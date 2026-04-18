# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE batched integrator (Julia port)

"""
upde_engine.jl

Batched Kuramoto / Sakaguchi UPDE integrator — three methods
(Euler, RK4, Dormand-Prince RK45 with adaptive step control).
Matches the Rust reference (`spo-engine/src/upde.rs`) line-for-line.
"""

module UPDEEngineJL

using LinearAlgebra

export upde_run

const TWO_PI = 2.0 * pi

# Dormand-Prince (1980) coefficients (shared with `spo-engine/dp_tableau.rs`).
const A21 = 1.0 / 5.0
const A31 = 3.0 / 40.0
const A32 = 9.0 / 40.0
const A41 = 44.0 / 45.0
const A42 = -56.0 / 15.0
const A43 = 32.0 / 9.0
const A51 = 19372.0 / 6561.0
const A52 = -25360.0 / 2187.0
const A53 = 64448.0 / 6561.0
const A54 = -212.0 / 729.0
const A61 = 9017.0 / 3168.0
const A62 = -355.0 / 33.0
const A63 = 46732.0 / 5247.0
const A64 = 49.0 / 176.0
const A65 = -5103.0 / 18656.0
const B5_0 = 35.0 / 384.0
const B5_2 = 500.0 / 1113.0
const B5_3 = 125.0 / 192.0
const B5_4 = -2187.0 / 6784.0
const B5_5 = 11.0 / 84.0
const B4_0 = 5179.0 / 57600.0
const B4_2 = 7571.0 / 16695.0
const B4_3 = 393.0 / 640.0
const B4_4 = -92097.0 / 339200.0
const B4_5 = 187.0 / 2100.0
const B4_6 = 1.0 / 40.0


function compute_derivative!(
    out::AbstractVector{Float64},
    theta::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
    n::Int,
)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            s += knm[i, j] * sin(theta[j] - theta[i] - alpha[i, j])
        end
        driving = zeta != 0.0 ? zeta * sin(psi - theta[i]) : 0.0
        out[i] = omegas[i] + s + driving
    end
    return out
end


function euler_substep!(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    buf::AbstractVector{Float64},
    n::Int,
)
    compute_derivative!(buf, phases, omegas, knm, alpha, zeta, psi, n)
    @inbounds for i in 1:n
        phases[i] += dt * buf[i]
    end
end


function rk4_substep!(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    k1::AbstractVector{Float64},
    k2::AbstractVector{Float64},
    k3::AbstractVector{Float64},
    k4::AbstractVector{Float64},
    tmp::AbstractVector{Float64},
    n::Int,
)
    compute_derivative!(k1, phases, omegas, knm, alpha, zeta, psi, n)
    @inbounds for i in 1:n
        tmp[i] = phases[i] + 0.5 * dt * k1[i]
    end
    compute_derivative!(k2, tmp, omegas, knm, alpha, zeta, psi, n)
    @inbounds for i in 1:n
        tmp[i] = phases[i] + 0.5 * dt * k2[i]
    end
    compute_derivative!(k3, tmp, omegas, knm, alpha, zeta, psi, n)
    @inbounds for i in 1:n
        tmp[i] = phases[i] + dt * k3[i]
    end
    compute_derivative!(k4, tmp, omegas, knm, alpha, zeta, psi, n)
    dt6 = dt / 6.0
    @inbounds for i in 1:n
        phases[i] += dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    end
end


function rk45_step!(
    phases::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm::AbstractMatrix{Float64},
    alpha::AbstractMatrix{Float64},
    zeta::Float64,
    psi::Float64,
    atol::Float64,
    rtol::Float64,
    dt_config::Float64,
    last_dt::Float64,
    k1::AbstractVector{Float64},
    k2::AbstractVector{Float64},
    k3::AbstractVector{Float64},
    k4::AbstractVector{Float64},
    k5::AbstractVector{Float64},
    k6::AbstractVector{Float64},
    k7::AbstractVector{Float64},
    y5::AbstractVector{Float64},
    tmp::AbstractVector{Float64},
    n::Int,
)
    dt = last_dt
    for _ in 0:3
        compute_derivative!(k1, phases, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            tmp[i] = phases[i] + dt * A21 * k1[i]
        end
        compute_derivative!(k2, tmp, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            tmp[i] = phases[i] + dt * (A31 * k1[i] + A32 * k2[i])
        end
        compute_derivative!(k3, tmp, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            tmp[i] = phases[i] + dt * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])
        end
        compute_derivative!(k4, tmp, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            tmp[i] = phases[i] + dt * (
                A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]
            )
        end
        compute_derivative!(k5, tmp, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            tmp[i] = phases[i] + dt * (
                A61 * k1[i] + A62 * k2[i] + A63 * k3[i]
                + A64 * k4[i] + A65 * k5[i]
            )
        end
        compute_derivative!(k6, tmp, omegas, knm, alpha, zeta, psi, n)
        @inbounds for i in 1:n
            y5[i] = phases[i] + dt * (
                B5_0 * k1[i] + B5_2 * k3[i] + B5_3 * k4[i]
                + B5_4 * k5[i] + B5_5 * k6[i]
            )
        end
        compute_derivative!(k7, y5, omegas, knm, alpha, zeta, psi, n)
        err_norm = 0.0
        @inbounds for i in 1:n
            y4_i = phases[i] + dt * (
                B4_0 * k1[i] + B4_2 * k3[i] + B4_3 * k4[i]
                + B4_4 * k5[i] + B4_5 * k6[i] + B4_6 * k7[i]
            )
            err_i = abs(y5[i] - y4_i)
            scale = atol + rtol * max(abs(phases[i]), abs(y5[i]))
            ratio = err_i / scale
            if ratio > err_norm
                err_norm = ratio
            end
        end
        if err_norm <= 1.0
            factor = err_norm > 0.0 ? min(0.9 * err_norm^(-0.2), 5.0) : 5.0
            new_last_dt = min(dt * factor, dt_config * 10.0)
            @inbounds for i in 1:n
                phases[i] = y5[i]
            end
            return new_last_dt
        end
        factor = max(0.9 * err_norm^(-0.25), 0.2)
        dt *= factor
    end
    @inbounds for i in 1:n
        phases[i] = y5[i]
    end
    return dt
end


"""
    upde_run(phases_init, omegas, knm_flat, alpha_flat, n,
             zeta, psi, dt, n_steps, method, n_substeps, atol, rtol)
        -> Vector{Float64}

Integrate the Kuramoto UPDE for ``n_steps`` steps and return the
final phases wrapped to ``[0, 2π)``. ``method`` is
``"euler" | "rk4" | "rk45"``.
"""
function upde_run(
    phases_init::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    n_steps::Integer,
    method::AbstractString,
    n_substeps::Integer,
    atol::Float64,
    rtol::Float64,
)
    length(phases_init) == n || error("phases shape mismatch")
    length(omegas) == n || error("omegas shape mismatch")
    length(knm_flat) == n * n || error("knm shape mismatch")
    length(alpha_flat) == n * n || error("alpha shape mismatch")
    n_substeps >= 1 || error("n_substeps must be ≥ 1")

    knm = permutedims(reshape(collect(knm_flat), n, n))
    alpha = permutedims(reshape(collect(alpha_flat), n, n))
    phases = collect(phases_init)

    k1 = zeros(Float64, n)
    k2 = zeros(Float64, n)
    k3 = zeros(Float64, n)
    k4 = zeros(Float64, n)
    k5 = zeros(Float64, n)
    k6 = zeros(Float64, n)
    k7 = zeros(Float64, n)
    y5 = zeros(Float64, n)
    tmp = zeros(Float64, n)

    last_dt = dt
    sub_dt = dt / Float64(n_substeps)

    for _ in 1:n_steps
        if method == "rk45"
            last_dt = rk45_step!(
                phases, omegas, knm, alpha, zeta, psi,
                atol, rtol, dt, last_dt,
                k1, k2, k3, k4, k5, k6, k7, y5, tmp, n,
            )
        elseif method == "rk4"
            for _ in 1:n_substeps
                rk4_substep!(
                    phases, omegas, knm, alpha, zeta, psi, sub_dt,
                    k1, k2, k3, k4, tmp, n,
                )
            end
        elseif method == "euler"
            for _ in 1:n_substeps
                euler_substep!(
                    phases, omegas, knm, alpha, zeta, psi, sub_dt,
                    k1, n,
                )
            end
        else
            error("unknown method: $method")
        end
        @inbounds for i in 1:n
            phases[i] = mod(phases[i], TWO_PI)
        end
    end

    return phases
end

end  # module
