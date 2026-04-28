# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen mean-field reduction (Julia port)

"""
reduction.jl — RK4 integrator for the Ott-Antonsen complex-scalar
ODE on the mean-field manifold:

    dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)

``oa_run(z_re, z_im, omega_0, delta, k_coupling, dt, n_steps)
-> (re, im, R, psi)``

Returns the final ``(z_re, z_im)`` along with ``R = |z|`` and
``ψ = arg(z)``. Matches ``spo-engine/src/reduction.rs``
bit-for-bit: same real/imaginary decomposition of the ODE, same
RK4 combining rule.
"""

module ReductionJL

export oa_run


function _oa_deriv(re::Float64, im::Float64, omega_0::Float64,
                   delta::Float64, half_k::Float64)
    abs_sq = re * re + im * im
    lin_re = -delta * re + omega_0 * im
    lin_im = -delta * im - omega_0 * re
    cubic_factor = half_k * (1.0 - abs_sq)
    cub_re = cubic_factor * re
    cub_im = cubic_factor * im
    return (lin_re + cub_re, lin_im + cub_im)
end


function oa_run(
    z_re::Float64,
    z_im::Float64,
    omega_0::Float64,
    delta::Float64,
    k_coupling::Float64,
    dt::Float64,
    n_steps::Integer,
)
    re = z_re
    im = z_im
    half_k = k_coupling / 2.0
    @inbounds for _ in 1:n_steps
        k1r, k1i = _oa_deriv(re, im, omega_0, delta, half_k)
        k2r, k2i = _oa_deriv(re + 0.5 * dt * k1r,
                             im + 0.5 * dt * k1i,
                             omega_0, delta, half_k)
        k3r, k3i = _oa_deriv(re + 0.5 * dt * k2r,
                             im + 0.5 * dt * k2i,
                             omega_0, delta, half_k)
        k4r, k4i = _oa_deriv(re + dt * k3r, im + dt * k3i,
                             omega_0, delta, half_k)
        re += (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        im += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
    end
    r = sqrt(re * re + im * im)
    psi = atan(im, re)
    return (re, im, r, psi)
end

end  # module
