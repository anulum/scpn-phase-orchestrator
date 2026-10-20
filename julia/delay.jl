# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Time-delayed Kuramoto integration (Julia port)

"""
delay.jl — explicit-Euler integration of the time-delayed Kuramoto model

    dθ_i/dt = ω_i + Σ_j K_ij·sin(θ_j(t−τ) − θ_i(t) − α_ij) + ζ·sin(Ψ − θ_i)

``delayed_kuramoto_run(phases_init, omegas, knm_flat, alpha_flat, n, zeta, psi,
dt, delay_steps, n_steps) -> final_phases``

A ring buffer of ``delay_steps + 1`` phase snapshots provides the delayed
source phase ``θ_j(t−τ)`` with ``τ = delay_steps·dt``; the first
``delay_steps`` steps use the current snapshot (zero-delay warmup), matching the
NumPy reference ``DelayedEngine.run``.
"""

module DelayJL

export delayed_kuramoto_run

const TWO_PI = 2.0 * pi

"""
    delayed_kuramoto_run(phases_init, omegas, knm_flat, alpha_flat, n,
                         zeta, psi, dt, delay_steps, n_steps)

Return an independent final phase vector from simultaneous Euler updates.
Flattened coupling and lag matrices use row-major order. Before the nonnegative
discrete delay is available, coupling uses the current snapshot. A positive
batch projects phases into `[0, 2pi)` with positive zero; zero steps preserve
the validated input without projection. Only usable history is allocated.

Dimensions and counts must be actual integers in the platform range, and all
numerical inputs must be finite with positive `dt`. Invalid arguments or
nonfinite arithmetic are refused without modifying caller-owned arrays.
"""
function delayed_kuramoto_run(
    phases_init::AbstractVector{Float64},
    omegas::AbstractVector{Float64},
    knm_flat::AbstractVector{Float64},
    alpha_flat::AbstractVector{Float64},
    n::Integer,
    zeta::Float64,
    psi::Float64,
    dt::Float64,
    delay_steps::Integer,
    n_steps::Integer,
)
    any(value -> value isa Bool, (n, delay_steps, n_steps)) &&
        throw(ArgumentError("delay counts must not be boolean"))
    0 < n <= typemax(Int) && 0 <= n_steps <= typemax(Int) &&
        0 <= delay_steps <= typemax(Int) ||
        throw(ArgumentError("delay counts must fit the platform integer range"))
    big_n = Int(n)
    steps = Int(n_steps)
    big_n > 0 || throw(ArgumentError("n must be positive"))
    delay_steps >= 0 && steps >= 0 || throw(ArgumentError("counts must be nonnegative"))
    isfinite(dt) && dt > 0 && isfinite(zeta) && isfinite(psi) ||
        throw(ArgumentError("controls must be finite and dt positive"))
    cells = Base.checked_mul(big_n, big_n)
    length(phases_init) == big_n && length(omegas) == big_n &&
        length(knm_flat) == cells && length(alpha_flat) == cells ||
        throw(ArgumentError("delay buffer dimensions must match n"))
    for values in (phases_init, omegas, knm_flat, alpha_flat)
        all(isfinite, values) || throw(ArgumentError("delay inputs must be finite"))
    end
    p = collect(Float64, phases_init)
    steps == 0 && return p
    delay = Int(min(delay_steps, steps))
    maxbuf = Base.checked_add(delay, 1)
    newp = zeros(Float64, big_n)
    hist = zeros(Float64, Base.checked_mul(maxbuf, big_n))
    alpha_zero = all(==(0.0), alpha_flat)

    @inbounds for i in 0:(steps - 1)
        ring = i % maxbuf
        for j in 1:big_n
            hist[ring * big_n + j] = p[j]
        end
        didx = (delay > 0 && i >= delay) ? ((i - delay) % maxbuf) : ring
        for ii in 1:big_n
            theta_i = p[ii]
            row = (ii - 1) * big_n
            coupling = 0.0
            for jj in 1:big_n
                dj = hist[didx * big_n + jj]
                if alpha_zero
                    coupling += knm_flat[row + jj] * sin(dj - theta_i)
                else
                    coupling += knm_flat[row + jj] * sin(dj - theta_i - alpha_flat[row + jj])
                end
            end
            dtheta = omegas[ii] + coupling
            if zeta != 0.0
                dtheta += zeta * sin(psi - theta_i)
            end
            raw = theta_i + dt * dtheta
            isfinite(raw) || throw(ArgumentError("delay step produced nonfinite phases"))
            wrapped = mod(raw, TWO_PI)
            newp[ii] = wrapped >= TWO_PI || wrapped == 0.0 ? 0.0 : wrapped
        end
        p, newp = newp, p
    end
    return p
end

end  # module
