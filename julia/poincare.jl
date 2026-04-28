# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincaré section kernels (Julia port)

"""
poincare.jl — two Poincaré-section kernels.

* ``poincare_section(traj_flat, T, d, normal, offset, direction_id)``
  — generic hyperplane crossings with linear interpolation.
* ``phase_poincare(phases_flat, T, N, oscillator_idx, section_phase)``
  — crossing of one oscillator's unwrapped phase mod ``2π``.

Both return ``(crossings_flat, times, n_crossings)`` where
``crossings_flat`` is pre-allocated to ``T * d`` / ``T * N`` (worst
case) and only the first ``n_crossings * d`` entries are populated.
``direction_id`` is 0 = positive, 1 = negative, 2 = both.
"""

module PoincareJL

using LinearAlgebra

export poincare_section, phase_poincare

const TWO_PI = 2.0 * pi


function poincare_section(
    traj_flat::AbstractVector{Float64},
    t::Integer,
    d::Integer,
    normal::AbstractVector{Float64},
    offset::Float64,
    direction_id::Integer,
)
    length(traj_flat) == t * d || error("traj shape mismatch")
    length(normal) == d || error("normal shape mismatch")
    norm_mag = sqrt(sum(x -> x * x, normal))
    if norm_mag <= 0.0
        return Float64[], Float64[], 0
    end
    n_vec = [x / norm_mag for x in normal]

    # Signed distances.
    signed = zeros(Float64, t)
    @inbounds for i in 1:t
        s = 0.0
        base = (i - 1) * d
        for k in 1:d
            s += traj_flat[base + k] * n_vec[k]
        end
        signed[i] = s - offset
    end

    crossings_flat = zeros(Float64, t * d)
    times = zeros(Float64, t)
    n_cr = 0
    @inbounds for i in 1:(t - 1)
        d0 = signed[i]
        d1 = signed[i + 1]
        is_cross = false
        if direction_id == 0
            is_cross = d0 < 0.0 && d1 >= 0.0
        elseif direction_id == 1
            is_cross = d0 > 0.0 && d1 <= 0.0
        else
            is_cross = d0 * d1 < 0.0
        end
        if !is_cross
            continue
        end
        α = abs(d1 - d0) > 1e-15 ? -d0 / (d1 - d0) : 0.5
        base_i = (i - 1) * d
        base_next = i * d
        for k in 1:d
            xi = traj_flat[base_i + k]
            xj = traj_flat[base_next + k]
            crossings_flat[n_cr * d + k] = xi + α * (xj - xi)
        end
        times[n_cr + 1] = (i - 1) + α
        n_cr += 1
    end
    return crossings_flat, times, n_cr
end


function phase_poincare(
    phases_flat::AbstractVector{Float64},
    t::Integer,
    n::Integer,
    oscillator_idx::Integer,
    section_phase::Float64,
)
    length(phases_flat) == t * n || error("phases shape mismatch")

    # Unwrap target oscillator's phase column.
    target = zeros(Float64, t)
    @inbounds for i in 1:t
        target[i] = phases_flat[(i - 1) * n + oscillator_idx + 1]
    end
    # Explicit unwrap (matches numpy.unwrap default discontinuity=π).
    unwrapped = copy(target)
    @inbounds for i in 2:t
        diff = unwrapped[i] - unwrapped[i - 1]
        if diff > pi
            delta = -TWO_PI * floor((diff + pi) / TWO_PI)
            unwrapped[i] += delta
        elseif diff < -pi
            delta = TWO_PI * floor((-diff + pi) / TWO_PI)
            unwrapped[i] += delta
        end
    end

    shifted = zeros(Float64, t)
    @inbounds for i in 1:t
        v = mod(unwrapped[i] - section_phase, TWO_PI)
        shifted[i] = v
    end

    crossings_flat = zeros(Float64, t * n)
    times = zeros(Float64, t)
    n_cr = 0
    @inbounds for i in 1:(t - 1)
        if shifted[i] > pi && shifted[i + 1] < pi
            denom = shifted[i] - shifted[i + 1] + TWO_PI
            α = denom != 0.0 ? shifted[i] / denom : 0.5
            if α < 0.0
                α = 0.0
            elseif α > 1.0
                α = 1.0
            end
            base_i = (i - 1) * n
            base_next = i * n
            for k in 1:n
                xi = phases_flat[base_i + k]
                xj = phases_flat[base_next + k]
                crossings_flat[n_cr * n + k] = xi + α * (xj - xi)
            end
            times[n_cr + 1] = (i - 1) + α
            n_cr += 1
        end
    end
    return crossings_flat, times, n_cr
end

end  # module
