# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameters (Julia port)

"""
order_params.jl

Kuramoto global order parameter ``R`` and the phase-locking value
``PLV``. Full-precision Julia ports that mirror the Rust kernel in
``spo-kernel/crates/spo-engine/src/order_params.rs`` and the NumPy
reference in ``src/scpn_phase_orchestrator/upde/order_params.py``.

Callable from Python via ``juliacall`` through
``src/scpn_phase_orchestrator/coupling/_order_params_julia.py``.
"""

module OrderParams

export order_parameter, plv, layer_coherence

const TWO_PI = 2.0 * pi

"""
    order_parameter(phases) -> (Float64, Float64)

Return ``(R, psi)`` where ``R = |mean(exp(i·θ))|`` and
``psi = arg(mean(exp(i·θ))) mod 2π``.
"""
function order_parameter(phases::AbstractVector{Float64})
    n = length(phases)
    n == 0 && return (0.0, 0.0)
    sx = 0.0
    sy = 0.0
    @inbounds for i in 1:n
        sx += cos(phases[i])
        sy += sin(phases[i])
    end
    sx /= n
    sy /= n
    r = sqrt(sx * sx + sy * sy)
    psi = mod(atan(sy, sx), TWO_PI)
    return (r, psi)
end

"""
    plv(phases_a, phases_b) -> Float64

Phase-locking value between two equal-length phase series.
Raises on length mismatch.
"""
function plv(
    phases_a::AbstractVector{Float64},
    phases_b::AbstractVector{Float64},
)
    n = length(phases_a)
    n == 0 && return 0.0
    length(phases_b) == n ||
        error("PLV requires equal-length arrays")
    sx = 0.0
    sy = 0.0
    @inbounds for i in 1:n
        diff = phases_a[i] - phases_b[i]
        sx += cos(diff)
        sy += sin(diff)
    end
    sx /= n
    sy /= n
    return sqrt(sx * sx + sy * sy)
end

"""
    layer_coherence(phases, indices) -> Float64

Order parameter R restricted to the oscillators at ``indices``
(0-based; matches the Python / Rust convention).
"""
function layer_coherence(
    phases::AbstractVector{Float64},
    indices::AbstractVector{<:Integer},
)
    n = length(indices)
    n == 0 && return 0.0
    sx = 0.0
    sy = 0.0
    @inbounds for i in 1:n
        idx = Int(indices[i]) + 1  # 0→1 index translation
        sx += cos(phases[idx])
        sy += sin(phases[idx])
    end
    sx /= n
    sy /= n
    return sqrt(sx * sx + sy * sy)
end

end  # module
