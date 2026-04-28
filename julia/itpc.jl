# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-Trial Phase Coherence (Julia port)

"""
itpc.jl — Lachaux et al. 1999 inter-trial phase coherence.

    ITPC_t = |⟨e^{iθ_{k,t}}⟩_k|  over ``k ∈ trials``.

Two exported kernels:

* ``compute_itpc(phases_flat, n_trials, n_timepoints) -> Vector{Float64}``
* ``itpc_persistence(phases_flat, n_trials, n_timepoints, pause_indices)
      -> Float64``

``phases_flat`` is the row-major ``(n_trials, n_timepoints)`` matrix
flattened as NumPy would pass it.
"""

module ITPC

export compute_itpc, itpc_persistence


function compute_itpc(
    phases_flat::AbstractVector{Float64},
    n_trials::Integer,
    n_timepoints::Integer,
)
    if n_trials == 0
        return Float64[]
    end
    length(phases_flat) == n_trials * n_timepoints ||
        error("phases shape mismatch")
    out = zeros(Float64, n_timepoints)
    inv_n = 1.0 / Float64(n_trials)
    @inbounds for t in 1:n_timepoints
        sr = 0.0
        si = 0.0
        for k in 0:(n_trials - 1)
            th = phases_flat[k * n_timepoints + t]
            sr += cos(th)
            si += sin(th)
        end
        sr *= inv_n
        si *= inv_n
        out[t] = sqrt(sr * sr + si * si)
    end
    return out
end


function itpc_persistence(
    phases_flat::AbstractVector{Float64},
    n_trials::Integer,
    n_timepoints::Integer,
    pause_indices::AbstractVector{<:Integer},
)
    if length(pause_indices) == 0
        return 0.0
    end
    itpc_full = compute_itpc(phases_flat, n_trials, n_timepoints)
    acc = 0.0
    count = 0
    @inbounds for idx in pause_indices
        if idx >= 0 && idx < n_timepoints
            acc += itpc_full[idx + 1]  # Python 0-indexed → Julia 1-indexed
            count += 1
        end
    end
    return count == 0 ? 0.0 : acc / Float64(count)
end

end  # module
