# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling (Julia port)

"""
pac.jl

Phase-amplitude coupling via Tort et al. 2010 (J. Neurophysiol.)
— bin amplitude by phase, compute KL divergence from uniform,
normalise by ``log(n_bins)``. Matches the NumPy and Rust references
bit-for-bit.

Callable from Python via juliacall through
``src/scpn_phase_orchestrator/upde/_pac_julia.py``.
"""

module PAC

export modulation_index, pac_matrix

const TWO_PI = 2.0 * pi

"""
    modulation_index(theta_low, amp_high, n_bins) -> Float64

Tort 2010 modulation index ∈ [0, 1].
"""
function modulation_index(
    theta_low::AbstractVector{Float64},
    amp_high::AbstractVector{Float64},
    n_bins::Integer,
)
    if n_bins < 2 || isempty(theta_low) || isempty(amp_high)
        return 0.0
    end
    n = min(length(theta_low), length(amp_high))
    bin_width = TWO_PI / n_bins
    mean_amp = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)
    @inbounds for i in 1:n
        wrapped = mod(theta_low[i], TWO_PI)
        k = Int(floor(wrapped / bin_width))
        if k >= n_bins
            k = n_bins - 1
        elseif k < 0
            k = 0
        end
        mean_amp[k + 1] += amp_high[i]
        counts[k + 1] += 1
    end
    @inbounds for k in 1:n_bins
        if counts[k] > 0
            mean_amp[k] /= counts[k]
        end
    end
    total = sum(mean_amp)
    total > 0.0 || return 0.0
    log_n = log(n_bins)
    log_n > 1e-15 || return 0.0
    kl = 0.0
    @inbounds for k in 1:n_bins
        pk = mean_amp[k] / total
        if pk > 0.0
            kl += pk * log(pk * n_bins)
        end
    end
    mi = kl / log_n
    return clamp(mi, 0.0, 1.0)
end

"""
    pac_matrix(phases, amplitudes, t, n, n_bins) -> Vector{Float64}

Flat row-major (N, N) matrix of pairwise modulation indices.
Inputs are flat ``(T·N,)`` arrays row-major in ``(time, osc)``.
"""
function pac_matrix(
    phases::AbstractVector{Float64},
    amplitudes::AbstractVector{Float64},
    t::Integer,
    n::Integer,
    n_bins::Integer,
)
    length(phases) == t * n || error("phases shape mismatch")
    length(amplitudes) == t * n || error("amplitudes shape mismatch")
    result = zeros(Float64, n * n)
    theta_col = zeros(Float64, t)
    amp_col = zeros(Float64, t)
    @inbounds for i in 0:(n - 1)
        for s in 0:(t - 1)
            theta_col[s + 1] = phases[s * n + i + 1]
        end
        for j in 0:(n - 1)
            for s in 0:(t - 1)
                amp_col[s + 1] = amplitudes[s * n + j + 1]
            end
            result[i * n + j + 1] = modulation_index(
                theta_col, amp_col, n_bins
            )
        end
    end
    return result
end

end  # module
