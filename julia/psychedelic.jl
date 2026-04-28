# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic observables (Julia port)

"""
psychedelic.jl — circular phase Shannon entropy.

``entropy_from_phases(phases, n_bins)`` wraps phases to ``[0, 2π)``,
bins into ``n_bins`` equal-width intervals, returns the entropy of
the normalised histogram in nats.
"""

module PsychedelicJL

export entropy_from_phases

const TWO_PI = 2.0 * pi


function entropy_from_phases(
    phases::AbstractVector{Float64},
    n_bins::Integer,
)
    t = length(phases)
    if t == 0
        return 0.0
    end
    counts = zeros(Int64, n_bins)
    bin_width = TWO_PI / Float64(n_bins)
    @inbounds for i in 1:t
        v = mod(phases[i], TWO_PI)
        bx = Int(floor(v / bin_width))
        if bx >= n_bins
            bx = n_bins - 1
        end
        counts[bx + 1] += 1
    end
    total = Float64(t)
    h = 0.0
    @inbounds for k in 1:n_bins
        c = counts[k]
        if c > 0
            p = Float64(c) / total
            h -= p * log(p)
        end
    end
    return h
end

end  # module
