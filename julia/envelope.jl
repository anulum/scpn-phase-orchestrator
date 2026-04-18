# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Envelope kernels (Julia port)

"""
envelope.jl

* ``extract_envelope(amplitudes, window)`` — sliding-window RMS
  over a 1-D amplitude trace. Front-padded with the first valid
  RMS value to match the input length.
* ``envelope_modulation_depth(envelope)`` — ``(max − min) /
  (max + min)`` clipped to ``[0, 1]``.
"""

module EnvelopeJL

export extract_envelope, envelope_modulation_depth


function extract_envelope(
    amps::AbstractVector{Float64},
    window::Integer,
)
    t = length(amps)
    if t == 0
        return Float64[]
    end
    if window < 1
        error("window must be >= 1")
    end
    cs = zeros(Float64, t + 1)
    @inbounds for i in 1:t
        cs[i + 1] = cs[i] + amps[i] * amps[i]
    end
    out = zeros(Float64, t)
    n_valid = t - window + 1
    if n_valid <= 0
        # Window exceeds series — fill with sqrt(mean(x²)).
        s = cs[t + 1]
        v = sqrt(s / Float64(t))
        fill!(out, v)
        return out
    end
    @inbounds for i in 1:n_valid
        v = sqrt((cs[i + window] - cs[i]) / Float64(window))
        out[window - 1 + i] = v
    end
    first_val = out[window]
    @inbounds for i in 1:(window - 1)
        out[i] = first_val
    end
    return out
end


function envelope_modulation_depth(envelope::AbstractVector{Float64})
    t = length(envelope)
    if t == 0
        return 0.0
    end
    vmax = envelope[1]
    vmin = envelope[1]
    @inbounds for i in 2:t
        v = envelope[i]
        if v > vmax
            vmax = v
        end
        if v < vmin
            vmin = v
        end
    end
    denom = vmax + vmin
    if denom <= 0.0
        return 0.0
    end
    return (vmax - vmin) / denom
end

end  # module
