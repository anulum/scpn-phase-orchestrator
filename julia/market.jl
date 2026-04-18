# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Financial market PLV / R(t) (Julia port)

"""
market.jl — per-row Kuramoto order parameter and windowed
phase-locking-value matrix on a ``(T, N)`` phase time series.

``market_order_parameter(phases_flat, t, n) -> Vector{Float64}``
  ``R(t) = |⟨exp(iθ)⟩_N|`` at every timestep (length ``T``).

``market_plv(phases_flat, t, n, window) -> Vector{Float64}``
  Flattened ``(n_windows × N × N)`` PLV matrix. For each rolling
  window ``[t, t+W)`` and each pair ``(i, j)`` computes

      PLV_ij = |⟨exp(i·(θ_j − θ_i))⟩_W|

  using the sincos expansion (``cos(θ_j − θ_i) = c_j·c_i + s_j·s_i``,
  ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i``) so the inner loop is
  trig-free and matches Rust bit-for-bit.
"""

module MarketJL

export market_order_parameter, market_plv


function market_order_parameter(
    phases_flat::AbstractVector{Float64},
    t::Integer,
    n::Integer,
)
    if n == 0 || t == 0
        return Float64[]
    end
    length(phases_flat) == t * n || error("phases_flat shape mismatch")
    inv_n = 1.0 / Float64(n)
    out = Vector{Float64}(undef, t)
    @inbounds for row in 1:t
        sum_cos = 0.0
        sum_sin = 0.0
        base = (row - 1) * n
        for i in 1:n
            θ = phases_flat[base + i]
            sum_cos += cos(θ)
            sum_sin += sin(θ)
        end
        mc = sum_cos * inv_n
        ms = sum_sin * inv_n
        out[row] = sqrt(mc * mc + ms * ms)
    end
    return out
end


function market_plv(
    phases_flat::AbstractVector{Float64},
    t::Integer,
    n::Integer,
    window::Integer,
)
    if t < window || n == 0 || window == 0
        return Float64[]
    end
    length(phases_flat) == t * n || error("phases_flat shape mismatch")
    n_windows = t - window + 1
    plv = Vector{Float64}(undef, n_windows * n * n)
    inv_w = 1.0 / Float64(window)

    @inbounds for w in 0:(n_windows - 1)
        # Precompute sincos for this window (O(N·W) trig per window)
        window_s = Vector{Float64}(undef, window * n)
        window_c = Vector{Float64}(undef, window * n)
        for k in 0:(window - 1)
            base_step = (w + k) * n
            for i in 1:n
                θ = phases_flat[base_step + i]
                window_s[k * n + i] = sin(θ)
                window_c[k * n + i] = cos(θ)
            end
        end
        mat_offset = w * n * n
        for i in 1:n
            for j in 1:n
                sum_cos = 0.0
                sum_sin = 0.0
                for k in 0:(window - 1)
                    si = window_s[k * n + i]
                    ci = window_c[k * n + i]
                    sj = window_s[k * n + j]
                    cj = window_c[k * n + j]
                    sum_cos += cj * ci + sj * si
                    sum_sin += sj * ci - cj * si
                end
                mc = sum_cos * inv_w
                ms = sum_sin * inv_w
                plv[mat_offset + (i - 1) * n + j] = sqrt(mc * mc + ms * ms)
            end
        end
    end
    return plv
end

end  # module
