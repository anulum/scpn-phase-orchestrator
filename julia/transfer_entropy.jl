# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy (Julia port)

"""
transfer_entropy.jl

Phase transfer entropy via binned histograms. Matches the Rust and
NumPy reference bit-for-bit.
"""

module TransferEntropy

export phase_transfer_entropy, transfer_entropy_matrix

const TWO_PI = 2.0 * pi


function conditional_entropy(
    target::AbstractVector{Int},
    condition::AbstractVector{Int},
    n_cond_bins::Integer,
)
    n = length(target)
    n == 0 && return 0.0
    h = 0.0
    # Group target by condition via dictionary of sub-histograms.
    # Matches the NumPy implementation: for each unique condition
    # value c, compute H(target | condition == c) weighted by p(c).
    groups = Dict{Int, Vector{Int}}()
    @inbounds for i in 1:n
        c = condition[i]
        if !haskey(groups, c)
            groups[c] = Int[]
        end
        push!(groups[c], target[i])
    end
    for c in 0:(n_cond_bins - 1)
        if !haskey(groups, c)
            continue
        end
        vals = groups[c]
        count = length(vals)
        count < 2 && continue
        counts = Dict{Int, Int}()
        for v in vals
            counts[v] = get(counts, v, 0) + 1
        end
        sub = 0.0
        for (_, cnt) in counts
            p = cnt / count
            sub += p * log(p + 1e-30)
        end
        h -= (count / n) * sub
    end
    return h
end


"""
    phase_transfer_entropy(source, target, n_bins) -> Float64

Transfer entropy from `source` → `target`. Returns a scalar in
``[0, log(n_bins)]``.
"""
function phase_transfer_entropy(
    source::AbstractVector{Float64},
    target::AbstractVector{Float64},
    n_bins::Integer,
)
    if length(source) < 3 || length(target) < 3
        return 0.0
    end
    n = min(length(source), length(target)) - 1
    bin_width = TWO_PI / n_bins

    src_binned = Vector{Int}(undef, n)
    tgt_binned = Vector{Int}(undef, n)
    tgt_next = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        s = mod(source[i], TWO_PI)
        t = mod(target[i], TWO_PI)
        tn = mod(target[i + 1], TWO_PI)
        src_binned[i] = min(Int(floor(s / bin_width)), n_bins - 1)
        tgt_binned[i] = min(Int(floor(t / bin_width)), n_bins - 1)
        tgt_next[i] = min(Int(floor(tn / bin_width)), n_bins - 1)
    end

    h_yt1_yt = conditional_entropy(tgt_next, tgt_binned, n_bins)
    joint_cond = [tgt_binned[i] * n_bins + src_binned[i] for i in 1:n]
    h_yt1_yt_xt = conditional_entropy(tgt_next, joint_cond, n_bins * n_bins)

    return max(0.0, h_yt1_yt - h_yt1_yt_xt)
end


"""
    transfer_entropy_matrix(phase_series, n_osc, n_time, n_bins) -> Vector{Float64}

Flat row-major ``(N, N)`` pairwise TE matrix. Diagonal is zero.
`phase_series` is a flat ``(n_osc × n_time)`` array row-major
in ``(oscillator, time)``.
"""
function transfer_entropy_matrix(
    phase_series::AbstractVector{Float64},
    n_osc::Integer,
    n_time::Integer,
    n_bins::Integer,
)
    length(phase_series) == n_osc * n_time ||
        error("phase_series shape mismatch")
    te = zeros(Float64, n_osc * n_osc)
    src = zeros(Float64, n_time)
    tgt = zeros(Float64, n_time)
    @inbounds for i in 0:(n_osc - 1)
        for s in 0:(n_time - 1)
            src[s + 1] = phase_series[i * n_time + s + 1]
        end
        for j in 0:(n_osc - 1)
            i == j && continue
            for s in 0:(n_time - 1)
                tgt[s + 1] = phase_series[j * n_time + s + 1]
            end
            te[i * n_osc + j + 1] = phase_transfer_entropy(src, tgt, n_bins)
        end
    end
    return te
end

end  # module
