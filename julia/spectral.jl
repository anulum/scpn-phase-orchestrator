# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral graph eigendecomposition (Julia port)

"""
spectral.jl — symmetric eigendecomposition of the combinatorial
graph Laplacian ``L = D − A`` where ``A`` is the reciprocal
undirected magnitude adjacency ``(|W| + |W'|) / 2`` with zeroed
diagonal.

``spectral_eig(knm_flat, n) -> (eigvals::Vector{Float64},
fiedler::Vector{Float64})``

Returns the eigenvalues in ascending order and the eigenvector
corresponding to ``λ₂`` (index 1 in Julia's 1-based indexing →
index 2 of the sorted array, which is the *second* smallest).
Uses Julia's ``LinearAlgebra.eigen`` which dispatches to LAPACK
``dsyev`` underneath — same numerics as NumPy and Mojo FFI.
"""

module SpectralJL

using LinearAlgebra

export spectral_eig


function spectral_eig(knm_flat::AbstractVector{Float64}, n::Integer)
    length(knm_flat) == n * n || error("knm_flat shape mismatch")
    W = reshape(abs.(knm_flat), (n, n))
    for i in 1:n
        W[i, i] = 0.0
    end
    A = 0.5 * (W + W')
    for i in 1:n
        A[i, i] = 0.0
    end
    degrees = vec(sum(A, dims=2))
    L = Diagonal(degrees) - A
    F = eigen(Symmetric(L))
    eigvals = Vector{Float64}(F.values)
    # Second smallest eigenvector (columns are eigenvectors,
    # Julia is 1-based, so column index 2).
    fiedler = Vector{Float64}(F.vectors[:, 2])
    return eigvals, fiedler
end

end  # module
