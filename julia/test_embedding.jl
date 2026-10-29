# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native embedding mutual-information binning contracts

using Test
include("embedding.jl")
using .EmbeddingJL

@testset "Embedding mutual-information binning" begin
    # Edges 0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004: the two
    # interior-edge samples open their own bins, so lag 0 gives entropy log(4).
    signal = [0.30000000000000004, 0.1, 0.2, 0.15000000000000002]
    @test mutual_information(signal, 0, 4) == log(4.0)
    edges = EmbeddingJL.histogram_edges(0.0, 4.0, 4)
    @test [EmbeddingJL.histogram_bin(edges, v) for v in (0.0, 0.999, 1.0, 3.0, 4.0)] == [0, 0, 1, 3, 3]
    # Separate window ranges: x = [0, 1], y = [10, 20] at lag 4.
    @test isapprox(mutual_information([0.0, 1.0, 0.0, 1.0, 10.0, 20.0], 4, 2), log(2.0); atol=1e-15)
end
