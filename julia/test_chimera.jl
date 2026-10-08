# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original Julia chimera contracts and inference

using Test
include("chimera.jl")
using .ChimeraJL

@testset "Original Julia chimera contracts" begin
    @test (@inferred local_order_parameter([2.0], [1e-16], 1)) == [0.0]
    @test isempty(@inferred local_order_parameter(Float64[], Float64[], 0))
    p = [1e308, -1e308]
    @test local_order_parameter(p, zeros(4), 2) == [0.0, 0.0]
    @test local_order_parameter(p, [0.0, 1.0, 1.0, 0.0], 2) ≈ [1.0, 1.0] atol=1e-12
    k = [0.0, 1e-300, 1e300, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test (@inferred local_order_parameter([0.0, 0.0, Float64(pi)], k, 3)) ≈ [0.0, 0.0, 1.0] atol=1e-12
    for (phases, knm, n) in (
        ([0.0], [0.0], true), ([0.0], [0.0], -1),
        (Float64[], Float64[], typemax(Int)),
        (Float64[], Float64[], big(typemax(Int)) + 1),
        ([0.0], Float64[], 1), ([0.0, 1.0], zeros(3), 2),
        ([NaN], [0.0], 1), ([Inf], [0.0], 1),
        ([0.0], [NaN], 1), ([0.0], [Inf], 1),
        ([0.0], [1e-14], 1), ([0.0], Float64[], 0),
    )
        @test_throws ArgumentError local_order_parameter(phases, knm, n)
    end
    # A genuine admitted call still works after every refusal above.
    @test local_order_parameter([0.0, 0.0], [0.0, 1.0, 1.0, 0.0], 2) == [1.0, 1.0]
end
