# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native dimension contracts

using Test
include("dimension.jl")
using .DimensionJL

@testset "Dimension numerical contracts" begin
    for (point, radius, expected) in [
        (floatmax(Float64) / 2, floatmax(Float64), 1.0),
        (1e-200, 5e-201, 0.0), (1.0, 1.0, 0.0),
        (1.0, nextfloat(1.0), 1.0), (0.0, 0.0, 0.0),
        (0.0, nextfloat(0.0), 1.0),
    ]
        @test @inferred(correlation_integral([0.0, point], 2, 1, [0], [1], [radius])) == [expected]
    end
    largest = floatmax(Float64)
    @test @inferred(kaplan_yorke_dimension([largest, largest, -largest, -largest, -largest])) == 4.0
    for scale in [1.0, 1e-305, 1e305]
        @test kaplan_yorke_dimension([-2scale, 0.0, scale]) == 2.5
    end
    @test correlation_integral([0.0, 1.0], 2, 1, Int[], Int[], [1.0]) == [0.0]
    @test kaplan_yorke_dimension(Float64[]) == 0.0
end

@testset "Dimension checked native boundary" begin
    @test_throws ArgumentError correlation_integral([0.0, 1.0], 2, 1, [0], [2], [1.0])
    @test_throws ArgumentError correlation_integral([0.0, 1.0], 2, 1, [0], [0], [1.0])
    @test_throws ArgumentError correlation_integral([0.0, 1.0], 2, 1, [0], [1], [2.0, 1.0])
    @test_throws ArgumentError correlation_integral([0.0, Inf], 2, 1, [0], [1], [1.0])
    @test_throws ArgumentError correlation_integral(Float64[], false, 1, Int[], Int[], [1.0])
    @test_throws ArgumentError correlation_integral([0.0, 1.0], 2, 1, [false], [true], [1.0])
    @test_throws OverflowError correlation_integral(Float64[], typemax(Int), 2, Int[], Int[], [1.0])
    @test_throws ArgumentError kaplan_yorke_dimension([NaN])
end
