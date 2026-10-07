# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Exported Julia phase attention contracts

using Test
using InteractiveUtils
include(joinpath(@__DIR__, "attnres.jl"))
using .AttnRes

"""Build identity projections for a complete Fourier head partition."""
function identity_weights(width::Int, heads::Int)
    head_width = width ÷ heads
    projection = zeros(Float64, width * width)
    for head in 0:(heads - 1), feature in 0:(head_width - 1)
        projection[head * width * head_width + (head * head_width + feature) * head_width + feature + 1] = 1.0
    end
    output = zeros(Float64, width * width)
    for feature in 0:(width - 1)
        output[feature * width + feature + 1] = 1.0
    end
    return projection, output
end

@testset "phase attention exported numerical API" begin
    for (width, heads) in [(2, 1), (4, 2), (6, 3), (8, 4), (12, 4), (16, 8)]
        projection, output = identity_weights(width, heads)
        phases = [0.1, 0.7]
        coupling = [0.0, -0.3, -0.3, 0.0]
        actual = @inferred attnres_modulate(coupling, phases, projection,
            projection, projection, output, 2, heads, -1, 1.0, 0.5)
        norm = sqrt(width / 2) + 1e-12
        cosine = sum(cos(harmonic * 0.6) for harmonic in 1:(width ÷ 2)) / norm^2
        expected = -0.3 * (1.0 + 0.5 * (1.0 + cosine) / 2.0)
        @test isapprox(actual[2], expected; rtol=0.0, atol=1e-12)
        @test actual[2] == actual[3]
        @test actual[1] == actual[4] == 0.0
    end
    projection, output = identity_weights(2, 1)
    coupling = [0.0, 0.3, 0.3, 0.0]
    phases = [0.0, 0.4]
    @test isempty(attnres_modulate(Float64[], Float64[], projection, projection,
        projection, output, 0, 1, -1, 1.0, 0.5))
    @test attnres_modulate(coupling, phases, projection, projection, projection,
        output, 2, 1, -1, 1.0, 0.0) == coupling
    for strength in (NaN, Inf, -0.1)
        @test_throws ErrorException attnres_modulate(coupling, phases, projection,
            projection, projection, output, 2, 1, -1, 1.0, strength)
    end
    for radius in (-2, 0)
        @test_throws ErrorException attnres_modulate(coupling, phases, projection,
            projection, projection, output, 2, 1, radius, 1.0, 0.0)
    end
    @test_throws ErrorException attnres_modulate(coupling, phases, projection,
        projection, projection, output, 2, 1, -1, nextfloat(0.0), 0.5)
    @test_throws ErrorException attnres_modulate(coupling, [Inf, 0.4], projection,
        projection, projection, output, 2, 1, -1, 1.0, 0.0)
    @test_throws ErrorException attnres_modulate([0.1, 0.3, 0.3, 0.0], phases,
        projection, projection, projection, output, 2, 1, -1, 1.0, 0.0)
    @test_throws ErrorException attnres_modulate([0.0, 0.3, 0.1, 0.0], phases,
        projection, projection, projection, output, 2, 1, -1, 1.0, 0.0)
    @test_throws ErrorException attnres_modulate(Float64[], Float64[], projection,
        projection, projection, output, 0, true, -1, 1.0, 0.0)
    @test_throws ErrorException attnres_modulate(coupling, phases, projection,
        projection, projection, Float64[], 2, 1, -1, 1.0, 0.0)
    zero_weights = zeros(Float64, 4)
    huge = attnres_modulate([0.0, 1e308, 1e308, 0.0], phases, zero_weights,
        zero_weights, zero_weights, zero_weights, 2, 1, -1, 1.0, 0.1)
    @test isapprox(huge[2] / 1e308, 1.05; rtol=0.0, atol=1e-14)
    masked = [0.0, 0.3, 0.2, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0]
    banded = attnres_modulate(masked, [0.1, 0.7, 1.2], projection,
        projection, projection, output, 3, 1, 1, 1.0, 0.5)
    expected_pair = 0.3 * (1.0 + 0.5 * (1.0 + cos(0.6) / (1.0 + 1e-12)^2) / 2.0)
    @test isapprox(banded[2], expected_pair; rtol=0.0, atol=1e-12)
    @test banded[3] == banded[7] == 0.2
    @test banded[6] == banded[8] == 0.0
end

@testset "large gain cosine endpoint bounds" begin
    projection, output = identity_weights(2, 1)
    output .*= 1e120
    phase = 0.1972727272727273
    for edge in (0.3, -0.3), antipodal in (true, false)
        second = antipodal ? phase + pi : phase
        actual = attnres_modulate([0.0, edge, edge, 0.0], [phase, second], projection,
            projection, projection, output, 2, 1, -1, 1.0, 1e16)
        expected = antipodal ? edge : edge * (1.0 + 1e16)
        @test isapprox(actual[2], expected; rtol=2e-15, atol=1e-12)
        @test signbit(actual[2]) == signbit(edge)
        @test abs(edge) <= abs(actual[2]) <= abs(edge) * (1.0 + 1e16)
    end
end

@test isempty(attnres_modulate(Float64[], Float64[], [1.,0.,0.,1.],
    [1.,0.,0.,1.], [1.,0.,0.,1.], [1.,0.,0.,1.],
    0, 1, -1, nextfloat(0.0), 0.5))
