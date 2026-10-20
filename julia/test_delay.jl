# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native Julia delayed integration and inference

using Test
include("delay.jl")
using .DelayJL

@testset "Actual Julia delay recurrence and inference" begin
    p = [0.4, 1.1]
    omega = ones(2)
    knm = [0.0, 0.5, 0.2, 0.0]
    alpha = fill(0.01, 4)
    expected = [0.45155943106653923, 1.1289796554407023]
    @test (@inferred delayed_kuramoto_run(p, omega, knm, alpha, 2, 0.2, 0.3, 0.01, 3, 4)) ≈ expected atol=1e-13 rtol=0
    @test p == [0.4, 1.1]
    interior = prevfloat(2pi)
    @test delayed_kuramoto_run([-nextfloat(0.0), -0.0, interior], zeros(3), zeros(9), zeros(9), 3, 0.0, 0.0, 0.01, 1, 1) == [0.0, 0.0, interior]
    output = delayed_kuramoto_run([-1.0], [0.0], [0.0], [0.0], 1, 0.0, 0.0, 0.01, typemax(Int), 0)
    @test output == [-1.0]
    output[1] = 8.0
    @test delayed_kuramoto_run([0.4], [1.0], [0.0], [0.0], 1, 0.0, 0.0, 0.01, typemax(Int), 2) ≈ [0.42] atol=1e-14
    for (phases, frequencies, n, zeta, dt, delay, steps) in (
        ([NaN], [0.0], 1, 0.0, 0.01, 1, 1),
        ([0.0], [Inf], 1, 0.0, 0.01, 1, 1),
        ([0.0], [0.0], 0, 0.0, 0.01, 1, 1),
        ([0.0], [0.0], 2, 0.0, 0.01, 1, 1),
        ([0.0], [0.0], true, 0.0, 0.01, 1, 1),
        ([0.0], [0.0], 1, 0.0, 0.01, true, 1),
        ([0.0], [0.0], 1, 0.0, 0.01, 1, false),
        ([0.0], [0.0], big(typemax(Int)) + 1, 0.0, 0.01, 1, 1),
        ([0.0], [0.0], 1, 0.0, 0.0, 1, 1),
        ([0.0], [0.0], 1, 0.0, 0.01, -1, 1),
        ([0.0], [0.0], 1, 0.0, 0.01, 1, -1),
        ([0.0], [floatmax(Float64)], 1, floatmax(Float64), 0.01, 1, 1),
    )
        @test_throws ArgumentError delayed_kuramoto_run(phases, frequencies, [0.0], [0.0], n, zeta, Float64(pi/2), dt, delay, steps)
    end
    @test delayed_kuramoto_run(p, omega, knm, alpha, 2, 0.2, 0.3, 0.01, 3, 4) ≈ expected atol=1e-13 rtol=0
end
