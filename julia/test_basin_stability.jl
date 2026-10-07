# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native finite-horizon trial contracts

using Test
include(joinpath(@__DIR__, "basin_stability.jl"))
using .BasinStabilityJL

@testset "original exported Euler trial" begin
    p = [0.0, pi/2]; o = zeros(2); a = zeros(4)
    @test isapprox(@inferred(steady_state_r(p,o,[0.0,5e-31,5e-31,0.0],a,2,1.0,1e30,0,1)), cos((pi/2-1)/2); atol=2e-15,rtol=0)
    @test p == [0.0,pi/2]
    @test steady_state_r(p,[1e308,1e308],a,a,2,1.0,2.0,typemax(Int),0) == 0.0
    @test isapprox(steady_state_r([1e308,-1e308],o,a,a,2,1.0,1.0,0,1),abs(cos(1e308));atol=2e-15,rtol=0)
    k = [0.4,-0.7,0.0,0.5]; lag = [0.3,-0.2,0.0,-0.4]; omega=[0.2,-0.1]
    theta0=0.09*(omega[1]+k[1]*sin(-lag[1])+k[2]*sin(pi/2-lag[2]))
    theta1=pi/2+0.09*(omega[2]+k[4]*sin(-lag[4]))
    @test isapprox(steady_state_r(p,omega,k,lag,2,1.0,0.09,0,1),abs(cos((theta1-theta0)/2));atol=2e-15,rtol=0)
    for n in [0, -1, true, typemax(Int)]
        @test_throws ArgumentError steady_state_r(p,o,a,a,n,1.0,0.1,0,1)
    end
    for (transient,measure) in [(-1,1),(0,-1),(true,1),(0,true),(big(typemax(Int))+1,1)]
        @test_throws ArgumentError steady_state_r(p,o,a,a,2,1.0,0.1,transient,measure)
    end
    for (scale,dt) in [(Inf,0.1),(1.0,NaN),(1.0,0.0),(1.0,-0.1)]
        @test_throws ArgumentError steady_state_r(p,o,a,a,2,scale,dt,0,1)
    end
    for (pp,oo,kk,aa) in [(p[1:1],o,a,a),(p,o[1:1],a,a),(p,o,a[1:3],a),(p,o,a,a[1:3]),([NaN,0.0],o,a,a),(p,[Inf,0.0],a,a),(p,o,[0.0,NaN,0.0,0.0],a),(p,o,a,[0.0,0.0,Inf,0.0])]
        @test_throws ArgumentError steady_state_r(pp,oo,kk,aa,2,1.0,0.1,0,0)
    end
    @test_throws ArgumentError steady_state_r(p,o,[0.0,1e308,0.0,0.0],a,2,2.0,1.0,0,1)
    @test_throws ArgumentError steady_state_r([1e308,-1e308],o,[0.0,1.0,0.0,0.0],a,2,1.0,1.0,0,1)
    @test_throws ArgumentError steady_state_r(p,[1e308,1e308],a,a,2,1.0,2.0,0,1)
    @test_throws ArgumentError steady_state_r(p,[1e308,1e308],a,a,2,1.0,2.0,1,1)
    @test steady_state_r(zeros(2),o,a,a,2,1.0,0.1,2,3) == 1.0
end
