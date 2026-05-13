using CartesianOperators
using CartesianRuns
using SparseArrays
using Test

@testset "CartesianOperators.jl" begin
    @testset "smoke" begin
        @test isdefined(CartesianOperators, :CartesianOperators)
    end
end
