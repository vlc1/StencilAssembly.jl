using CartesianOperators
using SparseArrays
using Test

include("reference.jl")

@testset "CartesianOperators.jl" begin
    @testset "smoke" begin
        @test isdefined(CartesianOperators, :CartesianOperators)
    end

    @testset "reference helper" begin
        row = (1:3,); col = (1:3,)
        offsets = (CartesianIndex(1), CartesianIndex(0))
        coefs = (1.0, -1.0)
        M = stencil_reference(offsets, coefs, row, col)
        @test M == sparse([1, 1, 2, 2, 3], [1, 2, 2, 3, 3], [-1.0, 1.0, -1.0, 1.0, -1.0], 3, 3)
    end
end

include("test_stencil.jl")
