using CartesianOperators
using CartesianOperators: _pattern!, _fill!
using FillArrays
using SparseArrays
using Test

@testset "LinearStencil constructor" begin
    # inner-constructor validation (well-typed inputs, invalid offset order / D)
    @test_throws ArgumentError LinearStencil{1}((0, 1), (Fill(1.0, 5), Fill(-1.0, 5)))   # ascending
    @test_throws ArgumentError LinearStencil{1}((1, 1), (Fill(1.0, 5), Fill(-1.0, 5)))   # equal
    @test_throws ArgumentError LinearStencil{0}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))   # D < 1
    @test_throws ArgumentError LinearStencil{-1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))  # D < 1
    # outer-constructor friendly errors (ill-typed coefs)
    @test_throws ArgumentError LinearStencil{1}((1, 0), (1.0, -1.0))                     # scalars, not arrays
    @test_throws ArgumentError LinearStencil{1}((1, 0), (Fill(1.0, 5),))                 # length mismatch
    @test_throws ArgumentError LinearStencil{1}((1, 0), (Fill(1f0, 5), Fill(-1.0, 5)))   # mixed eltype
    @test_throws ArgumentError LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, (5, 1))))  # mixed ndims
    # happy path: heterogeneous-container tuple typechecks
    st = LinearStencil{1}((1, 0), (Fill(1.0, 5), [-1.0, -1.0, -1.0, -1.0, -1.0]))
    @test st.offsets == (1, 0)
    @test st.coefs[1] == Fill(1.0, 5)
    @test st.coefs[2] == fill(-1.0, 5)
end

@testset "_pattern! (1-D kernel)" begin
    @testset "row=col=1:5, offsets (1, 0)" begin
        row = 1:5; col = 1:5
        offsets = (1, 0)
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        @test colptr == [1, 2, 4, 6, 8, 10]
        @test rowval == [1, 1, 2, 2, 3, 3, 4, 4, 5]
    end

    @testset "shifted: row=1:5, col=3:7, offsets (0,)" begin
        row = 1:5; col = 3:7
        offsets = (0,)
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        # c=3→r=3; c=4→r=4; c=5→r=5; c=6,7→out
        @test colptr == [1, 2, 3, 4, 4, 4]
        @test rowval == [3, 4, 5]
    end
end

@testset "_fill! (1-D kernel)" begin
    @testset "row=col=1:5, offsets (1, 0), constant coefs via Fill" begin
        row = 1:5; col = 1:5
        offsets = (1, 0); coefs = (Fill(1.0, 5), Fill(-1.0, 5))
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        nzval = Vector{Float64}(undef, length(rowval))
        _fill!(nzval, colptr, offsets, coefs, row, col)
        @test nzval == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    end
end

@testset "assemble + update! 1-D" begin
    @testset "forward_x" begin
        @testset "row=col=(1:5,)" begin
            row = (1:5,); col = (1:5,)
            st = LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "Float32, row=col=(1:4,)" begin
            row = (1:4,); col = (1:4,)
            st32 = LinearStencil{1}((1, 0), (Fill(1f0, 4), Fill(-1f0, 4)))
            J = assemble(st32, row, col); update!(J, st32, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1f0, -1f0), row, col)
            @test J == ref
            @test eltype(J) == Float32
        end
        @testset "unequal lengths: row=(1:5,), col=(1:3,)" begin
            row = (1:5,); col = (1:3,)
            st = LinearStencil{1}((1, 0), (Fill(1.0, 3), Fill(-1.0, 3)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "shifted ranges: row=(1:5,), col=(3:7,)" begin
            # coefs indexed at c ∈ col[1] = 3:7; Fill(.., 7) has axes 1:7, covers 3..7.
            row = (1:5,); col = (3:7,)
            st = LinearStencil{1}((1, 0), (Fill(1.0, 7), Fill(-1.0, 7)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
    end

    @testset "backward_x, row=col=(1:5,)" begin
        row = (1:5,); col = (1:5,)
        st = LinearStencil{1}((0, -1), (Fill(1.0, 5), Fill(-1.0, 5)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)
        @test J == ref
    end

    @testset "central_x, row=col=(1:5,)" begin
        row = (1:5,); col = (1:5,)
        st = LinearStencil{1}((1, -1), (Fill(1.0, 5), Fill(-1.0, 5)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)
        @test J == ref
    end

    @testset "variable coefs: density-weighted gradient" begin
        # ψ[i] = (φ[i] − φ[i−1]) / ρ[i] on row = col = (1:4,).
        # Offsets (0, -1); column-anchored coefs.
        #   slot Δ=0:  matrix entry (r=c, c)   → coef +1/ρ[c]    → coefs[1] = 1 ./ ρ
        #   slot Δ=-1: matrix entry (r=c+1, c) → coef −1/ρ[c+1]  → coefs[2] is the +1-shifted -1/ρ
        ρ = [2.0, 3.0, 5.0, 7.0]
        coef0  = 1 ./ ρ                          # length 4, indexed by c
        coefm1 = vcat(-1 ./ ρ[2:end], 0.0)       # length 4; index 4 never read (c=4, Δ=-1 → r=5 ∉ row)
        row = (1:4,); col = (1:4,)
        st = LinearStencil{1}((0, -1), (coef0, coefm1))
        J = assemble(st, row, col); update!(J, st, row, col)
        I_exp = [1, 2, 2, 3, 3, 4, 4]
        J_exp = [1, 1, 2, 2, 3, 3, 4]
        V_exp = [1/ρ[1], -1/ρ[2], 1/ρ[2], -1/ρ[3], 1/ρ[3], -1/ρ[4], 1/ρ[4]]
        ref = sparse(I_exp, J_exp, V_exp, 4, 4)
        @test J == ref
    end
end

@testset "D ≤ N constructor invariant" begin
    # D > N is rejected at construction.
    @test_throws ArgumentError LinearStencil{2}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))
    @test_throws ArgumentError LinearStencil{3}((1, 0), (Fill(1.0, (5,3)), Fill(-1.0, (5,3))))
end

@testset "N mismatch (1-D row/col requires LinearStencil with N=1)" begin
    # Constructor succeeds (D=1 ≤ N=2) but 1-D assemble/update! pin N=1 → MethodError.
    st_n2 = LinearStencil{1}((1, 0), (Fill(1.0, (5,3)), Fill(-1.0, (5,3))))
    row = (1:5,); col = (1:5,)
    @test_throws MethodError assemble(st_n2, row, col)
    J = assemble(LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5))), row, col)
    @test_throws MethodError update!(J, st_n2, row, col)
end

@testset "build convenience" begin
    # build(...) == update!(assemble(...), ...)
    row = (1:5,); col = (1:5,)
    st = LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))
    J_two_step = update!(assemble(st, row, col), st, row, col)
    J_build    = build(st, row, col)
    @test J_build == J_two_step
end
