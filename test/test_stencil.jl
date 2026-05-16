using CartesianOperators
using CartesianOperators: _pattern_runs!, _fill_runs!
using CartesianRuns
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

@testset "_pattern_runs! (1-D base kernel)" begin
    @testset "full mask, offsets (1, 0)" begin
        row = CartesianRunIndices(trues(5))
        col = CartesianRunIndices(trues(5))
        offsets = (1, 0)
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern_runs!(
            rowval, colptr, offsets,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        @test colptr == [1, 2, 4, 6, 8, 10]
        @test rowval == [1, 1, 2, 2, 3, 3, 4, 4, 5]
    end

    @testset "holed masks, offsets (1, 0)" begin
        row = CartesianRunIndices(Bool[1, 0, 1, 1])
        col = CartesianRunIndices(Bool[1, 1, 0, 1])
        offsets = (1, 0)
        colptr = Vector{Int}(undef, 4); colptr[1] = 1
        rowval = Int[]
        _pattern_runs!(
            rowval, colptr, offsets,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        @test colptr == [1, 2, 3, 5]
        @test rowval == [1, 1, 2, 3]
    end
end

@testset "_fill_runs! (1-D base kernel)" begin
    @testset "full mask, offsets (1, 0), constant coefs via Fill" begin
        row = CartesianRunIndices(trues(5))
        col = CartesianRunIndices(trues(5))
        offsets = (1, 0); coefs = (Fill(1.0, 5), Fill(-1.0, 5))
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern_runs!(
            rowval, colptr, offsets,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        nzval = Vector{Float64}(undef, length(rowval))
        _fill_runs!(
            nzval, colptr, offsets, coefs,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        @test nzval == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    end
end

@testset "assemble + update! 1-D" begin
    @testset "forward_x" begin
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            st = LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            st = LinearStencil{1}((1, 0), (Fill(1.0, 4), Fill(-1.0, 4)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "Float32" begin
            row = CartesianRunIndices(trues(4)); col = CartesianRunIndices(trues(4))
            st32 = LinearStencil{1}((1, 0), (Fill(1f0, 4), Fill(-1f0, 4)))
            J = assemble(st32, row, col)
            update!(J, st32, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1f0, -1f0), row, col)
            @test J == ref
            @test eltype(J) == Float32
        end
    end

    @testset "backward_x" begin
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            st = LinearStencil{1}((0, -1), (Fill(1.0, 5), Fill(-1.0, 5)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            st = LinearStencil{1}((0, -1), (Fill(1.0, 4), Fill(-1.0, 4)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
    end

    @testset "central_x" begin
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            st = LinearStencil{1}((1, -1), (Fill(1.0, 5), Fill(-1.0, 5)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            st = LinearStencil{1}((1, -1), (Fill(1.0, 4), Fill(-1.0, 4)))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
    end

    @testset "variable coefs: density-weighted gradient" begin
        # ψ[i] = (φ[i] − φ[i−1]) / ρ[i] on a full mask 1:4.
        # Offsets (0, -1); column-anchored coefs.
        #   slot Δ=0:  matrix entry (r=c, c) → coef +1/ρ[c]  → coefs[1] = 1 ./ ρ
        #   slot Δ=-1: matrix entry (r=c+1, c) → coef −1/ρ[c+1] → coefs[2] is the +1-shifted -1/ρ
        ρ = [2.0, 3.0, 5.0, 7.0]
        coef0  = 1 ./ ρ                          # length 4, indexed by c
        coefm1 = vcat(-1 ./ ρ[2:end], 0.0)       # length 4; index 4 never read (c=4, Δ=-1 → r=5 ∉ row mask)
        row = CartesianRunIndices(trues(4)); col = CartesianRunIndices(trues(4))
        st = LinearStencil{1}((0, -1), (coef0, coefm1))
        J = assemble(st, row, col)
        update!(J, st, row, col)
        I_exp = [1, 2, 2, 3, 3, 4, 4]
        J_exp = [1, 1, 2, 2, 3, 3, 4]
        V_exp = [1/ρ[1], -1/ρ[2], 1/ρ[2], -1/ρ[3], 1/ρ[3], -1/ρ[4], 1/ρ[4]]
        ref = sparse(I_exp, J_exp, V_exp, 4, 4)
        @test J == ref
    end
end

@testset "D mismatch (1-D cri requires LinearStencil{1})" begin
    st_dim2 = LinearStencil{2}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5)))
    row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
    @test_throws ArgumentError assemble(st_dim2, row, col)
    J = assemble(LinearStencil{1}((1, 0), (Fill(1.0, 5), Fill(-1.0, 5))), row, col)
    @test_throws ArgumentError update!(J, st_dim2, row, col)
end
