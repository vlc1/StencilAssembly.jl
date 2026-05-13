using CartesianOperators
using CartesianOperators: _pattern_runs!, _fill_runs!
using CartesianRuns
using SparseArrays
using Test

@testset "LinearStencil constructor" begin
    @test_throws ArgumentError LinearStencil{1}((0, 1), (1.0, -1.0))   # ascending
    @test_throws ArgumentError LinearStencil{1}((1, 1), (1.0, -1.0))   # equal
    @test_throws ArgumentError LinearStencil{0}((1, 0), (1.0, -1.0))   # D < 1
    @test_throws ArgumentError LinearStencil{-1}((1, 0), (1.0, -1.0))  # D < 1
    st = LinearStencil{1}((1, 0), (1.0, -1.0))
    @test st.offsets == (1, 0)
    @test st.coefs == (1.0, -1.0)
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
    @testset "full mask, offsets (1, 0), coefs (1.0, -1.0)" begin
        row = CartesianRunIndices(trues(5))
        col = CartesianRunIndices(trues(5))
        offsets = (1, 0); coefs = (1.0, -1.0)
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
        st = LinearStencil{1}((1, 0), (1.0, -1.0))
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "Float32" begin
            st32 = LinearStencil{1}((1, 0), (1f0, -1f0))
            row = CartesianRunIndices(trues(4)); col = CartesianRunIndices(trues(4))
            J = assemble(st32, row, col)
            update!(J, st32, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(0)), (1f0, -1f0), row, col)
            @test J == ref
            @test eltype(J) == Float32
        end
    end

    @testset "backward_x" begin
        st = LinearStencil{1}((0, -1), (1.0, -1.0))
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
    end

    @testset "central_x" begin
        st = LinearStencil{1}((1, -1), (1.0, -1.0))
        @testset "full mask 5" begin
            row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
        @testset "holed masks" begin
            row = CartesianRunIndices(Bool[1, 0, 1, 1])
            col = CartesianRunIndices(Bool[1, 1, 0, 1])
            J = assemble(st, row, col)
            update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(1), CartesianIndex(-1)), (1.0, -1.0), row, col)
            @test J == ref
        end
    end
end

@testset "D mismatch (1-D cri requires LinearStencil{1})" begin
    st_dim2 = LinearStencil{2}((1, 0), (1.0, -1.0))
    row = CartesianRunIndices(trues(5)); col = CartesianRunIndices(trues(5))
    @test_throws ArgumentError assemble(st_dim2, row, col)
    J = assemble(LinearStencil{1}((1, 0), (1.0, -1.0)), row, col)
    @test_throws ArgumentError update!(J, st_dim2, row, col)
end
