using CartesianOperators
using CartesianOperators: _x_diff_pattern_runs!, _x_diff_fill_runs!
using CartesianRuns
using SparseArrays
using Test

@testset "_x_diff_pattern_runs! (1-D)" begin
    @testset "full mask, offsets (1, 0)" begin
        row = CartesianRunIndices(trues(5))
        col = CartesianRunIndices(trues(5))
        offsets = (1, 0)
        colptr = Vector{Int}(undef, 6)
        colptr[1] = 1
        rowval = Int[]
        _x_diff_pattern_runs!(
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
        colptr = Vector{Int}(undef, 4)
        colptr[1] = 1
        rowval = Int[]
        _x_diff_pattern_runs!(
            rowval, colptr, offsets,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        # col 1 (mesh 1): Δ=1→r=0 miss; Δ=0→r=1 in row (compact 1). One entry.
        # col 2 (mesh 2): Δ=1→r=1 in row (compact 1); Δ=0→r=2 miss. One entry.
        # col 3 (mesh 4): Δ=1→r=3 in row (compact 2); Δ=0→r=4 in row (compact 3). Two entries.
        @test colptr == [1, 2, 3, 5]
        @test rowval == [1, 1, 2, 3]
    end
end

@testset "_x_diff_fill_runs! (1-D)" begin
    @testset "full mask, offsets (1, 0), coefs (1.0, -1.0)" begin
        row = CartesianRunIndices(trues(5))
        col = CartesianRunIndices(trues(5))
        offsets = (1, 0); coefs = (1.0, -1.0)
        # Pre-build pattern using the just-tested kernel
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        CartesianOperators._x_diff_pattern_runs!(
            rowval, colptr, offsets,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        nzval = Vector{Float64}(undef, length(rowval))
        CartesianOperators._x_diff_fill_runs!(
            nzval, colptr, offsets, coefs,
            row.intervals[1], 1, length(row.intervals[1]),
            col.intervals[1], 1, length(col.intervals[1]),
        )
        @test nzval == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    end
end
