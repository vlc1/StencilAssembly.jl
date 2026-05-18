using CartesianOperators
using CartesianOperators: _pattern!, _fill!
using FillArrays
using SparseArrays
using StaticArrays: SUnitRange
using Test

include("reference.jl")

@testset "LinearStencil constructor" begin
    # inner-constructor validation (well-typed inputs, invalid D)
    @test_throws ArgumentError LinearStencil{0}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))   # D < 1
    @test_throws ArgumentError LinearStencil{-1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))  # D < 1
    # outer-constructor friendly errors (ill-typed coefs)
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), (1.0, -1.0))                     # scalars, not arrays
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), (Fill(1.0, 5),))                 # length mismatch
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), (Fill(1f0, 5), Fill(-1.0, 5)))   # mixed eltype
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), (Fill(1.0, 5), Fill(-1.0, (5, 1))))  # mixed ndims
    # non-SUnitRange offsets get the migration error
    @test_throws ArgumentError LinearStencil{1}((0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
    @test_throws ArgumentError LinearStencil{1}(0:1, (Fill(-1.0, 5), Fill(1.0, 5)))
    # happy path: heterogeneous-container tuple typechecks
    st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), [1.0, 1.0, 1.0, 1.0, 1.0]))
    @test st.offsets === SUnitRange(0, 1)
    @test st.coefs[1] == Fill(-1.0, 5)
    @test st.coefs[2] == fill(1.0, 5)
end

@testset "_pattern! (1-D kernel)" begin
    @testset "row=col=1:5, SUnitRange(0, 1)" begin
        row = 1:5; col = 1:5
        offsets = SUnitRange(0, 1)
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        # Per column c: rows {c, c−1} ∩ 1:5 (descending δ ⇒ ascending row).
        # c=1: row 1 only (c−1=0 dropped). c=2..5: rows c−1, c. c=5: rows 4, 5.
        @test colptr == [1, 2, 4, 6, 8, 10]
        @test rowval == [1, 1, 2, 2, 3, 3, 4, 4, 5]
    end

    @testset "shifted: row=1:5, col=3:7, SUnitRange(0, 0)" begin
        row = 1:5; col = 3:7
        offsets = SUnitRange(0, 0)   # length 1, single offset 0
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        # c=3→r=3; c=4→r=4; c=5→r=5; c=6,7→out
        @test colptr == [1, 2, 3, 4, 4, 4]
        @test rowval == [3, 4, 5]
    end
end

@testset "_fill! (1-D kernel)" begin
    @testset "row=col=1:5, SUnitRange(0, 1), ascending coefs (-1.0, 1.0)" begin
        row = 1:5; col = 1:5
        offsets = SUnitRange(0, 1)
        coefs = (Fill(-1.0, 5), Fill(1.0, 5))    # coefs[1] for δ=0, coefs[2] for δ=1
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        nzval = Vector{Float64}(undef, length(rowval))
        _fill!(nzval, offsets, coefs, row, col)
        # Per column c, slots in row-ascending (= δ-descending) order.
        # c=1: only δ=0 active (δ=1 would land on row 0). Row 1, coef coefs[1][1]=−1.0.
        # c=2..5: row c−1 from δ=1 (coef coefs[2][c]=1.0), row c from δ=0 (coef coefs[1][c]=−1.0).
        @test nzval == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    end
end

@testset "assemble + update! 1-D" begin
    @testset "forward_x" begin
        @testset "row=col=(1:5,)" begin
            row = (1:5,); col = (1:5,)
            st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
        @testset "Float32, row=col=(1:4,)" begin
            row = (1:4,); col = (1:4,)
            st32 = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1f0, 4), Fill(1f0, 4)))
            J = assemble(st32, row, col); update!(J, st32, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1f0, 1f0), row, col)
            @test J == ref
            @test eltype(J) == Float32
        end
        @testset "unequal lengths: row=(1:5,), col=(1:3,)" begin
            row = (1:5,); col = (1:3,)
            st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 3), Fill(1.0, 3)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
        @testset "shifted ranges: row=(1:5,), col=(3:7,)" begin
            # coefs indexed at c ∈ col[1] = 3:7; Fill(.., 7) has axes 1:7, covers 3..7.
            row = (1:5,); col = (3:7,)
            st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 7), Fill(1.0, 7)))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
    end

    @testset "backward_x, row=col=(1:5,)" begin
        row = (1:5,); col = (1:5,)
        st = LinearStencil{1}(SUnitRange(-1, 0), (Fill(-1.0, 5), Fill(1.0, 5)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference((CartesianIndex(-1), CartesianIndex(0)), (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "central_x (with explicit zero middle), row=col=(1:5,)" begin
        # Contiguous offsets force the middle offset 0 to be represented; its
        # zero coef pollutes the sparsity pattern (matrix carries an explicit
        # zero entry at offset 0). The brute-force oracle does the same, so
        # the test still cross-checks structurally.
        row = (1:5,); col = (1:5,)
        st = LinearStencil{1}(SUnitRange(-1, 1),
            (Fill(-1.0, 5), Fill(0.0, 5), Fill(1.0, 5)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)),
            (-1.0, 0.0, 1.0), row, col)
        @test J == ref
    end

    @testset "variable coefs: density-weighted gradient" begin
        # ψ[i] = (φ[i] − φ[i−1]) / ρ[i] on row = col = (1:4,).
        # Offsets SUnitRange(-1, 0); coefs ascending.
        #   δ=-1: matrix entry (r=c+1, c) → coef −1/ρ[c+1] → coefs[1] is shifted -1/ρ
        #   δ=0:  matrix entry (r=c, c)   → coef +1/ρ[c]   → coefs[2] = 1 ./ ρ
        ρ = [2.0, 3.0, 5.0, 7.0]
        coefm1 = vcat(-1 ./ ρ[2:end], 0.0)       # length 4; index 4 never read (c=4, δ=-1 → r=5 ∉ row)
        coef0  = 1 ./ ρ                          # length 4, indexed by c
        row = (1:4,); col = (1:4,)
        st = LinearStencil{1}(SUnitRange(-1, 0), (coefm1, coef0))
        J = assemble(st, row, col); update!(J, st, row, col)
        I_exp = [1, 2, 2, 3, 3, 4, 4]
        J_exp = [1, 1, 2, 2, 3, 3, 4]
        V_exp = [1/ρ[1], -1/ρ[2], 1/ρ[2], -1/ρ[3], 1/ρ[3], -1/ρ[4], 1/ρ[4]]
        ref = sparse(I_exp, J_exp, V_exp, 4, 4)
        @test J == ref
    end
end

@testset "edge cases (three-phase kernel)" begin
    @testset "L=0: empty stencil (kernel-only)" begin
        # LinearStencil with L=0 cannot be constructed without unbound T/N;
        # exercise the kernels directly with SUnitRange{0,0}().
        row = 1:5; col = 1:5
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, SUnitRange{0, 0}(), row, col)
        @test colptr == [1, 1, 1, 1, 1, 1]
        @test rowval == Int[]
        nzval = Float64[]
        _fill!(nzval, SUnitRange{0, 0}(), (), row, col)
        @test nzval == Float64[]
    end

    @testset "all offsets out of range: row=col=(1:3,), SUnitRange(3, 4)" begin
        # cmax−rmin = 2, so offsets 3 and 4 are both trimmed → empty operator.
        row = (1:3,); col = (1:3,)
        st = LinearStencil{1}(SUnitRange(3, 4), (Fill(1.0, 3), Fill(2.0, 3)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(3), CartesianIndex(4)), (1.0, 2.0), row, col)
        @test J == ref
        @test nnz(J) == 0
    end

    @testset "partial out of range: row=col=(1:3,), SUnitRange(-3, 0)" begin
        # cmin−rmax = −2, so offset −3 is trimmed (prefix); offsets −2, −1, 0 survive.
        # L=4, length(row)=3, L−1=3 ≤ 3 ✓ (exact boundary).
        row = (1:3,); col = (1:3,)
        st = LinearStencil{1}(SUnitRange(-3, 0),
            (Fill(1.0, 3), Fill(2.0, 3), Fill(3.0, 3), Fill(4.0, 3)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-3), CartesianIndex(-2), CartesianIndex(-1), CartesianIndex(0)),
            (1.0, 2.0, 3.0, 4.0), row, col)
        @test J == ref
    end

    @testset "disjoint row/col, bridging offsets: row=(1:5,), col=(100:105,)" begin
        row = (1:5,); col = (100:105,)
        st = LinearStencil{1}(SUnitRange(99, 100), (Fill(-1.0, 105), Fill(1.0, 105)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(99), CartesianIndex(100)), (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "right ramp at upper col boundary: row=col=(1:10,), SUnitRange(4, 5)" begin
        # Tests the right-ramp formula at the col=cmax end (the case the old
        # segment-walk kernel called "coincident events at cmax+1").
        row = (1:10,); col = (1:10,)
        st = LinearStencil{1}(SUnitRange(4, 5), (Fill(-1.0, 10), Fill(1.0, 10)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(4), CartesianIndex(5)), (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "L = length(row) + 1 boundary: empty interior, ramps tile" begin
        # row=col=(1:3,), L=4 → m+1=4, c_LR = c_RR + 1, interior empty.
        # Offsets -1..2, all reach valid rows somewhere; verify via oracle.
        row = (1:3,); col = (1:5,)
        st = LinearStencil{1}(SUnitRange(-1, 2),
            (Fill(1.0, 5), Fill(2.0, 5), Fill(3.0, 5), Fill(4.0, 5)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1), CartesianIndex(2)),
            (1.0, 2.0, 3.0, 4.0), row, col)
        @test J == ref
    end

    @testset "off-mesh column tails: row=(1:5,), col=(1:100,), SUnitRange(-1, 1)" begin
        # Columns 7..100 sit beyond rmax + δ_hi = 5 + 1 = 6, so they contribute
        # no entries; the right-ramp max(0, active) clip must keep cur consistent.
        row = (1:5,); col = (1:100,)
        st = LinearStencil{1}(SUnitRange(-1, 1),
            (Fill(-1.0, 100), Fill(0.0, 100), Fill(1.0, 100)))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)),
            (-1.0, 0.0, 1.0), row, col)
        @test J == ref
    end
end

@testset "L > length(row) + 1 guard" begin
    # L=5, length(row)=3 → L−1=4 > 3 ⇒ rejected by assemble and update!.
    st = LinearStencil{1}(SUnitRange(-2, 2),
        (Fill(1.0, 3), Fill(2.0, 3), Fill(3.0, 3), Fill(4.0, 3), Fill(5.0, 3)))
    row = (1:3,); col = (1:3,)
    @test_throws ArgumentError assemble(st, row, col)
    # update! gets the same guard. Build a matching matrix from a smaller stencil
    # to call update! against (assemble would otherwise fail before update!).
    st_small = LinearStencil{1}(SUnitRange(-1, 1),
        (Fill(1.0, 3), Fill(2.0, 3), Fill(3.0, 3)))
    J = assemble(st_small, row, col)
    @test_throws ArgumentError update!(J, st, row, col)
end

@testset "D ≤ N constructor invariant" begin
    # D > N is rejected at construction.
    @test_throws ArgumentError LinearStencil{2}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
    @test_throws ArgumentError LinearStencil{3}(SUnitRange(0, 1), (Fill(-1.0, (5, 3)), Fill(1.0, (5, 3))))
end

@testset "N mismatch (1-D row/col requires LinearStencil with N=1)" begin
    # Constructor succeeds (D=1 ≤ N=2) but 1-D assemble/update! pin N=1 → MethodError.
    st_n2 = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, (5, 3)), Fill(1.0, (5, 3))))
    row = (1:5,); col = (1:5,)
    @test_throws MethodError assemble(st_n2, row, col)
    J = assemble(LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5))), row, col)
    @test_throws MethodError update!(J, st_n2, row, col)
end

@testset "build convenience" begin
    # build(...) == update!(assemble(...), ...)
    row = (1:5,); col = (1:5,)
    st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 5), Fill(1.0, 5)))
    J_two_step = update!(assemble(st, row, col), st, row, col)
    J_build    = build(st, row, col)
    @test J_build == J_two_step
end

@testset "assemble + update! 2-D" begin
    @testset "2-D D=1, row=col=(1:3, 1:4)" begin
        row = (1:3, 1:4); col = (1:3, 1:4)
        st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 3, 4), Fill(1.0, 3, 4)))
        J = build(st, row, col)
        # Offsets for D=1: (CartesianIndex(0,0), CartesianIndex(1,0))
        offsets = (CartesianIndex(0, 0), CartesianIndex(1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "2-D D=2, row=col=(1:3, 1:4)" begin
        row = (1:3, 1:4); col = (1:3, 1:4)
        st = LinearStencil{2}(SUnitRange(0, 1), (Fill(-1.0, 3, 4), Fill(1.0, 3, 4)))
        J = build(st, row, col)
        # Offsets for D=2: (CartesianIndex(0,0), CartesianIndex(0,1))
        offsets = (CartesianIndex(0, 0), CartesianIndex(0, 1))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end
end

@testset "assemble + update! 3-D" begin
    @testset "3-D D=1, row=col=(1:3, 1:4, 1:5)" begin
        row = (1:3, 1:4, 1:5); col = (1:3, 1:4, 1:5)
        st = LinearStencil{1}(SUnitRange(0, 1), (Fill(-1.0, 3, 4, 5), Fill(1.0, 3, 4, 5)))
        J = build(st, row, col)
        # Offsets for D=1: (CartesianIndex(0,0,0), CartesianIndex(1,0,0))
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(1, 0, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "3-D D=2, row=col=(1:3, 1:4, 1:5)" begin
        row = (1:3, 1:4, 1:5); col = (1:3, 1:4, 1:5)
        st = LinearStencil{2}(SUnitRange(0, 1), (Fill(-1.0, 3, 4, 5), Fill(1.0, 3, 4, 5)))
        J = build(st, row, col)
        # Offsets for D=2: (CartesianIndex(0,0,0), CartesianIndex(0,1,0))
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(0, 1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "3-D D=3, row=col=(1:3, 1:4, 1:5)" begin
        row = (1:3, 1:4, 1:5); col = (1:3, 1:4, 1:5)
        st = LinearStencil{3}(SUnitRange(0, 1), (Fill(-1.0, 3, 4, 5), Fill(1.0, 3, 4, 5)))
        J = build(st, row, col)
        # Offsets for D=3: (CartesianIndex(0,0,0), CartesianIndex(0,0,1))
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 1))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end
end
