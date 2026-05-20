using CartesianOperators
using CartesianOperators: _pattern!, _fill!
using FillArrays
using SparseArrays
using StaticArrays: SUnitRange, SVector
using Test

# reference.jl is included by runtests.jl before this file runs.

@testset "LinearStencil constructor" begin
    # inner-constructor validation (well-typed inputs, invalid D)
    @test_throws ArgumentError LinearStencil{0}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))   # D < 1
    @test_throws ArgumentError LinearStencil{-1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))  # D < 1
    # outer-constructor friendly errors (ill-typed term)
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), 1.0)                            # not an AbstractArray
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), [1.0, 2.0, 3.0])                # eltype not SVector
    @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(1.0, 2.0, 3.0), 5))# SVector length ≠ L
    # non-SUnitRange offsets get the migration error
    @test_throws ArgumentError LinearStencil{1}((0, 1), Fill(SVector(-1.0, 1.0), 5))
    @test_throws ArgumentError LinearStencil{1}(0:1, Fill(SVector(-1.0, 1.0), 5))
    # happy path: array of SVector typechecks
    st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
    @test st.offsets === SUnitRange(0, 1)
    @test st.term == Fill(SVector(-1.0, 1.0), 5)
    @test st.term[1] === SVector(-1.0, 1.0)
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
    @testset "row=col=1:5, SUnitRange(0, 1), ascending term (-1.0, 1.0)" begin
        row = 1:5; col = 1:5
        offsets = SUnitRange(0, 1)
        term = Fill(SVector(-1.0, 1.0), 5)    # element[1] for δ=0, element[2] for δ=1
        colptr = Vector{Int}(undef, 6); colptr[1] = 1
        rowval = Int[]
        _pattern!(rowval, colptr, offsets, row, col)
        nzval = Vector{Float64}(undef, length(rowval))
        _fill!(nzval, offsets, term, row, col)
        # Per column c, slots in row-ascending (= δ-descending) order.
        # c=1: only δ=0 active (δ=1 would land on row 0). Row 1, coef term[1][1]=−1.0.
        # c=2..5: row c−1 from δ=1 (coef term[c][2]=1.0), row c from δ=0 (coef term[c][1]=−1.0).
        @test nzval == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    end
end

@testset "assemble + update! 1-D" begin
    @testset "forward_x" begin
        @testset "row=col=(1:5,)" begin
            row = (1:5,); col = (1:5,)
            st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
        @testset "Float32, row=col=(1:4,)" begin
            row = (1:4,); col = (1:4,)
            st32 = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1f0, 1f0), 4))
            J = assemble(st32, row, col); update!(J, st32, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1f0, 1f0), row, col)
            @test J == ref
            @test eltype(J) == Float32
        end
        @testset "unequal lengths: row=(1:5,), col=(1:3,)" begin
            row = (1:5,); col = (1:3,)
            st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
        @testset "shifted ranges: row=(1:5,), col=(3:7,)" begin
            # term indexed at c ∈ col[1] = 3:7; Fill(.., 7) has axes 1:7, covers 3..7.
            row = (1:5,); col = (3:7,)
            st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 7))
            J = assemble(st, row, col); update!(J, st, row, col)
            ref = stencil_reference((CartesianIndex(0), CartesianIndex(1)), (-1.0, 1.0), row, col)
            @test J == ref
        end
    end

    @testset "backward_x, row=col=(1:5,)" begin
        row = (1:5,); col = (1:5,)
        st = LinearStencil{1}(SUnitRange(-1, 0), Fill(SVector(-1.0, 1.0), 5))
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
        st = LinearStencil{1}(SUnitRange(-1, 1), Fill(SVector(-1.0, 0.0, 1.0), 5))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)),
            (-1.0, 0.0, 1.0), row, col)
        @test J == ref
    end

    @testset "variable coefficients: density-weighted gradient" begin
        # ψ[i] = (φ[i] − φ[i−1]) / ρ[i] on row = col = (1:4,).
        # Offsets SUnitRange(-1, 0); each column's SVector is (coef@δ=-1, coef@δ=0).
        #   δ=-1: matrix entry (r=c+1, c) → coef −1/ρ[c+1] → element[1] is shifted -1/ρ
        #   δ=0:  matrix entry (r=c, c)   → coef +1/ρ[c]   → element[2] = 1 ./ ρ
        ρ = [2.0, 3.0, 5.0, 7.0]
        coefm1 = vcat(-1 ./ ρ[2:end], 0.0)       # length 4; index 4 never read (c=4, δ=-1 → r=5 ∉ row)
        coef0  = 1 ./ ρ                          # length 4, indexed by c
        term   = SVector.(coefm1, coef0)         # Vector{SVector{2,Float64}}
        row = (1:4,); col = (1:4,)
        st = LinearStencil{1}(SUnitRange(-1, 0), term)
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
        _fill!(nzval, SUnitRange{0, 0}(), Fill(SVector{0, Float64}(), 5), row, col)
        @test nzval == Float64[]
    end

    @testset "all offsets out of range: row=col=(1:3,), SUnitRange(3, 4)" begin
        # cmax−rmin = 2, so offsets 3 and 4 are both trimmed → empty operator.
        row = (1:3,); col = (1:3,)
        st = LinearStencil{1}(SUnitRange(3, 4), Fill(SVector(1.0, 2.0), 3))
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
        st = LinearStencil{1}(SUnitRange(-3, 0), Fill(SVector(1.0, 2.0, 3.0, 4.0), 3))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-3), CartesianIndex(-2), CartesianIndex(-1), CartesianIndex(0)),
            (1.0, 2.0, 3.0, 4.0), row, col)
        @test J == ref
    end

    @testset "disjoint row/col, bridging offsets: row=(1:5,), col=(100:105,)" begin
        row = (1:5,); col = (100:105,)
        st = LinearStencil{1}(SUnitRange(99, 100), Fill(SVector(-1.0, 1.0), 105))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(99), CartesianIndex(100)), (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "right ramp at upper col boundary: row=col=(1:10,), SUnitRange(4, 5)" begin
        # Tests the right-ramp formula at the col=cmax end (the case the old
        # segment-walk kernel called "coincident events at cmax+1").
        row = (1:10,); col = (1:10,)
        st = LinearStencil{1}(SUnitRange(4, 5), Fill(SVector(-1.0, 1.0), 10))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(4), CartesianIndex(5)), (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "L = length(row) + 1 boundary: empty interior, ramps tile" begin
        # row=col=(1:3,), L=4 → m+1=4, c_LR = c_RR + 1, interior empty.
        # Offsets -1..2, all reach valid rows somewhere; verify via oracle.
        row = (1:3,); col = (1:5,)
        st = LinearStencil{1}(SUnitRange(-1, 2), Fill(SVector(1.0, 2.0, 3.0, 4.0), 5))
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
        st = LinearStencil{1}(SUnitRange(-1, 1), Fill(SVector(-1.0, 0.0, 1.0), 100))
        J = assemble(st, row, col); update!(J, st, row, col)
        ref = stencil_reference(
            (CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)),
            (-1.0, 0.0, 1.0), row, col)
        @test J == ref
    end
end

@testset "L > length(row) + 1 guard" begin
    # L=5, length(row)=3 → L−1=4 > 3 ⇒ rejected by assemble and update!.
    st = LinearStencil{1}(SUnitRange(-2, 2), Fill(SVector(1.0, 2.0, 3.0, 4.0, 5.0), 3))
    row = (1:3,); col = (1:3,)
    @test_throws ArgumentError assemble(st, row, col)
    # update! gets the same guard. Build a matching matrix from a smaller stencil
    # to call update! against (assemble would otherwise fail before update!).
    st_small = LinearStencil{1}(SUnitRange(-1, 1), Fill(SVector(1.0, 2.0, 3.0), 3))
    J = assemble(st_small, row, col)
    @test_throws ArgumentError update!(J, st, row, col)
end

@testset "D ≤ N constructor invariant" begin
    # D > N is rejected at construction.
    @test_throws ArgumentError LinearStencil{2}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
    @test_throws ArgumentError LinearStencil{3}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5, 3))
end

@testset "N mismatch (1-D row/col requires LinearStencil with N=1)" begin
    # Constructor succeeds (D=1 ≤ N=2) but 1-D assemble/update! pin N=1 → MethodError.
    st_n2 = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5, 3))
    row = (1:5,); col = (1:5,)
    @test_throws MethodError assemble(st_n2, row, col)
    J = assemble(LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5)), row, col)
    @test_throws MethodError update!(J, st_n2, row, col)
end

@testset "build convenience" begin
    # build(...) == update!(assemble(...), ...)
    row = (1:5,); col = (1:5,)
    st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
    J_two_step = update!(assemble(st, row, col), st, row, col)
    J_build    = build(st, row, col)
    @test J_build == J_two_step
end

@testset "assemble + update! 2-D" begin
    @testset "2-D D=1, row=col=(1:3, 1:4)" begin
        row = (1:3, 1:4); col = (1:3, 1:4)
        st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3, 4))
        J = build(st, row, col)
        # Offsets for D=1: (CartesianIndex(0,0), CartesianIndex(1,0))
        offsets = (CartesianIndex(0, 0), CartesianIndex(1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "2-D D=2, row=col=(1:3, 1:4)" begin
        row = (1:3, 1:4); col = (1:3, 1:4)
        st = LinearStencil{2}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3, 4))
        J = build(st, row, col)
        # Offsets for D=2: (CartesianIndex(0,0), CartesianIndex(0,1))
        offsets = (CartesianIndex(0, 0), CartesianIndex(0, 1))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "2-D D=1, row=(2:4,1:4) col=(1:3,1:4) (shifted stencil dim)" begin
        row = (2:4, 1:4); col = (1:3, 1:4)
        st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3, 4))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0), CartesianIndex(1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "2-D D=2, row=(1:3,2:5) col=(1:3,1:4) (shifted stencil dim)" begin
        row = (1:3, 2:5); col = (1:3, 1:4)
        st = LinearStencil{2}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3, 4))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0), CartesianIndex(0, 1))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "2-D D=1, row=(1:3,2:4) col=(1:3,1:4) (shifted non-stencil dim)" begin
        row = (1:3, 2:4); col = (1:3, 1:4)
        st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 3, 4))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0), CartesianIndex(1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end
end

@testset "assemble + update! 3-D" begin
    @testset "3-D D=1, row=col=(1:2,1:2,1:3)" begin
        row = (1:2, 1:2, 1:3); col = (1:2, 1:2, 1:3)
        st = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 2, 2, 3))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(1, 0, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "3-D D=2, row=col=(1:2,1:2,1:3)" begin
        row = (1:2, 1:2, 1:3); col = (1:2, 1:2, 1:3)
        st = LinearStencil{2}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 2, 2, 3))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(0, 1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "3-D D=3, row=col=(1:2,1:2,1:3)" begin
        row = (1:2, 1:2, 1:3); col = (1:2, 1:2, 1:3)
        st = LinearStencil{3}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 2, 2, 3))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(0, 0, 1))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end

    @testset "3-D D=2, shifted stencil dim, row=(1:2,2:4,1:3) col=(1:2,1:3,1:3)" begin
        row = (1:2, 2:4, 1:3); col = (1:2, 1:3, 1:3)
        st = LinearStencil{2}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 2, 3, 3))
        J = build(st, row, col)
        offsets = (CartesianIndex(0, 0, 0), CartesianIndex(0, 1, 0))
        ref = stencil_reference(offsets, (-1.0, 1.0), row, col)
        @test J == ref
    end
end

@testset "StarStencil constructor" begin
    # L invalid
    @test_throws ArgumentError StarStencil{0}((Fill(SVector(1.0, 1.0, 1.0), 5),))
    @test_throws ArgumentError StarStencil{-1}((Fill(SVector(1.0, 1.0, 1.0), 5),))
    @test_throws ArgumentError StarStencil{1.0}((Fill(SVector(1.0, 1.0, 1.0), 5),))
    # M ≠ 2L+1 (wrong SVector length)
    @test_throws ArgumentError StarStencil{1}((Fill(SVector(1.0, 1.0), 5),))                  # M=2
    @test_throws ArgumentError StarStencil{1}((Fill(SVector(1.0, 1.0, 1.0, 1.0), 5),))        # M=4
    # Non-tuple terms
    @test_throws ArgumentError StarStencil{1}(Fill(SVector(1.0, 1.0, 1.0), 5))
    # Outer-constructor friendly errors
    @test_throws ArgumentError StarStencil{1}(((1.0, 2.0, 3.0),))                             # per-axis not an array
    @test_throws ArgumentError StarStencil{1}(([1.0, 2.0, 3.0],))                             # eltype not SVector
    @test_throws ArgumentError StarStencil{1}((Fill(SVector(1.0, 1.0, 1.0), 5),               # mixed scalar eltype
                                              Fill(SVector(1f0, 1f0, 1f0), 5)))
    @test_throws ArgumentError StarStencil{1}((Fill(SVector(1.0, 1.0, 1.0), 5),               # ndims ≠ N (1-D arrays, N=2)
                                              Fill(SVector(1.0, 1.0, 1.0), 5)))
    # Happy path: store stays as given
    st = StarStencil{1}((Fill(SVector(-1.0, 2.0, -1.0), 5),))
    @test st.terms[1][1] == SVector(-1.0, 2.0, -1.0)
end

@testset "StarStencil 1-D (delegates to LinearStencil)" begin
    @testset "Laplacian shape, row=col=(1:6,)" begin
        row = (1:6,); col = (1:6,)
        n = 6
        terms = (Fill(SVector(-1.0, 2.0, -1.0), n),)
        st = StarStencil{1}(terms)
        ln = LinearStencil{1}(SUnitRange(-1, 1), terms[1])
        @test build(st, row, col) == build(ln, row, col)
    end

    @testset "asymmetric values, row=col=(1:5,)" begin
        # Asymmetric values to confirm the kernel doesn't assume value-symmetry.
        n = 5
        terms = (Fill(SVector(0.3, 1.7, -2.1), n),)
        st = StarStencil{1}(terms)
        ln = LinearStencil{1}(SUnitRange(-1, 1), terms[1])
        @test build(st, (1:n,), (1:n,)) == build(ln, (1:n,), (1:n,))
    end

    @testset "L=2, asymmetric values, row=col=(1:6,)" begin
        n = 6
        terms = (Fill(SVector(0.1, 0.2, 0.3, 0.4, 0.5), n),)
        st = StarStencil{2}(terms)
        ln = LinearStencil{1}(SUnitRange(-2, 2), terms[1])
        @test build(st, (1:n,), (1:n,)) == build(ln, (1:n,), (1:n,))
    end
end

# Decomposition oracle: a StarStencil equals the sum of N LinearStencils,
# one per axis, with the same offsets and per-axis terms.
function _star_decomposition_oracle(st::StarStencil{L, N, M}, row, col) where {L, N, M}
    Σ = build(LinearStencil{1}(SUnitRange(-L, L), st.terms[1]), row, col)
    for d in 2:N
        Σ = Σ + build(LinearStencil{d}(SUnitRange(-L, L), st.terms[d]), row, col)
    end
    return Σ
end

@testset "StarStencil 2-D vs sum(LinearStencils)" begin
    @testset "Laplacian, row=col=(1:5,1:4)" begin
        row = (1:5, 1:4); col = (1:5, 1:4)
        n1 = length(row[1]); n2 = length(row[2])
        terms = (
            Fill(SVector(-1.0, 2.0, -1.0), n1, n2),
            Fill(SVector(-1.0, 2.0, -1.0), n1, n2),
        )
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "asymmetric values, row=col=(1:6,1:5)" begin
        row = (1:6, 1:5); col = (1:6, 1:5)
        n1, n2 = 6, 5
        terms = (
            Fill(SVector(0.1, 1.7, -2.3), n1, n2),
            Fill(SVector(-0.5, 3.1, 0.8), n1, n2),
        )
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "L=2, row=col=(1:4,1:4)" begin
        row = (1:4, 1:4); col = (1:4, 1:4)
        n1, n2 = 4, 4
        ax_terms() = Fill(SVector(0.1, 0.2, 0.3, 0.4, 0.5), n1, n2)
        terms = (ax_terms(), ax_terms())
        st = StarStencil{2}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "Float32, row=col=(1:4,1:4)" begin
        row = (1:4, 1:4); col = (1:4, 1:4)
        terms = (
            Fill(SVector(-1f0, 2f0, -1f0), 4, 4),
            Fill(SVector(-1f0, 2f0, -1f0), 4, 4),
        )
        st = StarStencil{1}(terms)
        J = build(st, row, col)
        @test eltype(J) == Float32
        @test J == _star_decomposition_oracle(st, row, col)
    end

    @testset "col ⊋ row in axis 1 (exercises axis-1-only branch), row=(1:5,1:4), col=(0:6,1:4)" begin
        row = (1:5, 1:4); col = (0:6, 1:4)
        # Terms must cover col[1] = 0:6 — wrap with OffsetArrays-style hand-rolled axes.
        # Easiest: use Fill with enough cells indexed by absolute mesh position via 0:6 axis.
        # Fill is shape-based so its axes are 1-based; indexing at c=0 would fail.
        # Simpler: shift to (1:7, 1:4) for row to keep col[1] indices valid.
        # Re-pick ranges so col[1] still extends beyond row[1] but stays positive.
        row = (2:6, 1:4); col = (1:7, 1:4)
        n_col1, n_col2 = 7, 4
        terms = (
            Fill(SVector(-1.0, 2.0, -1.0), n_col1, n_col2),
            Fill(SVector(-1.0, 2.0, -1.0), n_col1, n_col2),
        )
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "col ⊋ row in axis 2 (exercises axis-2-only branch), row=(1:5,2:4), col=(1:5,1:5)" begin
        row = (1:5, 2:4); col = (1:5, 1:5)
        terms = (
            Fill(SVector(-1.0, 2.0, -1.0), 5, 5),
            Fill(SVector(-1.0, 2.0, -1.0), 5, 5),
        )
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end
end

@testset "StarStencil 3-D vs sum(LinearStencils)" begin
    @testset "Laplacian, row=col=(1:3,1:3,1:3)" begin
        row = (1:3, 1:3, 1:3); col = (1:3, 1:3, 1:3)
        terms = ntuple(_ -> Fill(SVector(-1.0, 2.0, -1.0), 3, 3, 3), 3)
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "asymmetric values, row=col=(1:3,1:3,1:3)" begin
        row = (1:3, 1:3, 1:3); col = (1:3, 1:3, 1:3)
        terms = (
            Fill(SVector(0.1, 1.0, -2.0), 3, 3, 3),
            Fill(SVector(-0.5, 2.5, 0.8), 3, 3, 3),
            Fill(SVector(0.3, -1.4, 0.6), 3, 3, 3),
        )
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "unequal lengths, row=col=(1:4,1:3,1:2)" begin
        row = (1:4, 1:3, 1:2); col = (1:4, 1:3, 1:2)
        terms = ntuple(_ -> Fill(SVector(-1.0, 2.0, -1.0), 4, 3, 2), 3)
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end

    @testset "col ⊋ row in axis 3, row=(1:3,1:3,2:4), col=(1:3,1:3,1:5)" begin
        row = (1:3, 1:3, 2:4); col = (1:3, 1:3, 1:5)
        terms = ntuple(_ -> Fill(SVector(-1.0, 2.0, -1.0), 3, 3, 5), 3)
        st = StarStencil{1}(terms)
        @test build(st, row, col) == _star_decomposition_oracle(st, row, col)
    end
end

@testset "StarStencil 2L > length(row[d]) guard" begin
    # L=2, length(row[1]) = 3 → 2L=4 > 3, rejected.
    terms = (Fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0), 3),)
    st = StarStencil{2}(terms)
    @test_throws ArgumentError build(st, (1:3,), (1:3,))

    # 2-D: violated in axis 2 only.
    terms2 = (
        Fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0), 5, 3),
        Fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0), 5, 3),
    )
    st2 = StarStencil{2}(terms2)
    @test_throws ArgumentError build(st2, (1:5, 1:3), (1:5, 1:3))  # axis 2: 2L=4 > 3
end

@testset "StarStencil build = update!(assemble(...), ...)" begin
    row = (1:4, 1:4); col = (1:4, 1:4)
    terms = (
        Fill(SVector(-1.0, 2.0, -1.0), 4, 4),
        Fill(SVector(-1.0, 2.0, -1.0), 4, 4),
    )
    st = StarStencil{1}(terms)
    J_two_step = update!(assemble(st, row, col), st, row, col)
    J_build    = build(st, row, col)
    @test J_two_step == J_build
end

@testset "AccessStyle trait + AbstractStencil supertype" begin
    @testset "default ctor → ColumnAccess" begin
        st_lin = LinearStencil{1}(SUnitRange(0, 1), Fill(SVector(-1.0, 1.0), 5))
        @test AccessStyle(st_lin) === ColumnAccess()
        @test AccessStyle(typeof(st_lin)) === ColumnAccess()
        @test st_lin isa AbstractStencil
        @test st_lin isa AbstractStencil{ColumnAccess}

        terms = (Fill(SVector(-1.0, 2.0, -1.0), 5),)
        st_star = StarStencil{1}(terms)
        @test AccessStyle(st_star) === ColumnAccess()
        @test AccessStyle(typeof(st_star)) === ColumnAccess()
        @test st_star isa AbstractStencil
        @test st_star isa AbstractStencil{ColumnAccess}
    end

    @testset "explicit ColumnAccess ctor" begin
        st_lin = LinearStencil{1}(ColumnAccess, SUnitRange(0, 1),
                                  Fill(SVector(-1.0, 1.0), 5))
        @test AccessStyle(st_lin) === ColumnAccess()
        terms = (Fill(SVector(-1.0, 2.0, -1.0), 5),)
        st_star = StarStencil{1}(ColumnAccess, terms)
        @test AccessStyle(st_star) === ColumnAccess()
    end

    @testset "explicit RowAccess ctor constructs but is unassemblable on CSC" begin
        st_lin = LinearStencil{1}(RowAccess, SUnitRange(0, 1),
                                  Fill(SVector(-1.0, 1.0), 5))
        @test AccessStyle(st_lin) === RowAccess()
        @test st_lin isa AbstractStencil{RowAccess}
        @test_throws MethodError assemble(st_lin, (1:5,), (1:5,))
        @test_throws MethodError build(st_lin, (1:5,), (1:5,))

        terms = (Fill(SVector(-1.0, 2.0, -1.0), 5),)
        st_star = StarStencil{1}(RowAccess, terms)
        @test AccessStyle(st_star) === RowAccess()
        @test st_star isa AbstractStencil{RowAccess}
        @test_throws MethodError assemble(st_star, (1:5,), (1:5,))
        @test_throws MethodError build(st_star, (1:5,), (1:5,))
    end

    @testset "_as_linear propagates S (RowAccess)" begin
        terms = (Fill(SVector(-1.0, 2.0, -1.0), 5),)
        st_star = StarStencil{1}(RowAccess, terms)
        ln = CartesianOperators._as_linear(st_star)
        @test ln isa LinearStencil
        @test AccessStyle(ln) === RowAccess()
    end

    @testset "RowAccess 2-D StarStencil is unassemblable on CSC" begin
        n1, n2 = 5, 4
        terms = (
            Fill(SVector(-1.0, 2.0, -1.0), n1, n2),
            Fill(SVector(-1.0, 2.0, -1.0), n1, n2),
        )
        st = StarStencil{1}(RowAccess, terms)
        @test_throws MethodError assemble(st, (1:n1, 1:n2), (1:n1, 1:n2))
    end
end
