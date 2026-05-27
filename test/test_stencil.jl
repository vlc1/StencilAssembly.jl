using StencilAssembly
using StencilAssembly: _pattern!, _fill!
using StencilCore: ô, ê₁, ê₂
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
    @test_throws ArgumentError StarStencil{0}(Fill(SVector(1.0, 1.0, 1.0), 5))
    @test_throws ArgumentError StarStencil{-1}(Fill(SVector(1.0, 1.0, 1.0), 5))
    @test_throws ArgumentError StarStencil{1.0}(Fill(SVector(1.0, 1.0, 1.0), 5))
    # M not of the form 2NL+1
    @test_throws ArgumentError StarStencil{1}(Fill(SVector(1.0, 1.0), 5))               # M=2
    @test_throws ArgumentError StarStencil{1}(Fill(SVector(1.0, 1.0, 1.0, 1.0), 5))     # M=4
    # ndims ≠ (M-1)/(2L): M=5 ⇒ N=2, but a 1-D array
    @test_throws ArgumentError StarStencil{1}(Fill(SVector(1.0, 1.0, 1.0, 1.0, 1.0), 5))
    # coefficient not an Array/Term of SVector
    @test_throws ArgumentError StarStencil{1}([1.0, 2.0, 3.0])                          # eltype not SVector
    @test_throws ArgumentError StarStencil{1}((1.0, 2.0, 3.0))                          # not array/term
    # Happy path (N=1, M=3): store stays as given
    st = StarStencil{1}(Fill(SVector(-1.0, 2.0, -1.0), 5))
    @test st.term[1] == SVector(-1.0, 2.0, -1.0)
end

@testset "StarStencil 1-D (delegates to LinearStencil)" begin
    @testset "Laplacian shape, row=col=(1:6,)" begin
        row = (1:6,); col = (1:6,)
        n = 6
        term = Fill(SVector(-1.0, 2.0, -1.0), n)
        st = StarStencil{1}(term)
        ln = LinearStencil{1}(SUnitRange(-1, 1), term)
        @test build(st, row, col) == build(ln, row, col)
    end

    @testset "asymmetric values, row=col=(1:5,)" begin
        # Asymmetric values to confirm the kernel doesn't assume value-symmetry.
        n = 5
        term = Fill(SVector(0.3, 1.7, -2.1), n)
        st = StarStencil{1}(term)
        ln = LinearStencil{1}(SUnitRange(-1, 1), term)
        @test build(st, (1:n,), (1:n,)) == build(ln, (1:n,), (1:n,))
    end

    @testset "L=2, asymmetric values, row=col=(1:6,)" begin
        n = 6
        term = Fill(SVector(0.1, 0.2, 0.3, 0.4, 0.5), n)
        st = StarStencil{2}(term)
        ln = LinearStencil{1}(SUnitRange(-2, 2), term)
        @test build(st, (1:n,), (1:n,)) == build(ln, (1:n,), (1:n,))
    end
end

# Build the interlaced whole-star SVector{2NL+1} from per-axis offset
# coefficients (each an SVector{2L+1} in offset order δ = -L..L). The diagonal
# slot is the SUM of the per-axis centers, so the resulting StarStencil equals
# the sum of the per-axis LinearStencils (the oracle below).
function _interlace(cs::NTuple{N, SVector{Mx, T}}, ::Val{L}) where {N, Mx, T, L}
    vals = T[]
    for d in N:-1:1, o in -L:-1
        push!(vals, cs[d][o + L + 1])
    end
    push!(vals, sum(cs[d][L + 1] for d in 1:N))
    for d in 1:N, o in 1:L
        push!(vals, cs[d][o + L + 1])
    end
    SVector{2N * L + 1, T}(vals...)
end

# Oracle: a StarStencil equals the sum of N axis-aligned LinearStencils
# (offsets -L..L, the per-axis coefficient arrays).
function _star_oracle(cs::NTuple{N}, ::Val{L}, row, col) where {N, L}
    Σ = build(LinearStencil{1}(SUnitRange(-L, L), cs[1]), row, col)
    for d in 2:N
        Σ = Σ + build(LinearStencil{d}(SUnitRange(-L, L), cs[d]), row, col)
    end
    return Σ
end

@testset "StarStencil 2-D vs sum(LinearStencils)" begin
    @testset "Laplacian, row=col=(1:5,1:4)" begin
        row = (1:5, 1:4); col = (1:5, 1:4)
        n1, n2 = 5, 4
        cx = SVector(-1.0, 2.0, -1.0); cy = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(Fill(_interlace((cx, cy), Val(1)), n1, n2))
        cs = (Fill(cx, n1, n2), Fill(cy, n1, n2))
        @test build(st, row, col) == _star_oracle(cs, Val(1), row, col)
    end

    @testset "asymmetric values, row=col=(1:6,1:5)" begin
        row = (1:6, 1:5); col = (1:6, 1:5)
        n1, n2 = 6, 5
        cx = SVector(0.1, 1.7, -2.3); cy = SVector(-0.5, 3.1, 0.8)
        st = StarStencil{1}(Fill(_interlace((cx, cy), Val(1)), n1, n2))
        cs = (Fill(cx, n1, n2), Fill(cy, n1, n2))
        @test build(st, row, col) == _star_oracle(cs, Val(1), row, col)
    end

    @testset "L=2, row=col=(1:4,1:4)" begin
        row = (1:4, 1:4); col = (1:4, 1:4)
        n1, n2 = 4, 4
        cx = SVector(0.1, 0.2, 0.3, 0.4, 0.5); cy = cx
        st = StarStencil{2}(Fill(_interlace((cx, cy), Val(2)), n1, n2))
        cs = (Fill(cx, n1, n2), Fill(cy, n1, n2))
        @test build(st, row, col) == _star_oracle(cs, Val(2), row, col)
    end

    @testset "Float32, row=col=(1:4,1:4)" begin
        row = (1:4, 1:4); col = (1:4, 1:4)
        cx = SVector(-1f0, 2f0, -1f0); cy = SVector(-1f0, 2f0, -1f0)
        st = StarStencil{1}(Fill(_interlace((cx, cy), Val(1)), 4, 4))
        J = build(st, row, col)
        @test eltype(J) == Float32
        @test J == _star_oracle((Fill(cx, 4, 4), Fill(cy, 4, 4)), Val(1), row, col)
    end

    @testset "free diagonal (Helmholtz-style), row=col=(1:5,1:4)" begin
        # Off-diagonals from per-axis stencils with ZERO centers; the diagonal
        # is a free coefficient d0 — impossible in the old summed-center format.
        row = (1:5, 1:4); col = (1:5, 1:4)
        n1, n2 = 5, 4
        cx = SVector(-1.0, 0.0, -1.0); cy = SVector(-1.0, 0.0, -1.0)
        d0 = 4.7
        # Interlaced (N=2, L=1): (axis2,-1), (axis1,-1), diagonal, (axis1,+1), (axis2,+1).
        iv = SVector(cy[1], cx[1], d0, cx[3], cy[3])
        st = StarStencil{1}(Fill(iv, n1, n2))
        Σ = _star_oracle((Fill(cx, n1, n2), Fill(cy, n1, n2)), Val(1), row, col)
        Σ += build(LinearStencil{1}(SUnitRange(0, 0), Fill(SVector(d0), n1, n2)), row, col)
        @test build(st, row, col) == Σ
    end

    @testset "col ⊋ row in axis 1, row=(2:6,1:4), col=(1:7,1:4)" begin
        row = (2:6, 1:4); col = (1:7, 1:4)
        n1, n2 = 7, 4   # term/coef arrays sized to col
        cx = SVector(-1.0, 2.0, -1.0); cy = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(Fill(_interlace((cx, cy), Val(1)), n1, n2))
        cs = (Fill(cx, n1, n2), Fill(cy, n1, n2))
        @test build(st, row, col) == _star_oracle(cs, Val(1), row, col)
    end

    @testset "col ⊋ row in axis 2, row=(1:5,2:4), col=(1:5,1:5)" begin
        row = (1:5, 2:4); col = (1:5, 1:5)
        n1, n2 = 5, 5
        cx = SVector(-1.0, 2.0, -1.0); cy = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(Fill(_interlace((cx, cy), Val(1)), n1, n2))
        cs = (Fill(cx, n1, n2), Fill(cy, n1, n2))
        @test build(st, row, col) == _star_oracle(cs, Val(1), row, col)
    end
end

@testset "StarStencil 3-D vs sum(LinearStencils)" begin
    @testset "Laplacian, row=col=(1:3,1:3,1:3)" begin
        row = (1:3, 1:3, 1:3); col = (1:3, 1:3, 1:3)
        c = SVector(-1.0, 2.0, -1.0); cs = (c, c, c)
        st = StarStencil{1}(Fill(_interlace(cs, Val(1)), 3, 3, 3))
        @test build(st, row, col) == _star_oracle(ntuple(_ -> Fill(c, 3, 3, 3), 3), Val(1), row, col)
    end

    @testset "asymmetric values, row=col=(1:3,1:3,1:3)" begin
        row = (1:3, 1:3, 1:3); col = (1:3, 1:3, 1:3)
        cx = SVector(0.1, 1.0, -2.0); cy = SVector(-0.5, 2.5, 0.8); cz = SVector(0.3, -1.4, 0.6)
        st = StarStencil{1}(Fill(_interlace((cx, cy, cz), Val(1)), 3, 3, 3))
        cs = (Fill(cx, 3, 3, 3), Fill(cy, 3, 3, 3), Fill(cz, 3, 3, 3))
        @test build(st, row, col) == _star_oracle(cs, Val(1), row, col)
    end

    @testset "unequal lengths, row=col=(1:4,1:3,1:2)" begin
        row = (1:4, 1:3, 1:2); col = (1:4, 1:3, 1:2)
        c = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(Fill(_interlace((c, c, c), Val(1)), 4, 3, 2))
        @test build(st, row, col) == _star_oracle(ntuple(_ -> Fill(c, 4, 3, 2), 3), Val(1), row, col)
    end

    @testset "col ⊋ row in axis 3, row=(1:3,1:3,2:4), col=(1:3,1:3,1:5)" begin
        row = (1:3, 1:3, 2:4); col = (1:3, 1:3, 1:5)
        c = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(Fill(_interlace((c, c, c), Val(1)), 3, 3, 5))
        @test build(st, row, col) == _star_oracle(ntuple(_ -> Fill(c, 3, 3, 5), 3), Val(1), row, col)
    end
end

@testset "StarStencil 2L > length(row[d]) guard" begin
    # 1-D, L=2, length(row[1]) = 3 → 2L=4 > 3, rejected (via the linear guard).
    st = StarStencil{2}(Fill(SVector(0.0, 0.0, 0.0, 0.0, 0.0), 3))
    @test_throws ArgumentError build(st, (1:3,), (1:3,))

    # 2-D, L=2 (M=2NL+1=9): violated in axis 2 only.
    st2 = StarStencil{2}(Fill(SVector(ntuple(_ -> 0.0, 9)...), 5, 3))
    @test_throws ArgumentError build(st2, (1:5, 1:3), (1:5, 1:3))  # axis 2: 2L=4 > 3
end

@testset "StarStencil build = update!(assemble(...), ...)" begin
    row = (1:4, 1:4); col = (1:4, 1:4)
    c = SVector(-1.0, 2.0, -1.0)
    st = StarStencil{1}(Fill(_interlace((c, c), Val(1)), 4, 4))
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

        term = Fill(SVector(-1.0, 2.0, -1.0), 5)
        st_star = StarStencil{1}(term)
        @test AccessStyle(st_star) === ColumnAccess()
        @test AccessStyle(typeof(st_star)) === ColumnAccess()
        @test st_star isa AbstractStencil
        @test st_star isa AbstractStencil{ColumnAccess}
    end

    @testset "explicit ColumnAccess ctor" begin
        st_lin = LinearStencil{1}(ColumnAccess, SUnitRange(0, 1),
                                  Fill(SVector(-1.0, 1.0), 5))
        @test AccessStyle(st_lin) === ColumnAccess()
        term = Fill(SVector(-1.0, 2.0, -1.0), 5)
        st_star = StarStencil{1}(ColumnAccess, term)
        @test AccessStyle(st_star) === ColumnAccess()
    end

    @testset "explicit RowAccess ctor constructs but is unassemblable on CSC" begin
        st_lin = LinearStencil{1}(RowAccess, SUnitRange(0, 1),
                                  Fill(SVector(-1.0, 1.0), 5))
        @test AccessStyle(st_lin) === RowAccess()
        @test st_lin isa AbstractStencil{RowAccess}
        @test_throws MethodError assemble(st_lin, (1:5,), (1:5,))
        @test_throws MethodError build(st_lin, (1:5,), (1:5,))

        term = Fill(SVector(-1.0, 2.0, -1.0), 5)
        st_star = StarStencil{1}(RowAccess, term)
        @test AccessStyle(st_star) === RowAccess()
        @test st_star isa AbstractStencil{RowAccess}
        @test_throws MethodError assemble(st_star, (1:5,), (1:5,))
        @test_throws MethodError build(st_star, (1:5,), (1:5,))
    end

    @testset "_as_linear propagates S (RowAccess)" begin
        term = Fill(SVector(-1.0, 2.0, -1.0), 5)
        st_star = StarStencil{1}(RowAccess, term)
        ln = StencilAssembly._as_linear(st_star)
        @test ln isa LinearStencil
        @test AccessStyle(ln) === RowAccess()
    end

    @testset "RowAccess 2-D StarStencil is unassemblable on CSC" begin
        n1, n2 = 5, 4
        c = SVector(-1.0, 2.0, -1.0)
        st = StarStencil{1}(RowAccess, Fill(_interlace((c, c), Val(1)), n1, n2))
        @test_throws MethodError assemble(st, (1:n1, 1:n2), (1:n1, 1:n2))
    end
end

@testset "Stencil narrowing → assemble" begin
    @testset "as_linear → LinearStencil → assemble" begin
        n = 6
        # SoA: one scalar coefficient per offset; narrowing interlaces them.
        terms = (Fill(1.0, n), Fill(-4.0, n), Fill(3.0, n))
        st = Stencil(ColumnAccess, (-2ê₁, -ê₁, ô), terms)
        ln = as_linear(st)
        @test ln isa LinearStencil{1, -2, 3, Float64, <:Any, ColumnAccess}
        ref = fill(SVector(1.0, -4.0, 3.0), n)
        @test ln.term == ref
        @test build(ln, (1:n,), (1:n,)) ==
              build(LinearStencil{1}(SUnitRange(-2, 0), ref), (1:n,), (1:n,))
    end

    @testset "as_star → StarStencil → assemble" begin
        n1, n2 = 5, 4
        terms = (Fill(-1.0, n1, n2), Fill(-1.0, n1, n2), Fill(4.0, n1, n2),
                 Fill(-1.0, n1, n2), Fill(-1.0, n1, n2))
        st = Stencil(ColumnAccess, (-ê₂, -ê₁, ô, ê₁, ê₂), terms)
        ss = as_star(st)
        @test ss isa StarStencil{1, 2, 5, Float64, <:Any, ColumnAccess}
        ref = fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), n1, n2)
        @test ss.term == ref
        @test build(ss, (1:n1, 1:n2), (1:n1, 1:n2)) ==
              build(StarStencil{1}(ref), (1:n1, 1:n2), (1:n1, 1:n2))
    end
end
