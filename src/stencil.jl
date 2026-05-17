"""
    LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}

A variable-coefficient stencil aligned with mesh dimension `D`. For column `c`
at mesh position `p_c` and offset `Δ_k`, the matrix entry lands on the row at
mesh position `p_c − Δ_k` and carries coefficient `coefs[k][p_c]` — i.e., coefs
are anchored at the **column** mesh position (see `AGENTS.md`).

# Type parameters

- `D::Int` — mesh dimension on which the stencil acts (1-based). Must satisfy
  `1 ≤ D ≤ N`; enforced at construction.
- `K::Int` — number of stencil terms.
- `T` — shared element type of every coef array.
- `N` — coef-array dimensionality; matches the row/col
  `NTuple{N, AbstractUnitRange{Int}}` at assembly time.
- `C` — concrete tuple type of the coef containers; inferred from the
  constructor call. Heterogeneous containers (e.g. `Fill` + `Vector`) are fine
  as long as they share `eltype` and `ndims`.

# Fields

- `offsets::NTuple{K,Int}` — strictly descending 1-D offsets along dim `D`.
- `coefs::C` — `NTuple{K,<:AbstractArray{T,N}}` of coefficient arrays.

# Construction

    LinearStencil{D}(offsets, coefs)

Inner constructor (well-typed path) validates `D ≥ 1`, `D ≤ N`, and
strict-descending offsets via `issorted(offsets; lt = >=)`. The shared
`eltype`/`ndims` of `coefs` are enforced by the method signature.

A catch-all outer constructor (`LinearStencil{D}(::Tuple, ::Tuple)`) reports
friendly errors when the inputs are ill-typed (length mismatch, non-`Int`
offsets, non-`AbstractArray` coefs, mixed `eltype`, or mixed `ndims`).

# Examples

```julia
using FillArrays
n = 5

# Forward x-difference, constant coefs via Fill.
forward_x = LinearStencil{1}((1, 0), (Fill(1.0, n), Fill(-1.0, n)))
J = build(forward_x, (1:n,), (1:n,))    # one-shot assemble + update!

# Variable-coef: density-weighted gradient ψ[i] = (φ[i] − φ[i−1]) / ρ[i].
ρ = rand(n)
grad = LinearStencil{1}((0, -1), (1 ./ ρ, -1 ./ ρ))
```

See [`assemble`](@ref), [`update!`](@ref), [`build`](@ref) for matrix
construction.
"""
struct LinearStencil{D,K,T,N,C<:NTuple{K,AbstractArray{T,N}}}
    offsets::NTuple{K,Int}
    coefs::C

    function LinearStencil{D}(
        offsets::NTuple{K,Int},
        coefs::NTuple{K,AbstractArray{T,N}},
    ) where {D,K,T,N}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        D <= N || throw(ArgumentError(
            "stencil dimension D=$D exceeds coef-array dimension N=$N"))
        issorted(offsets; lt = >=) || throw(ArgumentError(
            "offsets must be strictly descending (got $offsets); required for SparseMatrixCSC rowval-per-column sortedness"))
        new{D,K,T,N,typeof(coefs)}(offsets, coefs)
    end
end

# Catch-all outer constructor: friendly errors for ill-typed inputs. Only fires
# when the inner method's NTuple{K,AbstractArray{T,N}} signature does not match
# (mixed eltype, mixed ndims, non-array elements, length mismatch, etc.).
function LinearStencil{D}(offsets::Tuple, coefs::Tuple) where {D}
    length(offsets) == length(coefs) || throw(ArgumentError(
        "offsets has length $(length(offsets)) but coefs has length $(length(coefs))"))
    all(o -> o isa Int, offsets) || throw(ArgumentError(
        "offsets must be a tuple of Int (got $(map(typeof, offsets)))"))
    all(c -> c isa AbstractArray, coefs) || throw(ArgumentError(
        "each coef must be an AbstractArray (got $(map(typeof, coefs)))"))
    Ts = map(eltype, coefs)
    all(==(first(Ts)), Ts) || throw(ArgumentError(
        "all coefs must share the same eltype (got $Ts)"))
    Ns = map(ndims, coefs)
    all(==(first(Ns)), Ns) || throw(ArgumentError(
        "all coefs must share the same ndims (got $Ns)"))
    throw(ArgumentError("LinearStencil could not be constructed; coefs = $coefs"))
end

"""
    _pattern!(rowval, colptr, offsets, row, col)

1-D pattern kernel. `K` outer loops, one per offset; each loop sweeps the
c-range where offset `k` produces an in-range row:

    c_lo_k = max(first(col), rmin + offsets[k])
    c_hi_k = min(last(col), rmax + offsets[k])

Four phases — clear → count → cumsum → fill+restore — using the counting-
sort-CSC pattern, where `colptr` doubles as the per-column "next free slot"
counter during fill and is restored to a proper CSC offset table by a
shift-right pass at the end.

# CSC sortedness

Strict-descending `offsets` give `r_k = c − offsets[k]` ascending in `k`.
Processing `k = 1, …, K` in order means for any column the writes happen
in k-ascending → row-ascending — the `SparseMatrixCSC` invariant — without
a sort.

# Allocation

`Vector{Int}` `rowval` is `resize!`d once to the exact final length; no
`push!`, no scratch buffer.

# Invariant under interruption

Mid-fill, `colptr[j]` temporarily stores the next-free-slot pointer for
column `j`, not the column start. The Phase-4 shift-right restores the
CSC offset table. If the kernel is interrupted between Phase 3 and Phase
4, `colptr` is left in this intermediate state. Single-threaded use only.
"""
function _pattern!(
    rowval::Vector{Int},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {K}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)
    n = length(col)
    # Phase 1 — count per-column emissions into colptr[2:n+1].
    fill!(view(colptr, 2:n+1), 0)
    for k in 1:K
        c_lo = max(cmin, rmin + offsets[k])
        c_hi = min(cmax, rmax + offsets[k])
        for c in c_lo:c_hi
            colptr[c - cmin + 2] += 1
        end
    end
    # Phase 2 — cumsum into colptr (colptr[1] = 1 set by caller).
    colptr[1] = 1
    for j in 2:n+1
        colptr[j] += colptr[j-1]
    end
    # Phase 3 — fill rowval; mutate colptr as the per-column slot tracker.
    resize!(rowval, colptr[n+1] - 1)
    for k in 1:K
        c_lo = max(cmin, rmin + offsets[k])
        c_hi = min(cmax, rmax + offsets[k])
        for c in c_lo:c_hi
            cc = c - cmin + 1
            rowval[colptr[cc]] = c - offsets[k] - rmin + 1
            colptr[cc] += 1
        end
    end
    # Phase 4 — restore colptr to the CSC offset table by shifting right by 1.
    for j in n+1:-1:2
        colptr[j] = colptr[j-1]
    end
    colptr[1] = 1
    return
end

"""
    _fill!(nzval, colptr, offsets, coefs, row, col)

1-D fill kernel. Same `K`-outer-loops shape as [`_pattern!`](@ref): for each
offset `k`, sweep its c-range and write `nzval[slot] = coefs[k][c]`, using
`colptr` as the per-column slot tracker (mutated in place, then restored
by a shift-right pass at the end).

Allocation-free apart from whatever `getindex` on the user's coef arrays
costs (`Vector`, `Fill`, `OffsetArray` etc. are O(1)).

Carries the same single-threaded / mid-fill caveat as `_pattern!` — `colptr`
is briefly inconsistent during the fill phase and restored at the end.
"""
function _fill!(
    nzval::AbstractVector{T},
    colptr::Vector{Int},
    offsets::NTuple{K,Int},
    coefs::NTuple{K,AbstractArray{T,1}},
    row::AbstractUnitRange{Int},
    col::AbstractUnitRange{Int},
) where {K,T}
    rmin, rmax = first(row), last(row)
    cmin, cmax = first(col), last(col)
    n = length(col)
    # Fill — mutate colptr as the slot tracker.
    for k in 1:K
        c_lo = max(cmin, rmin + offsets[k])
        c_hi = min(cmax, rmax + offsets[k])
        for c in c_lo:c_hi
            cc = c - cmin + 1
            nzval[colptr[cc]] = coefs[k][c]
            colptr[cc] += 1
        end
    end
    # Restore colptr by shifting right by 1.
    for j in n+1:-1:2
        colptr[j] = colptr[j-1]
    end
    colptr[1] = 1
    return
end

"""
    assemble(st::LinearStencil{1,K,T,1},
             row::NTuple{1,AbstractUnitRange{Int}},
             col::NTuple{1,AbstractUnitRange{Int}}) -> SparseMatrixCSC{T,Int}

Build the sparsity pattern (`colptr`, `rowval`) of the operator induced by
`st` between the row and column index sets, and allocate `nzval`
**uninitialised**. Call `update!` to populate `nzval` before using the matrix
— or use [`build`](@ref) to do both in one shot.

`row` and `col` are interpreted on a *shared* integer mesh; for a column at
position `c ∈ col[1]` and offset `Δ_k`, the row position is `c − Δ_k`, and the
entry is emitted iff that position lies in `row[1]`.

Dispatch pins `D = 1` and `N = 1`; misuse turns into `MethodError`. The
`D ≤ N` invariant at the `LinearStencil` constructor catches the most common
misuse case (e.g. `LinearStencil{2}` with 1-D coefs) earlier with a friendly
`ArgumentError`.
"""
function assemble(
    st::LinearStencil{1,K,T,1},
    row::NTuple{1,AbstractUnitRange{Int}},
    col::NTuple{1,AbstractUnitRange{Int}},
) where {K,T}
    m, n = length(row[1]), length(col[1])
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    rowval = Int[]
    _pattern!(rowval, colptr, st.offsets, row[1], col[1])
    nzval = Vector{T}(undef, length(rowval))
    SparseMatrixCSC{T,Int}(m, n, colptr, rowval, nzval)
end

"""
    update!(mat::SparseMatrixCSC{T,Int}, st::LinearStencil{1,K,T,1},
            row::NTuple{1,AbstractUnitRange{Int}},
            col::NTuple{1,AbstractUnitRange{Int}}) -> mat

Write `mat.nzval` in place by re-walking `row`/`col` with `st`. `mat` must
have been produced by a matching `assemble(st, row, col)` call (same stencil,
same row/col) so its `colptr`/`rowval` align with the kernel's sweep.

Dispatch pins `D = 1` and `N = 1`; misuse turns into `MethodError`.
"""
function update!(
    mat::SparseMatrixCSC{T,Int},
    st::LinearStencil{1,K,T,1},
    row::NTuple{1,AbstractUnitRange{Int}},
    col::NTuple{1,AbstractUnitRange{Int}},
) where {T,K}
    _fill!(mat.nzval, mat.colptr, st.offsets, st.coefs, row[1], col[1])
    return mat
end

"""
    build(st::LinearStencil, row, col) -> SparseMatrixCSC

Equivalent to `update!(assemble(st, row, col), st, row, col)`. Use when you
don't need the pattern/values split — the returned matrix is fully populated
and ready to use.
"""
build(st::LinearStencil, row, col) = update!(assemble(st, row, col), st, row, col)
